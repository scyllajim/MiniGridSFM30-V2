from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, SAGEConv, Linear


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GridSFM30(nn.Module):
    """
    MiniGridSFM30-v2 model.

    GridSFM-style 3-layer heterogeneous GNN.

    Node types:
      bus
      generator
      load
      branch_ac
      cycle

    Edge types:
      bus <-> generator
      bus <-> load
      bus <-> branch_ac
      cycle <-> branch_ac

    Outputs:
      bus_pred:       [num_bus, 2] = [theta_rad, vm_pu]
      generator_pred: [num_generator, 2] = [Pg_pu, Qg_pu]
      branch_pred:    [num_branch_ac, 4] = [p_from_pu, q_from_pu, p_to_pu, q_to_pu]
      feas_logit:     [num_graphs]
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # ----------------------------
        # Input encoders
        # ----------------------------
        self.bus_encoder = Linear(7, hidden_dim)
        self.generator_encoder = Linear(12, hidden_dim)
        self.load_encoder = Linear(2, hidden_dim)
        self.branch_ac_encoder = Linear(7, hidden_dim)
        self.cycle_encoder = Linear(4, hidden_dim)

        self.input_norm = nn.ModuleDict({
            "bus": nn.LayerNorm(hidden_dim),
            "generator": nn.LayerNorm(hidden_dim),
            "load": nn.LayerNorm(hidden_dim),
            "branch_ac": nn.LayerNorm(hidden_dim),
            "cycle": nn.LayerNorm(hidden_dim),
        })

        # ----------------------------
        # 3-layer heterogeneous GNN
        # Edge names are aligned with graph_builder.py
        # ----------------------------
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    # bus <-> generator
                    ("bus", "generator_link", "generator"):
                        SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    ("generator", "generator_link", "bus"):
                        SAGEConv((hidden_dim, hidden_dim), hidden_dim),

                    # bus <-> load
                    ("bus", "load_link", "load"):
                        SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    ("load", "load_link", "bus"):
                        SAGEConv((hidden_dim, hidden_dim), hidden_dim),

                    # bus <-> branch_ac
                    ("bus", "endpoint_of", "branch_ac"):
                        SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    ("branch_ac", "endpoint_of", "bus"):
                        SAGEConv((hidden_dim, hidden_dim), hidden_dim),

                    # cycle <-> branch_ac
                    ("cycle", "in_cycle", "branch_ac"):
                        SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                    ("branch_ac", "in_cycle", "cycle"):
                        SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                },
                aggr="sum",
            )

            self.convs.append(conv)

            self.norms.append(
                nn.ModuleDict({
                    "bus": nn.LayerNorm(hidden_dim),
                    "generator": nn.LayerNorm(hidden_dim),
                    "load": nn.LayerNorm(hidden_dim),
                    "branch_ac": nn.LayerNorm(hidden_dim),
                    "cycle": nn.LayerNorm(hidden_dim),
                })
            )

        # ----------------------------
        # Prediction heads
        # Official GridSFM internally uses bus.pred = [theta, V]
        # ----------------------------
        self.bus_head = MLP(hidden_dim, 2, hidden_dim)
        self.generator_head = MLP(hidden_dim, 2, hidden_dim)
        self.branch_ac_head = MLP(hidden_dim, 4, hidden_dim)

        # Graph-level feasibility head
        self.feas_head = MLP(hidden_dim, 1, hidden_dim)

    def _encode(self, data):
        x_dict = {}

        x_dict["bus"] = self.input_norm["bus"](
            self.bus_encoder(data["bus"].x)
        )

        x_dict["generator"] = self.input_norm["generator"](
            self.generator_encoder(data["generator"].x)
        )

        x_dict["load"] = self.input_norm["load"](
            self.load_encoder(data["load"].x)
        )

        x_dict["branch_ac"] = self.input_norm["branch_ac"](
            self.branch_ac_encoder(data["branch_ac"].x)
        )

        x_dict["cycle"] = self.input_norm["cycle"](
            self.cycle_encoder(data["cycle"].x)
        )

        return x_dict

    def _global_pool_bus(self, h_bus, data):
        """
        Simple graph-level pooling from bus embeddings.
        Works for both single graph and PyG batched HeteroData.
        """
        if hasattr(data["bus"], "batch"):
            batch = data["bus"].batch
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

            out = torch.zeros(
                num_graphs,
                h_bus.size(-1),
                dtype=h_bus.dtype,
                device=h_bus.device,
            )
            cnt = torch.zeros(
                num_graphs,
                dtype=h_bus.dtype,
                device=h_bus.device,
            )

            out.scatter_add_(0, batch.unsqueeze(-1).expand_as(h_bus), h_bus)
            cnt.scatter_add_(0, batch, torch.ones_like(batch, dtype=h_bus.dtype))
            out = out / cnt.clamp_min(1.0).unsqueeze(-1)
            return out

        return h_bus.mean(dim=0, keepdim=True)

    def forward(self, data):
        x_dict = self._encode(data)
        edge_index_dict = data.edge_index_dict

        # ----------------------------
        # Message passing
        # ----------------------------
        for conv, norm in zip(self.convs, self.norms):
            x_old = x_dict
            x_msg = conv(x_dict, edge_index_dict)

            x_new = {}
            for nt in x_old.keys():
                if nt in x_msg:
                    h = x_old[nt] + F.dropout(
                        F.relu(x_msg[nt]),
                        p=self.dropout,
                        training=self.training,
                    )
                else:
                    h = x_old[nt]

                x_new[nt] = norm[nt](h)

            x_dict = x_new

        # ----------------------------
        # Prediction
        # ----------------------------
        bus_pred = self.bus_head(x_dict["bus"])
        generator_pred = self.generator_head(x_dict["generator"])
        branch_ac_pred = self.branch_ac_head(x_dict["branch_ac"])

        # Graph-level feasibility
        h_global = self._global_pool_bus(x_dict["bus"], data)
        feas_logit = self.feas_head(h_global).squeeze(-1)

        return {
            "bus_pred": bus_pred,
            "generator_pred": generator_pred,
            "branch_ac_pred": branch_ac_pred,
            "feas_logit": feas_logit,
            "x_dict": x_dict,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
