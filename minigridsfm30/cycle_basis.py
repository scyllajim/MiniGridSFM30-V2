from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch


def build_cycle_basis_from_lines(
    n_bus: int,
    line_endpoints: List[Tuple[int, int]],
    branch_x: List[List[float]],
):
    """
    Build simple cycle basis for an undirected power grid graph.

    Args:
        n_bus:
            Number of buses.
        line_endpoints:
            List of (from_bus_pos, to_bus_pos), aligned with branch_ac index.
        branch_x:
            List of branch_ac features, aligned with branch_ac index.
            branch_x columns:
              [angmin, angmax, b_fr, b_to, r, x, rate]

    Returns:
        cycle_x:
            Tensor [n_cycle, 4], each row:
              [cycle_length, sum_r, sum_x, mean_rate]
        cycle_to_branch_edge_index:
            Tensor [2, n_links], rows [cycle_idx, branch_idx]
        cycle_to_branch_edge_attr:
            Tensor [n_links, 1], sign of branch orientation in the cycle.
        branch_to_cycle_edge_index:
            Tensor [2, n_links], rows [branch_idx, cycle_idx]
        branch_to_cycle_edge_attr:
            Tensor [n_links, 1], same sign.
    """
    g = nx.Graph()

    for b in range(n_bus):
        g.add_node(b)

    edge_to_branch: Dict[Tuple[int, int], int] = {}

    for br_idx, (u, v) in enumerate(line_endpoints):
        g.add_edge(u, v)
        edge_to_branch[(min(u, v), max(u, v))] = br_idx

    cycles = nx.cycle_basis(g)

    cycle_features = []
    c_src = []
    c_dst = []
    c_attr = []

    b_src = []
    b_dst = []
    b_attr = []

    branch_arr = np.array(branch_x, dtype=np.float32) if len(branch_x) else np.zeros((0, 7), dtype=np.float32)

    for c_idx, cyc in enumerate(cycles):
        if len(cyc) < 3:
            continue

        branch_ids = []
        signs = []

        # cycle_basis returns a bus sequence. Connect consecutive buses,
        # including the last back to the first.
        for k in range(len(cyc)):
            u = int(cyc[k])
            v = int(cyc[(k + 1) % len(cyc)])
            key = (min(u, v), max(u, v))

            if key not in edge_to_branch:
                continue

            br_idx = edge_to_branch[key]
            branch_ids.append(br_idx)

            # sign is +1 if traversing min->max, -1 otherwise.
            # This is a simple orientation convention.
            sign = 1.0 if u < v else -1.0
            signs.append(sign)

        if len(branch_ids) == 0:
            continue

        bx = branch_arr[branch_ids]
        r_sum = float(bx[:, 4].sum())
        x_sum = float(bx[:, 5].sum())
        rate_mean = float(bx[:, 6].mean()) if bx.shape[0] > 0 else 0.0
        cycle_len = float(len(branch_ids))

        cycle_features.append([
            cycle_len,
            r_sum,
            x_sum,
            rate_mean,
        ])

        real_c_idx = len(cycle_features) - 1

        for br_idx, sign in zip(branch_ids, signs):
            c_src.append(real_c_idx)
            c_dst.append(br_idx)
            c_attr.append([sign])

            b_src.append(br_idx)
            b_dst.append(real_c_idx)
            b_attr.append([sign])

    if len(cycle_features) == 0:
        cycle_x = torch.zeros((0, 4), dtype=torch.float32)
        cycle_to_branch_edge_index = torch.zeros((2, 0), dtype=torch.long)
        cycle_to_branch_edge_attr = torch.zeros((0, 1), dtype=torch.float32)
        branch_to_cycle_edge_index = torch.zeros((2, 0), dtype=torch.long)
        branch_to_cycle_edge_attr = torch.zeros((0, 1), dtype=torch.float32)
    else:
        cycle_x = torch.tensor(cycle_features, dtype=torch.float32)
        cycle_to_branch_edge_index = torch.tensor([c_src, c_dst], dtype=torch.long)
        cycle_to_branch_edge_attr = torch.tensor(c_attr, dtype=torch.float32)
        branch_to_cycle_edge_index = torch.tensor([b_src, b_dst], dtype=torch.long)
        branch_to_cycle_edge_attr = torch.tensor(b_attr, dtype=torch.float32)

    return (
        cycle_x,
        cycle_to_branch_edge_index,
        cycle_to_branch_edge_attr,
        branch_to_cycle_edge_index,
        branch_to_cycle_edge_attr,
    )
