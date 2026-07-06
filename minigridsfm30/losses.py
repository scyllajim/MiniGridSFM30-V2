from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


PER_ELEM_CAP = 100.0


def _tanh_capped_mse(err: torch.Tensor, cap: float = PER_ELEM_CAP):
    """
    GridSFM-style robust squared error:
      cap * tanh(err^2 / cap)
    """
    sq = err.pow(2)
    return cap * torch.tanh(sq / cap)


def _mean_or_zero(x: torch.Tensor):
    if x.numel() == 0:
        return x.new_zeros(())
    return x.mean()


def _num_graphs_from_data(data):
    if hasattr(data, "num_graphs") and data.num_graphs is not None:
        return int(data.num_graphs)

    if hasattr(data["bus"], "batch"):
        batch = data["bus"].batch
        if batch.numel() > 0:
            return int(batch.max().item()) + 1

    return 1




def _sn_mva_per_graph(data, n_graphs: int, device, dtype):
    sn = getattr(data, "sn_mva", 100.0)

    if torch.is_tensor(sn):
        sn = sn.to(device=device, dtype=dtype).view(-1)
        if sn.numel() == 1:
            return sn.repeat(n_graphs)
        return sn[:n_graphs]

    if isinstance(sn, (list, tuple)):
        sn = torch.tensor(sn, device=device, dtype=dtype).view(-1)
        if sn.numel() == 1:
            return sn.repeat(n_graphs)
        return sn[:n_graphs]

    return torch.full((n_graphs,), float(sn), device=device, dtype=dtype)


def _get_batch(data, node_type: str):
    n = data[node_type].x.size(0)
    dev = data[node_type].x.device

    if hasattr(data[node_type], "batch"):
        return data[node_type].batch

    return torch.zeros(n, dtype=torch.long, device=dev)


def _scatter_sum(values: torch.Tensor, index: torch.Tensor, dim_size: int):
    """
    values: [N] or [N, C]
    index:  [N]
    return: [dim_size] or [dim_size, C]
    """
    if values.dim() == 1:
        out = values.new_zeros(dim_size)
        out.scatter_add_(0, index, values)
        return out

    out = values.new_zeros(dim_size, values.size(-1))
    out.scatter_add_(0, index.unsqueeze(-1).expand_as(values), values)
    return out


def state_supervised_loss(out, data):
    """
    Supervised loss for:
      bus.pred = [theta, V]
      generator.pred = [Pg, Qg]
      branch_ac.pred = [p_from, q_from, p_to, q_to]
    """
    bus_pred = out["bus_pred"]
    gen_pred = out["generator_pred"]
    br_pred = out["branch_ac_pred"]

    bus_y = data["bus"].y
    gen_y = data["generator"].y
    br_y = data["branch_ac"].y

    # theta uses angle wrapping
    theta_err = torch.atan2(
        torch.sin(bus_pred[:, 0] - bus_y[:, 0]),
        torch.cos(bus_pred[:, 0] - bus_y[:, 0]),
    )

    v_err = bus_pred[:, 1] - bus_y[:, 1]

    pg_err = gen_pred[:, 0] - gen_y[:, 0]
    qg_err = gen_pred[:, 1] - gen_y[:, 1]

    br_err = br_pred - br_y

    loss_theta = _mean_or_zero(_tanh_capped_mse(theta_err))
    loss_v = _mean_or_zero(_tanh_capped_mse(v_err))
    loss_pg = _mean_or_zero(_tanh_capped_mse(pg_err))
    loss_qg = _mean_or_zero(_tanh_capped_mse(qg_err))

    loss_branch_p = _mean_or_zero(
        _tanh_capped_mse(br_err[:, [0, 2]])
    )
    loss_branch_q = _mean_or_zero(
        _tanh_capped_mse(br_err[:, [1, 3]])
    )

    metrics = {
        "loss_theta": float(loss_theta.detach().cpu()),
        "loss_v": float(loss_v.detach().cpu()),
        "loss_pg": float(loss_pg.detach().cpu()),
        "loss_qg": float(loss_qg.detach().cpu()),
        "loss_branch_p": float(loss_branch_p.detach().cpu()),
        "loss_branch_q": float(loss_branch_q.detach().cpu()),

        "theta_mae": float(theta_err.abs().mean().detach().cpu()),
        "v_mae": float(v_err.abs().mean().detach().cpu()),
        "pg_mae": float(pg_err.abs().mean().detach().cpu()),
        "qg_mae": float(qg_err.abs().mean().detach().cpu()),
        "branch_p_mae": float(br_err[:, [0, 2]].abs().mean().detach().cpu()),
        "branch_q_mae": float(br_err[:, [1, 3]].abs().mean().detach().cpu()),
    }

    return (
        loss_theta,
        loss_v,
        loss_pg,
        loss_qg,
        loss_branch_p,
        loss_branch_q,
        metrics,
    )


def graph_power_balance_loss(out, data):
    """
    Graph-level active/reactive balance.

    This is weaker than full KCL, but useful:
      sum(Pg) - sum(Pd) - total_line_loss ≈ 0

    For Q:
      sum(Qg) - sum(Qd) - total_line_Q_loss ≈ 0

    branch_ac_pred:
      [p_from, q_from, p_to, q_to]
    line loss per branch:
      p_from + p_to
      q_from + q_to
    """
    n_graphs = _num_graphs_from_data(data)

    gen_batch = _get_batch(data, "generator")
    load_batch = _get_batch(data, "load")
    branch_batch = _get_batch(data, "branch_ac")

    gen_pred = out["generator_pred"]
    br_pred = out["branch_ac_pred"]

    pg_g = _scatter_sum(gen_pred[:, 0], gen_batch, n_graphs)
    qg_g = _scatter_sum(gen_pred[:, 1], gen_batch, n_graphs)

    if data["load"].x.size(0) > 0:
        pd_g = _scatter_sum(data["load"].x[:, 0], load_batch, n_graphs)
        qd_g = _scatter_sum(data["load"].x[:, 1], load_batch, n_graphs)
    else:
        pd_g = pg_g.new_zeros(n_graphs)
        qd_g = qg_g.new_zeros(n_graphs)

    p_loss_branch = br_pred[:, 0] + br_pred[:, 2]
    q_loss_branch = br_pred[:, 1] + br_pred[:, 3]

    if br_pred.size(0) > 0:
        p_loss_g = _scatter_sum(p_loss_branch, branch_batch, n_graphs)
        q_loss_g = _scatter_sum(q_loss_branch, branch_batch, n_graphs)
    else:
        p_loss_g = pg_g.new_zeros(n_graphs)
        q_loss_g = qg_g.new_zeros(n_graphs)

    res_p = pg_g - pd_g - p_loss_g
    res_q = qg_g - qd_g - q_loss_g

    loss_balance_p = res_p.pow(2).mean()
    loss_balance_q = res_q.pow(2).mean()

    metrics = {
        "loss_balance_p": float(loss_balance_p.detach().cpu()),
        "loss_balance_q": float(loss_balance_q.detach().cpu()),
        "balance_p_mae": float(res_p.abs().mean().detach().cpu()),
        "balance_q_mae": float(res_q.abs().mean().detach().cpu()),
    }

    return loss_balance_p, loss_balance_q, metrics


def bus_kcl_loss(out, data):
    """
    Bus-level KCL residual.

    For each bus:
      flow_out_P - (Pgen - Pload) ≈ 0
      flow_out_Q - (Qgen - Qload) ≈ 0

    We use branch_ac_pred endpoint convention:
      p_from/q_from: flow leaving from_bus into branch
      p_to/q_to:     flow leaving to_bus into branch

    graph_builder created:
      ("branch_ac", "endpoint_of", "bus").edge_index
      edge_attr = -1 for from-side, +1 for to-side
    """
    bus_x = data["bus"].x
    dev = bus_x.device
    n_bus = bus_x.size(0)

    gen_pred = out["generator_pred"]
    br_pred = out["branch_ac_pred"]

    pgen_bus = torch.zeros(n_bus, device=dev, dtype=bus_x.dtype)
    qgen_bus = torch.zeros(n_bus, device=dev, dtype=bus_x.dtype)

    if data["generator"].x.size(0) > 0:
        g2b = data[("generator", "generator_link", "bus")].edge_index
        gen_idx = g2b[0]
        bus_idx = g2b[1]
        pgen_bus.scatter_add_(0, bus_idx, gen_pred[gen_idx, 0])
        qgen_bus.scatter_add_(0, bus_idx, gen_pred[gen_idx, 1])

    pload_bus = torch.zeros(n_bus, device=dev, dtype=bus_x.dtype)
    qload_bus = torch.zeros(n_bus, device=dev, dtype=bus_x.dtype)

    if data["load"].x.size(0) > 0:
        l2b = data[("load", "load_link", "bus")].edge_index
        load_idx = l2b[0]
        bus_idx = l2b[1]
        pload_bus.scatter_add_(0, bus_idx, data["load"].x[load_idx, 0])
        qload_bus.scatter_add_(0, bus_idx, data["load"].x[load_idx, 1])

    flow_p_bus = torch.zeros(n_bus, device=dev, dtype=bus_x.dtype)
    flow_q_bus = torch.zeros(n_bus, device=dev, dtype=bus_x.dtype)

    b2bus = data[("branch_ac", "endpoint_of", "bus")].edge_index
    b2bus_attr = data[("branch_ac", "endpoint_of", "bus")].edge_attr.squeeze(-1)

    if b2bus.size(1) > 0:
        branch_idx = b2bus[0]
        bus_idx = b2bus[1]
        sign = b2bus_attr

        # sign < 0 means from-side, sign > 0 means to-side
        p_endpoint = torch.where(sign < 0, br_pred[branch_idx, 0], br_pred[branch_idx, 2])
        q_endpoint = torch.where(sign < 0, br_pred[branch_idx, 1], br_pred[branch_idx, 3])

        flow_p_bus.scatter_add_(0, bus_idx, p_endpoint)
        flow_q_bus.scatter_add_(0, bus_idx, q_endpoint)

    resid_p = flow_p_bus - (pgen_bus - pload_bus)
    resid_q = flow_q_bus - (qgen_bus - qload_bus)

    loss_kcl_p = resid_p.pow(2).mean()
    loss_kcl_q = resid_q.pow(2).mean()

    metrics = {
        "loss_kcl_p": float(loss_kcl_p.detach().cpu()),
        "loss_kcl_q": float(loss_kcl_q.detach().cpu()),
        "kcl_p_mae": float(resid_p.abs().mean().detach().cpu()),
        "kcl_q_mae": float(resid_q.abs().mean().detach().cpu()),
    }

    return loss_kcl_p, loss_kcl_q, metrics


def generation_cost_loss(out, data):
    """
    Cost loss based on predicted Pg.

    generator.x columns:
      cp2 = x[:, 8]
      cp1 = x[:, 9]
      cp0 = x[:, 10]

    Cost is computed in the same approximate scale as stored coefficients.
    Because Pg is p.u., convert Pg_pu to MW using data.sn_mva.
    """
    n_graphs = _num_graphs_from_data(data)
    gen_batch = _get_batch(data, "generator")

    gx = data["generator"].x
    sn_g = _sn_mva_per_graph(
        data,
        n_graphs,
        device=out["generator_pred"].device,
        dtype=out["generator_pred"].dtype,
    )
    sn_gen = sn_g[gen_batch]
    pg_pred_mw = out["generator_pred"][:, 0] * sn_gen

    cp2 = gx[:, 8]
    cp1 = gx[:, 9]
    cp0 = gx[:, 10]

    per_gen_cost = cp2 * pg_pred_mw.pow(2) + cp1 * pg_pred_mw + cp0
    pred_cost = _scatter_sum(per_gen_cost, gen_batch, n_graphs)

    if hasattr(data, "res_cost"):
        true_cost = data.res_cost.view(-1).to(pred_cost.device).to(pred_cost.dtype)
        if true_cost.numel() != pred_cost.numel():
            true_cost = true_cost[: pred_cost.numel()]
    else:
        true_cost = pred_cost.detach()

    loss_cost = (
        torch.log1p(pred_cost.clamp_min(0.0))
        - torch.log1p(true_cost.clamp_min(0.0))
    ).pow(2).mean()

    cost_mape = (
        (pred_cost - true_cost).abs()
        / true_cost.abs().clamp_min(1e-6)
    ).mean()

    metrics = {
        "loss_cost": float(loss_cost.detach().cpu()),
        "cost_mape": float(cost_mape.detach().cpu()),
    }

    return loss_cost, metrics


def feasibility_loss(out, data):
    """
    BCE on feasibility logit.
    For current successful OPF samples, feasible = 1.
    Later, when we add infeasible perturbations, this will become meaningful.
    """
    if "feas_logit" not in out:
        z = data["bus"].x.new_zeros(())
        return z, {"loss_feas": 0.0}

    feas_logit = out["feas_logit"]

    if hasattr(data, "feasible"):
        label = data.feasible.view(-1).to(feas_logit.device).to(feas_logit.dtype)
        if label.numel() != feas_logit.numel():
            label = label[: feas_logit.numel()]
    else:
        label = torch.ones_like(feas_logit)

    loss = F.binary_cross_entropy_with_logits(feas_logit, label)

    metrics = {
        "loss_feas": float(loss.detach().cpu()),
        "feas_prob_mean": float(torch.sigmoid(feas_logit).mean().detach().cpu()),
    }

    return loss, metrics


def compute_loss(
    out,
    data,
    lambda_theta: float = 1.0,
    lambda_v: float = 1.0,
    lambda_pg: float = 1.0,
    lambda_qg: float = 1.0,
    lambda_branch_p: float = 1.0,
    lambda_branch_q: float = 1.0,
    lambda_balance_p: float = 0.1,
    lambda_balance_q: float = 0.1,
    lambda_kcl_p: float = 1.0,
    lambda_kcl_q: float = 1.0,
    lambda_cost: float = 0.1,
    lambda_feas: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    GridSFM-style composite loss for MiniGridSFM30-v2.

    Total:
      supervised theta/V/Pg/Qg
      + branch flow P/Q
      + graph-level P/Q balance
      + bus-level KCL P/Q
      + cost log-MSE
      + feasibility BCE

    This is intentionally close to GridSFM's public compute_loss structure,
    but simplified for pandapower case30.
    """
    (
        loss_theta,
        loss_v,
        loss_pg,
        loss_qg,
        loss_branch_p,
        loss_branch_q,
        m_state,
    ) = state_supervised_loss(out, data)

    loss_balance_p, loss_balance_q, m_balance = graph_power_balance_loss(out, data)
    loss_kcl_p, loss_kcl_q, m_kcl = bus_kcl_loss(out, data)
    loss_cost, m_cost = generation_cost_loss(out, data)
    loss_feas, m_feas = feasibility_loss(out, data)

    total = (
        lambda_theta * loss_theta
        + lambda_v * loss_v
        + lambda_pg * loss_pg
        + lambda_qg * loss_qg
        + lambda_branch_p * loss_branch_p
        + lambda_branch_q * loss_branch_q
        + lambda_balance_p * torch.log1p(loss_balance_p)
        + lambda_balance_q * torch.log1p(loss_balance_q)
        + lambda_kcl_p * torch.log1p(loss_kcl_p)
        + lambda_kcl_q * torch.log1p(loss_kcl_q)
        + lambda_cost * loss_cost
        + lambda_feas * loss_feas
    )

    metrics = {}
    metrics.update(m_state)
    metrics.update(m_balance)
    metrics.update(m_kcl)
    metrics.update(m_cost)
    metrics.update(m_feas)

    metrics["loss_total"] = float(total.detach().cpu())

    return total, metrics
