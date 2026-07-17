from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

import pandapower as pp
import pandapower.networks as pn

from minigridsfm30.schema import SN_BASE_MVA
from minigridsfm30.cycle_basis import build_cycle_basis_from_lines


def _safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _safe_bool(x, default=True):
    try:
        if pd.isna(x):
            return bool(default)
        return bool(x)
    except Exception:
        return bool(default)


def _get_poly_cost(net, element: str, element_index: int):
    """
    Return quadratic cost coefficients cp2, cp1, cp0 for a pandapower element.

    pandapower poly_cost usually has:
      et, element, cp0_eur, cp1_eur_per_mw, cp2_eur_per_mw2
    """
    if not hasattr(net, "poly_cost") or len(net.poly_cost) == 0:
        return 0.0, 0.0, 0.0

    pc = net.poly_cost
    rows = pc[(pc["et"] == element) & (pc["element"] == element_index)]
    if len(rows) == 0:
        return 0.0, 0.0, 0.0

    r = rows.iloc[0]
    cp2 = _safe_float(r.get("cp2_eur_per_mw2", 0.0))
    cp1 = _safe_float(r.get("cp1_eur_per_mw", 0.0))
    cp0 = _safe_float(r.get("cp0_eur", 0.0))
    return cp2, cp1, cp0


def _online_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows with in_service=True if the column exists.

    This is important for killgen/outage-aware graph construction.
    Offline generators should not define PV bus type.
    """
    if df is None or len(df) == 0:
        return df
    if "in_service" not in df.columns:
        return df
    return df[df["in_service"].astype(bool)]


def _bus_type_code(net, bus_idx: int) -> float:
    """
    Rough MATPOWER-style bus type:
      1 = PQ
      2 = PV
      3 = slack/reference

    Only online ext_grid/gen rows should define slack/PV status.
    """
    if hasattr(net, "ext_grid") and len(net.ext_grid) > 0:
        eg = _online_table(net.ext_grid)
        if len(eg) > 0 and bus_idx in set(eg["bus"].astype(int).tolist()):
            return 3.0

    if hasattr(net, "gen") and len(net.gen) > 0:
        gen = _online_table(net.gen)
        if len(gen) > 0 and bus_idx in set(gen["bus"].astype(int).tolist()):
            return 2.0

    return 1.0


def _aggregate_load_by_bus(net, bus_to_pos: Dict[int, int], n_bus: int, sn_mva: float):
    pd_pu = np.zeros(n_bus, dtype=np.float32)
    qd_pu = np.zeros(n_bus, dtype=np.float32)

    for _, row in net.load.iterrows():
        if not _safe_bool(row.get("in_service", True), True):
            continue
        b = int(row["bus"])
        if b not in bus_to_pos:
            continue
        pos = bus_to_pos[b]
        scaling = _safe_float(row.get("scaling", 1.0), 1.0)
        pd_pu[pos] += _safe_float(row.get("p_mw", 0.0)) * scaling / sn_mva
        qd_pu[pos] += _safe_float(row.get("q_mvar", 0.0)) * scaling / sn_mva

    return pd_pu, qd_pu


def _line_pu_params(net, line_row, bus_from_idx: int, sn_mva: float):
    """
    Convert pandapower line physical parameters approximately to p.u.

    Zbase = Vbase^2 / Sbase, using from-bus nominal voltage.
    r_pu = r_ohm / Zbase
    x_pu = x_ohm / Zbase

    b_pu is approximated from line capacitance:
    B = 2*pi*f*C
    b_pu = B * Zbase
    """
    length = _safe_float(line_row.get("length_km", 1.0), 1.0)
    r_ohm = _safe_float(line_row.get("r_ohm_per_km", 0.0)) * length
    x_ohm = _safe_float(line_row.get("x_ohm_per_km", 0.0)) * length
    c_nf = _safe_float(line_row.get("c_nf_per_km", 0.0)) * length

    vn_kv = _safe_float(net.bus.loc[bus_from_idx, "vn_kv"], 1.0)
    z_base = (vn_kv ** 2) / sn_mva

    r_pu = r_ohm / z_base if z_base > 0 else 0.0
    x_pu = x_ohm / z_base if z_base > 0 else 0.0

    f_hz = _safe_float(getattr(net, "f_hz", 50.0), 50.0)
    c_f = c_nf * 1e-9
    b_si = 2.0 * math.pi * f_hz * c_f
    b_pu_total = b_si * z_base
    b_fr_pu = 0.5 * b_pu_total
    b_to_pu = 0.5 * b_pu_total

    max_i_ka = _safe_float(line_row.get("max_i_ka", 0.0))
    parallel = max(1.0, _safe_float(line_row.get("parallel", 1.0), 1.0))
    rate_a_mva = math.sqrt(3.0) * vn_kv * max_i_ka * parallel
    rate_a_pu = rate_a_mva / sn_mva if sn_mva > 0 else 0.0

    return r_pu, x_pu, b_fr_pu, b_to_pu, rate_a_pu


def build_case30_net():
    """
    Return a fresh pandapower case30 network.
    """
    return pn.case30()


def run_ac_opf(net, verbose: bool = False):
    """
    Run AC OPF on a pandapower network.
    """
    pp.runopp(net, verbose=verbose, numba=True, suppress_warnings=True)
    return net


def net_to_heterodata(net, require_solution: bool = True) -> HeteroData:
    """
    Convert a solved pandapower case30 network to GridSFM-style HeteroData.

    Node types:
      bus
      generator
      load
      shunt
      branch_ac
      branch_tr
      cycle

    Key design:
      pandapower line -> branch_ac node
      bus --endpoint_of--> branch_ac
      branch_ac --endpoint_of--> bus

    Generator feature layout, 12 dims:
      0  is_online
      1  reserved
      2  pmin_pu
      3  pmax_pu
      4  reserved
      5  qmin_pu
      6  qmax_pu
      7  vm_set_pu
      8  cp2
      9  cp1
      10 cp0
      11 is_ext_grid

    Important:
      Offline regular generators are kept as generator nodes, but is_online=0
      and effective p/q bounds are set to 0. This makes killgen/outage
      information explicit to the GNN without changing input dimension.
    """
    if require_solution:
        required = ["res_bus", "res_gen", "res_ext_grid", "res_line"]
        for name in required:
            if not hasattr(net, name) or len(getattr(net, name)) == 0:
                raise RuntimeError(f"net is missing solved result table: {name}")

    data = HeteroData()

    sn_mva = _safe_float(getattr(net, "sn_mva", SN_BASE_MVA), SN_BASE_MVA)
    data.sn_mva = sn_mva

    # ----------------------------
    # Bus nodes
    # ----------------------------
    bus_indices = list(net.bus.index.astype(int))
    bus_to_pos = {b: i for i, b in enumerate(bus_indices)}
    n_bus = len(bus_indices)

    pd_pu, qd_pu = _aggregate_load_by_bus(net, bus_to_pos, n_bus, sn_mva)

    bus_x = []
    bus_y = []

    for b in bus_indices:
        row = net.bus.loc[b]
        pos = bus_to_pos[b]

        base_kv = _safe_float(row.get("vn_kv", 0.0))
        btype = _bus_type_code(net, b)
        vmin = _safe_float(row.get("min_vm_pu", 0.95), 0.95)
        vmax = _safe_float(row.get("max_vm_pu", 1.05), 1.05)
        is_slack = 1.0 if btype == 3.0 else 0.0

        bus_x.append([
            base_kv / 100.0,
            btype,
            vmin,
            vmax,
            pd_pu[pos],
            qd_pu[pos],
            is_slack,
        ])

        if require_solution:
            rb = net.res_bus.loc[b]
            theta_rad = math.radians(_safe_float(rb.get("va_degree", 0.0)))
            vm_pu = _safe_float(rb.get("vm_pu", 1.0), 1.0)
            bus_y.append([theta_rad, vm_pu])

    data["bus"].x = torch.tensor(bus_x, dtype=torch.float32)
    if require_solution:
        data["bus"].y = torch.tensor(bus_y, dtype=torch.float32)

    # ----------------------------
    # Generator nodes
    # regular gen first, ext_grid appended after
    # ----------------------------
    gen_x = []
    gen_y = []
    gen_bus_pos = []

    for gen_idx, row in net.gen.iterrows():
        bus_idx = int(row["bus"])
        bus_pos = bus_to_pos[bus_idx]

        is_online = 1.0 if _safe_bool(row.get("in_service", True), True) else 0.0

        pmin_raw = _safe_float(row.get("min_p_mw", -1e3)) / sn_mva
        pmax_raw = _safe_float(row.get("max_p_mw", 1e3)) / sn_mva
        qmin_raw = _safe_float(row.get("min_q_mvar", -1e3)) / sn_mva
        qmax_raw = _safe_float(row.get("max_q_mvar", 1e3)) / sn_mva

        if is_online <= 0.0:
            pmin = 0.0
            pmax = 0.0
            qmin = 0.0
            qmax = 0.0
        else:
            pmin = pmin_raw
            pmax = pmax_raw
            qmin = qmin_raw
            qmax = qmax_raw

        vg = _safe_float(row.get("vm_pu", 1.0), 1.0)
        cp2, cp1, cp0 = _get_poly_cost(net, "gen", int(gen_idx))

        gen_x.append([
            is_online,
            0.0,
            pmin,
            pmax,
            0.0,
            qmin,
            qmax,
            vg,
            cp2,
            cp1,
            cp0,
            0.0,
        ])
        gen_bus_pos.append(bus_pos)

        if require_solution:
            if gen_idx in net.res_gen.index:
                rg = net.res_gen.loc[gen_idx]
                pg = _safe_float(rg.get("p_mw", 0.0)) / sn_mva
                qg = _safe_float(rg.get("q_mvar", 0.0)) / sn_mva
            else:
                pg = 0.0
                qg = 0.0

            if is_online <= 0.0:
                pg = 0.0
                qg = 0.0

            gen_y.append([pg, qg])

    for eg_idx, row in net.ext_grid.iterrows():
        bus_idx = int(row["bus"])
        bus_pos = bus_to_pos[bus_idx]

        is_online = 1.0 if _safe_bool(row.get("in_service", True), True) else 0.0

        pmin_raw = _safe_float(row.get("min_p_mw", -1e3)) / sn_mva
        pmax_raw = _safe_float(row.get("max_p_mw", 1e3)) / sn_mva
        qmin_raw = _safe_float(row.get("min_q_mvar", -1e3)) / sn_mva
        qmax_raw = _safe_float(row.get("max_q_mvar", 1e3)) / sn_mva

        if is_online <= 0.0:
            pmin = 0.0
            pmax = 0.0
            qmin = 0.0
            qmax = 0.0
        else:
            pmin = pmin_raw
            pmax = pmax_raw
            qmin = qmin_raw
            qmax = qmax_raw

        vg = _safe_float(row.get("vm_pu", 1.0), 1.0)
        cp2, cp1, cp0 = _get_poly_cost(net, "ext_grid", int(eg_idx))

        gen_x.append([
            is_online,
            0.0,
            pmin,
            pmax,
            0.0,
            qmin,
            qmax,
            vg,
            cp2,
            cp1,
            cp0,
            1.0,
        ])
        gen_bus_pos.append(bus_pos)

        if require_solution:
            if eg_idx in net.res_ext_grid.index:
                reg = net.res_ext_grid.loc[eg_idx]
                pg = _safe_float(reg.get("p_mw", 0.0)) / sn_mva
                qg = _safe_float(reg.get("q_mvar", 0.0)) / sn_mva
            else:
                pg = 0.0
                qg = 0.0

            if is_online <= 0.0:
                pg = 0.0
                qg = 0.0

            gen_y.append([pg, qg])

    data["generator"].x = torch.tensor(gen_x, dtype=torch.float32)
    if require_solution:
        data["generator"].y = torch.tensor(gen_y, dtype=torch.float32)

    gen_src = torch.arange(len(gen_bus_pos), dtype=torch.long)
    gen_dst = torch.tensor(gen_bus_pos, dtype=torch.long)
    data[("generator", "generator_link", "bus")].edge_index = torch.stack([gen_src, gen_dst], dim=0)
    data[("bus", "generator_link", "generator")].edge_index = torch.stack([gen_dst, gen_src], dim=0)

    # ----------------------------
    # Load nodes
    # ----------------------------
    load_x = []
    load_bus_pos = []

    for _, row in net.load.iterrows():
        if not _safe_bool(row.get("in_service", True), True):
            continue

        bus_idx = int(row["bus"])
        bus_pos = bus_to_pos[bus_idx]
        scaling = _safe_float(row.get("scaling", 1.0), 1.0)
        p_pu = _safe_float(row.get("p_mw", 0.0)) * scaling / sn_mva
        q_pu = _safe_float(row.get("q_mvar", 0.0)) * scaling / sn_mva

        load_x.append([p_pu, q_pu])
        load_bus_pos.append(bus_pos)

    if len(load_x) == 0:
        data["load"].x = torch.zeros((0, 2), dtype=torch.float32)
        data[("load", "load_link", "bus")].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data[("bus", "load_link", "load")].edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        data["load"].x = torch.tensor(load_x, dtype=torch.float32)
        load_src = torch.arange(len(load_bus_pos), dtype=torch.long)
        load_dst = torch.tensor(load_bus_pos, dtype=torch.long)
        data[("load", "load_link", "bus")].edge_index = torch.stack([load_src, load_dst], dim=0)
        data[("bus", "load_link", "load")].edge_index = torch.stack([load_dst, load_src], dim=0)

    # ----------------------------
    # Shunt nodes
    # first version: empty, but keep official-compatible type
    # ----------------------------
    data["shunt"].x = torch.zeros((0, 2), dtype=torch.float32)
    data[("shunt", "shunt_link", "bus")].edge_index = torch.zeros((2, 0), dtype=torch.long)
    data[("bus", "shunt_link", "shunt")].edge_index = torch.zeros((2, 0), dtype=torch.long)

    # ----------------------------
    # Branch AC nodes
    # pandapower line -> branch_ac node
    # ----------------------------
    branch_x = []
    branch_y = []
    line_endpoints = []
    bus_to_branch_src = []
    bus_to_branch_dst = []
    bus_to_branch_attr = []
    branch_to_bus_src = []
    branch_to_bus_dst = []
    branch_to_bus_attr = []

    for line_idx, row in net.line.iterrows():
        if not _safe_bool(row.get("in_service", True), True):
            continue

        # Use a compact branch index after filtering offline lines.
        #
        # The original pandapower line index may contain gaps, and skipping
        # an offline line must not leave gaps in branch_ac node positions.
        # Endpoint edges must always reference [0, num_branch_ac).
        br_pos = len(branch_x)

        fb = int(row["from_bus"])
        tb = int(row["to_bus"])
        fpos = bus_to_pos[fb]
        tpos = bus_to_pos[tb]

        r_pu, x_pu, b_fr_pu, b_to_pu, rate_a_pu = _line_pu_params(net, row, fb, sn_mva)

        line_endpoints.append((fpos, tpos))

        angmin = -math.pi
        angmax = math.pi

        branch_x.append([
            angmin,
            angmax,
            b_fr_pu,
            b_to_pu,
            r_pu,
            x_pu,
            rate_a_pu,
        ])

        if require_solution:
            rl = net.res_line.loc[line_idx]
            p_from = _safe_float(rl.get("p_from_mw", 0.0)) / sn_mva
            q_from = _safe_float(rl.get("q_from_mvar", 0.0)) / sn_mva
            p_to = _safe_float(rl.get("p_to_mw", 0.0)) / sn_mva
            q_to = _safe_float(rl.get("q_to_mvar", 0.0)) / sn_mva
            branch_y.append([p_from, q_from, p_to, q_to])

        # bus -> branch endpoint edges
        # endpoint sign: -1 for from-side, +1 for to-side
        bus_to_branch_src.extend([fpos, tpos])
        bus_to_branch_dst.extend([br_pos, br_pos])
        bus_to_branch_attr.extend([[-1.0], [1.0]])

        # branch -> bus endpoint edges
        branch_to_bus_src.extend([br_pos, br_pos])
        branch_to_bus_dst.extend([fpos, tpos])
        branch_to_bus_attr.extend([[-1.0], [1.0]])

    data["branch_ac"].x = torch.tensor(branch_x, dtype=torch.float32)
    if require_solution:
        data["branch_ac"].y = torch.tensor(branch_y, dtype=torch.float32)

    data[("bus", "endpoint_of", "branch_ac")].edge_index = torch.tensor(
        [bus_to_branch_src, bus_to_branch_dst], dtype=torch.long
    )
    data[("bus", "endpoint_of", "branch_ac")].edge_attr = torch.tensor(
        bus_to_branch_attr, dtype=torch.float32
    )

    data[("branch_ac", "endpoint_of", "bus")].edge_index = torch.tensor(
        [branch_to_bus_src, branch_to_bus_dst], dtype=torch.long
    )
    data[("branch_ac", "endpoint_of", "bus")].edge_attr = torch.tensor(
        branch_to_bus_attr, dtype=torch.float32
    )

    # ----------------------------
    # Branch transformer nodes
    # first version: empty, but keep official-compatible type
    # ----------------------------
    data["branch_tr"].x = torch.zeros((0, 7), dtype=torch.float32)
    data[("bus", "endpoint_of", "branch_tr")].edge_index = torch.zeros((2, 0), dtype=torch.long)
    data[("branch_tr", "endpoint_of", "bus")].edge_index = torch.zeros((2, 0), dtype=torch.long)

    # ----------------------------
    # Cycle nodes
    # GridSFM-style cycle basis over bus-branch topology.
    # ----------------------------
    (
        cycle_x,
        cycle_to_branch_ei,
        cycle_to_branch_ea,
        branch_to_cycle_ei,
        branch_to_cycle_ea,
    ) = build_cycle_basis_from_lines(
        n_bus=n_bus,
        line_endpoints=line_endpoints,
        branch_x=branch_x,
    )

    data["cycle"].x = cycle_x

    data[("cycle", "in_cycle", "branch_ac")].edge_index = cycle_to_branch_ei
    data[("cycle", "in_cycle", "branch_ac")].edge_attr = cycle_to_branch_ea

    data[("branch_ac", "in_cycle", "cycle")].edge_index = branch_to_cycle_ei
    data[("branch_ac", "in_cycle", "cycle")].edge_attr = branch_to_cycle_ea

    data[("cycle", "in_cycle", "branch_tr")].edge_index = torch.zeros((2, 0), dtype=torch.long)
    data[("branch_tr", "in_cycle", "cycle")].edge_index = torch.zeros((2, 0), dtype=torch.long)

    # ----------------------------
    # Graph-level labels
    # ----------------------------
    data.feasible = torch.tensor([1.0], dtype=torch.float32)
    if require_solution and hasattr(net, "res_cost"):
        data.res_cost = torch.tensor([_safe_float(net.res_cost, 0.0)], dtype=torch.float32)
    else:
        data.res_cost = torch.tensor([0.0], dtype=torch.float32)

    return data


def build_solved_case30_heterodata(verbose: bool = False) -> HeteroData:
    net = build_case30_net()
    run_ac_opf(net, verbose=verbose)
    return net_to_heterodata(net, require_solution=True)


def sample_to_net(sample: dict):
    """
    Reconstruct a pandapower net from a saved sample dict.

    sample contains:
      net_tables: original pandapower input tables
      res_tables: OPF result tables
      res_cost
      sn_mva
      f_hz
    """
    net = pp.create_empty_network(
        sn_mva=float(sample.get("sn_mva", SN_BASE_MVA)),
        f_hz=float(sample.get("f_hz", 50.0)),
    )

    net_tables = sample["net_tables"]
    res_tables = sample.get("res_tables", {})

    # Restore core input tables.
    for name, df in net_tables.items():
        setattr(net, name, df.copy())

    # Restore result tables.
    for name, df in res_tables.items():
        setattr(net, name, df.copy())

    net.res_cost = float(sample.get("res_cost", 0.0))
    return net


def sample_to_heterodata(sample: dict) -> HeteroData:
    """
    Convert a saved sample dict into GridSFM-style HeteroData.

    Feasible samples have OPF labels and participate in full supervised/physics losses.
    Infeasible samples do not have valid OPF labels; they are converted with dummy labels
    so that batching works, but later losses should mask them out except feasibility BCE.
    """
    feasible = bool(sample.get("feasible", True))

    net = sample_to_net(sample)
    data = net_to_heterodata(net, require_solution=feasible)

    data.sample_id = int(sample.get("sample_id", -1))
    data.feasible = torch.tensor([1.0 if feasible else 0.0], dtype=torch.float32)

    if "perturb_mode" in sample:
        data.perturb_mode = sample.get("perturb_mode")


    if "merged_index" in sample:
        data.merged_index = int(sample.get("merged_index", -1))

    if not feasible:
        data["bus"].y = torch.zeros((data["bus"].x.size(0), 2), dtype=torch.float32)
        data["generator"].y = torch.zeros((data["generator"].x.size(0), 2), dtype=torch.float32)
        data["branch_ac"].y = torch.zeros((data["branch_ac"].x.size(0), 4), dtype=torch.float32)
        data.res_cost = torch.tensor([0.0], dtype=torch.float32)

    return data
