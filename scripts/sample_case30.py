from __future__ import annotations

import argparse
import pickle
import random
import traceback
from pathlib import Path

import numpy as np
import pandapower as pp
import pandapower.networks as pn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def build_base_net(case_name: str):
    if case_name == "case30":
        return pn.case30()
    if case_name == "case_ieee30":
        return pn.case_ieee30()
    raise ValueError(f"Unknown case_name: {case_name}")


def perturb_loads(net, load_min: float, load_max: float):
    """
    Randomly scale load p/q.
    """
    if len(net.load) == 0:
        return

    scales = np.random.uniform(load_min, load_max, size=len(net.load))
    net.load["p_mw"] = net.load["p_mw"].astype(float).values * scales
    net.load["q_mvar"] = net.load["q_mvar"].astype(float).values * scales


def perturb_costs(net, cost_min: float, cost_max: float):
    """
    Randomly scale polynomial cost coefficients.
    """
    if not hasattr(net, "poly_cost") or len(net.poly_cost) == 0:
        return

    for col in ["cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2"]:
        if col in net.poly_cost.columns:
            scales = np.random.uniform(cost_min, cost_max, size=len(net.poly_cost))
            net.poly_cost[col] = net.poly_cost[col].astype(float).values * scales


def maybe_derate_lines(net, derate_prob: float, derate_min: float, derate_max: float):
    """
    Optional line rating derating.
    Keep default derate_prob=0 for first stable dataset.
    """
    if derate_prob <= 0 or len(net.line) == 0:
        return

    mask = np.random.rand(len(net.line)) < derate_prob
    factors = np.random.uniform(derate_min, derate_max, size=len(net.line))

    if "max_i_ka" in net.line.columns:
        vals = net.line["max_i_ka"].astype(float).values
        vals[mask] = vals[mask] * factors[mask]
        net.line["max_i_ka"] = vals


def run_opf(net, numba: bool = True):
    pp.runopp(
        net,
        verbose=False,
        numba=numba,
        suppress_warnings=True,
    )


def _copy_table(net, name: str):
    if hasattr(net, name):
        obj = getattr(net, name)
        if obj is not None:
            return obj.copy()
    return None


def export_sample(net, sample_id: int, feasible: bool, error_msg: str = ""):
    """
    Save only the tables needed to reconstruct a solved pandapower net.
    """
    input_table_names = [
        "bus",
        "line",
        "trafo",
        "gen",
        "ext_grid",
        "load",
        "shunt",
        "poly_cost",
    ]

    result_table_names = [
        "res_bus",
        "res_line",
        "res_trafo",
        "res_gen",
        "res_ext_grid",
        "res_load",
        "res_shunt",
    ]

    net_tables = {}
    for name in input_table_names:
        df = _copy_table(net, name)
        if df is not None:
            net_tables[name] = df

    res_tables = {}
    for name in result_table_names:
        df = _copy_table(net, name)
        if df is not None:
            res_tables[name] = df

    return {
        "sample_id": int(sample_id),
        "feasible": bool(feasible),
        "error_msg": error_msg,
        "sn_mva": float(getattr(net, "sn_mva", 100.0)),
        "f_hz": float(getattr(net, "f_hz", 50.0)),
        "res_cost": float(getattr(net, "res_cost", 0.0)) if feasible else 0.0,
        "net_tables": net_tables,
        "res_tables": res_tables,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--case", type=str, default="case30", choices=["case30", "case_ieee30"])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", type=str, default="data/raw/case30_1000_v2.pkl")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--load-min", type=float, default=0.95)
    parser.add_argument("--load-max", type=float, default=1.05)
    parser.add_argument("--cost-min", type=float, default=0.98)
    parser.add_argument("--cost-max", type=float, default=1.02)

    parser.add_argument("--derate-prob", type=float, default=0.0)
    parser.add_argument("--derate-min", type=float, default=0.8)
    parser.add_argument("--derate-max", type=float, default=1.0)

    parser.add_argument("--numba", action="store_true")
    parser.add_argument("--keep-failed", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    n_success = 0
    n_failed = 0

    print("=== MiniGridSFM30-v2 sampling ===")
    print("case:", args.case)
    print("n:", args.n)
    print("out:", args.out)
    print("load range:", args.load_min, args.load_max)
    print("cost range:", args.cost_min, args.cost_max)
    print("derate prob:", args.derate_prob)
    print()

    for i in range(args.n):
        net = build_base_net(args.case)

        try:
            perturb_loads(net, args.load_min, args.load_max)
            perturb_costs(net, args.cost_min, args.cost_max)
            maybe_derate_lines(net, args.derate_prob, args.derate_min, args.derate_max)

            run_opf(net, numba=args.numba)

            sample = export_sample(net, i, feasible=True)
            samples.append(sample)
            n_success += 1

        except Exception as e:
            n_failed += 1
            err = f"{type(e).__name__}: {str(e)}"

            if args.keep_failed:
                sample = export_sample(net, i, feasible=False, error_msg=err)
                samples.append(sample)

        if (i + 1) % 50 == 0 or i + 1 == args.n:
            print(f"[{i + 1}/{args.n}] success={n_success} failed={n_failed} saved={len(samples)}")

    dataset = {
        "case": args.case,
        "n_requested": args.n,
        "n_success": n_success,
        "n_failed": n_failed,
        "n_saved": len(samples),
        "seed": args.seed,
        "settings": {
            "load_min": args.load_min,
            "load_max": args.load_max,
            "cost_min": args.cost_min,
            "cost_max": args.cost_max,
            "derate_prob": args.derate_prob,
            "derate_min": args.derate_min,
            "derate_max": args.derate_max,
        },
        "sample_format": {
            "bus_y": ["theta_rad", "vm_pu"],
            "generator_y": ["pg_pu", "qg_pu"],
            "branch_ac_y": ["p_from_pu", "q_from_pu", "p_to_pu", "q_to_pu"],
            "note": "v2 saves pandapower tables and converts to GridSFM-style HeteroData using graph_builder.sample_to_heterodata",
        },
        "samples": samples,
    }

    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)

    print()
    print("=== Sampling finished ===")
    print("case:      ", args.case)
    print("requested: ", args.n)
    print("success:   ", n_success)
    print("failed:    ", n_failed)
    print("saved:     ", len(samples))
    print("out:       ", args.out)
    print()
    print("target format:")
    print("  bus_y       = [theta_rad, vm_pu]")
    print("  generator_y = [pg_pu, qg_pu]")
    print("  branch_y    = [p_from_pu, q_from_pu, p_to_pu, q_to_pu]")


if __name__ == "__main__":
    main()
