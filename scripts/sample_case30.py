from __future__ import annotations

import argparse
import pickle
import random
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


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def perturb_loads(net, load_min: float, load_max: float, load_jitter: float):
    """
    Pure loads mode.

    Similar idea to GridSFM loads perturbation:
      global scale * per-load jitter
    """
    params = {
        "mode": "loads",
        "load_min": load_min,
        "load_max": load_max,
        "load_jitter": load_jitter,
        "n_load": int(len(net.load)),
    }

    if len(net.load) == 0:
        params["applied"] = False
        return params

    sf = float(np.random.uniform(load_min, load_max))
    jitter_low = 1.0 - load_jitter
    jitter_high = 1.0 + load_jitter
    jitters = np.random.uniform(jitter_low, jitter_high, size=len(net.load))
    scales = sf * jitters

    net.load["p_mw"] = net.load["p_mw"].astype(float).values * scales
    net.load["q_mvar"] = net.load["q_mvar"].astype(float).values * scales

    params.update({
        "applied": True,
        "global_scale": sf,
        "scale_min": float(scales.min()),
        "scale_max": float(scales.max()),
        "scale_mean": float(scales.mean()),
    })
    return params


def perturb_costs(net, cost_min: float, cost_max: float):
    """
    Pure costs mode.

    Current implementation scales polynomial cost coefficients.
    This is simpler than official GridSFM cost shuffling, but keeps the same purpose:
    changing economic dispatch conditions.
    """
    params = {
        "mode": "costs",
        "cost_min": cost_min,
        "cost_max": cost_max,
        "n_poly_cost": int(len(net.poly_cost)) if hasattr(net, "poly_cost") else 0,
    }

    if not hasattr(net, "poly_cost") or len(net.poly_cost) == 0:
        params["applied"] = False
        return params

    col_stats = {}

    for col in ["cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2"]:
        if col in net.poly_cost.columns:
            scales = np.random.uniform(cost_min, cost_max, size=len(net.poly_cost))
            net.poly_cost[col] = net.poly_cost[col].astype(float).values * scales
            col_stats[col] = {
                "scale_min": float(scales.min()),
                "scale_max": float(scales.max()),
                "scale_mean": float(scales.mean()),
            }

    params.update({
        "applied": True,
        "columns": col_stats,
    })
    return params


def perturb_derate(net, derate_prob: float, derate_min: float, derate_max: float):
    """
    Pure derate mode.

    Randomly derate line current limits.
    """
    params = {
        "mode": "derate",
        "derate_prob": derate_prob,
        "derate_min": derate_min,
        "derate_max": derate_max,
        "n_line": int(len(net.line)),
    }

    if derate_prob <= 0 or len(net.line) == 0:
        params["applied"] = False
        params["n_derated"] = 0
        return params

    if "max_i_ka" not in net.line.columns:
        params["applied"] = False
        params["n_derated"] = 0
        params["reason"] = "net.line has no max_i_ka column"
        return params

    mask = np.random.rand(len(net.line)) < derate_prob
    factors = np.random.uniform(derate_min, derate_max, size=len(net.line))

    vals = net.line["max_i_ka"].astype(float).values
    vals[mask] = vals[mask] * factors[mask]
    net.line["max_i_ka"] = vals

    selected = net.line.index[mask].astype(int).tolist()

    params.update({
        "applied": True,
        "n_derated": int(mask.sum()),
        "derated_line_indices": selected,
        "factor_min": float(factors[mask].min()) if mask.sum() > 0 else None,
        "factor_max": float(factors[mask].max()) if mask.sum() > 0 else None,
        "factor_mean": float(factors[mask].mean()) if mask.sum() > 0 else None,
    })
    return params


def perturb_killgen(net, killgen_n: int, killgen_keep_min: int):
    """
    Pure killgen mode.

    For case30, this is intentionally conservative:
    only regular generators in net.gen are considered.
    ext_grid is kept online.
    """
    params = {
        "mode": "killgen",
        "killgen_n": killgen_n,
        "killgen_keep_min": killgen_keep_min,
        "n_gen": int(len(net.gen)),
    }

    if len(net.gen) == 0:
        params["applied"] = False
        params["n_killed"] = 0
        return params

    if "in_service" not in net.gen.columns:
        net.gen["in_service"] = True

    active = [
        int(idx)
        for idx, row in net.gen.iterrows()
        if bool(row.get("in_service", True))
    ]

    max_kill = max(0, len(active) - killgen_keep_min)
    n_kill = min(int(killgen_n), max_kill)

    if n_kill <= 0:
        params["applied"] = False
        params["n_killed"] = 0
        params["active_before"] = active
        return params

    killed = random.sample(active, n_kill)
    net.gen.loc[killed, "in_service"] = False

    params.update({
        "applied": True,
        "active_before": active,
        "killed_gen_indices": killed,
        "n_killed": int(n_kill),
        "note": "ext_grid is kept online",
    })
    return params


def perturb_vsqueeze(net, vsqueeze_prob: float, vsqueeze_eps: float, vm_margin: float):
    """
    Pure vsqueeze mode.

    Tighten voltage bounds on a subset of buses:
      min_vm_pu += random small amount
      max_vm_pu -= random small amount
    """
    params = {
        "mode": "vsqueeze",
        "vsqueeze_prob": vsqueeze_prob,
        "vsqueeze_eps": vsqueeze_eps,
        "vm_margin": vm_margin,
        "n_bus": int(len(net.bus)),
    }

    if len(net.bus) == 0 or vsqueeze_prob <= 0:
        params["applied"] = False
        params["n_squeezed"] = 0
        return params

    if "min_vm_pu" not in net.bus.columns:
        net.bus["min_vm_pu"] = 0.95

    if "max_vm_pu" not in net.bus.columns:
        net.bus["max_vm_pu"] = 1.05

    mask = np.random.rand(len(net.bus)) < vsqueeze_prob
    selected = net.bus.index[mask].astype(int).tolist()

    changes = []

    for b in selected:
        old_min = float(net.bus.loc[b, "min_vm_pu"])
        old_max = float(net.bus.loc[b, "max_vm_pu"])

        inc = float(np.random.rand() * vsqueeze_eps)
        dec = float(np.random.rand() * vsqueeze_eps)

        new_min = old_min + inc
        new_max = old_max - dec

        if new_max - new_min < vm_margin:
            center = 0.5 * (old_min + old_max)
            new_min = center - 0.5 * vm_margin
            new_max = center + 0.5 * vm_margin

        net.bus.loc[b, "min_vm_pu"] = new_min
        net.bus.loc[b, "max_vm_pu"] = new_max

        changes.append({
            "bus": int(b),
            "old_min": old_min,
            "old_max": old_max,
            "new_min": float(new_min),
            "new_max": float(new_max),
        })

    params.update({
        "applied": True,
        "n_squeezed": len(changes),
        "squeezed_buses": changes,
    })
    return params


def apply_perturbation(net, args):
    """
    Apply exactly one pure perturbation mode.

    mode:
      base     : no perturbation
      loads    : load scaling only
      costs    : cost scaling only
      derate   : line derating only
      killgen  : generator outage only
      vsqueeze : voltage bound squeezing only
      mixed    : backward-compatible old behavior
    """
    mode = args.mode

    if mode == "base":
        return {
            "mode": "base",
            "applied": False,
            "note": "unperturbed base case",
        }

    if mode == "loads":
        return perturb_loads(net, args.load_min, args.load_max, args.load_jitter)

    if mode == "costs":
        return perturb_costs(net, args.cost_min, args.cost_max)

    if mode == "derate":
        return perturb_derate(net, args.derate_prob, args.derate_min, args.derate_max)

    if mode == "killgen":
        return perturb_killgen(net, args.killgen_n, args.killgen_keep_min)

    if mode == "vsqueeze":
        return perturb_vsqueeze(net, args.vsqueeze_prob, args.vsqueeze_eps, args.vm_margin)

    if mode == "mixed":
        params = {
            "mode": "mixed",
            "loads": perturb_loads(net, args.load_min, args.load_max, args.load_jitter),
            "costs": perturb_costs(net, args.cost_min, args.cost_max),
        }

        if args.derate_prob > 0:
            params["derate"] = perturb_derate(
                net,
                args.derate_prob,
                args.derate_min,
                args.derate_max,
            )

        return params

    raise ValueError(f"Unknown perturbation mode: {mode}")


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


def export_sample(
    net,
    sample_id: int,
    feasible: bool,
    perturb_mode: str,
    perturb_params: dict,
    error_msg: str = "",
):
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
        "perturb_mode": str(perturb_mode),
        "perturb_params": _jsonable(perturb_params),
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

    parser.add_argument(
        "--mode",
        type=str,
        default="mixed",
        choices=["base", "loads", "costs", "derate", "killgen", "vsqueeze", "mixed"],
        help="Pure perturbation mode. Use mixed for backward-compatible old behavior.",
    )

    parser.add_argument("--load-min", type=float, default=0.95)
    parser.add_argument("--load-max", type=float, default=1.05)
    parser.add_argument("--load-jitter", type=float, default=0.10)

    parser.add_argument("--cost-min", type=float, default=0.98)
    parser.add_argument("--cost-max", type=float, default=1.02)

    parser.add_argument("--derate-prob", type=float, default=0.10)
    parser.add_argument("--derate-min", type=float, default=0.8)
    parser.add_argument("--derate-max", type=float, default=1.0)

    parser.add_argument("--killgen-n", type=int, default=1)
    parser.add_argument("--killgen-keep-min", type=int, default=1)

    parser.add_argument("--vsqueeze-prob", type=float, default=0.10)
    parser.add_argument("--vsqueeze-eps", type=float, default=0.005)
    parser.add_argument("--vm-margin", type=float, default=0.02)

    parser.add_argument("--numba", action="store_true")
    parser.add_argument("--keep-failed", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    n_success = 0
    n_failed = 0

    print("=== MiniGridSFM30-v2 pure-mode sampling ===")
    print("case:", args.case)
    print("mode:", args.mode)
    print("n:", args.n)
    print("out:", args.out)
    print("seed:", args.seed)
    print("keep_failed:", args.keep_failed)
    print()

    for i in range(args.n):
        net = build_base_net(args.case)
        perturb_params = {}

        try:
            perturb_params = apply_perturbation(net, args)

            run_opf(net, numba=args.numba)

            sample = export_sample(
                net=net,
                sample_id=i,
                feasible=True,
                perturb_mode=args.mode,
                perturb_params=perturb_params,
            )
            samples.append(sample)
            n_success += 1

        except Exception as e:
            n_failed += 1
            err = f"{type(e).__name__}: {str(e)}"

            if args.keep_failed:
                sample = export_sample(
                    net=net,
                    sample_id=i,
                    feasible=False,
                    perturb_mode=args.mode,
                    perturb_params=perturb_params,
                    error_msg=err,
                )
                samples.append(sample)

        if (i + 1) % 50 == 0 or i + 1 == args.n:
            print(
                f"[{i + 1}/{args.n}] "
                f"mode={args.mode} success={n_success} "
                f"failed={n_failed} saved={len(samples)}"
            )

    dataset = {
        "case": args.case,
        "mode": args.mode,
        "n_requested": args.n,
        "n_success": n_success,
        "n_failed": n_failed,
        "n_saved": len(samples),
        "seed": args.seed,
        "settings": {
            "mode": args.mode,
            "load_min": args.load_min,
            "load_max": args.load_max,
            "load_jitter": args.load_jitter,
            "cost_min": args.cost_min,
            "cost_max": args.cost_max,
            "derate_prob": args.derate_prob,
            "derate_min": args.derate_min,
            "derate_max": args.derate_max,
            "killgen_n": args.killgen_n,
            "killgen_keep_min": args.killgen_keep_min,
            "vsqueeze_prob": args.vsqueeze_prob,
            "vsqueeze_eps": args.vsqueeze_eps,
            "vm_margin": args.vm_margin,
        },
        "sample_format": {
            "bus_y": ["theta_rad", "vm_pu"],
            "generator_y": ["pg_pu", "qg_pu"],
            "branch_ac_y": ["p_from_pu", "q_from_pu", "p_to_pu", "q_to_pu"],
            "metadata": ["perturb_mode", "perturb_params", "feasible", "res_cost"],
            "note": "v2 pure-mode samples save pandapower tables and convert to GridSFM-style HeteroData using graph_builder.sample_to_heterodata",
        },
        "samples": samples,
    }

    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)

    print()
    print("=== Sampling finished ===")
    print("case:       ", args.case)
    print("mode:       ", args.mode)
    print("requested:  ", args.n)
    print("success:    ", n_success)
    print("failed:     ", n_failed)
    print("saved:      ", len(samples))
    print("out:        ", args.out)
    print()
    print("target format:")
    print("  bus_y        = [theta_rad, vm_pu]")
    print("  generator_y  = [pg_pu, qg_pu]")
    print("  branch_ac_y  = [p_from_pu, q_from_pu, p_to_pu, q_to_pu]")
    print("  metadata     = perturb_mode, perturb_params, feasible, res_cost")


if __name__ == "__main__":
    main()