from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import pandapower as pp

from minigridsfm30.graph_builder import sample_to_net


def safe_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def rel_err(a, b, eps=1e-9):
    a = safe_float(a)
    b = safe_float(b)
    return abs(a - b) / max(abs(a), abs(b), eps)


def abs_err(a, b):
    return abs(safe_float(a) - safe_float(b))


def try_resolve(sample, numba: bool = True):
    net = sample_to_net(sample)

    # 删除旧结果，避免误读旧 res_*。
    for name in [
        "res_bus",
        "res_line",
        "res_trafo",
        "res_gen",
        "res_ext_grid",
        "res_load",
        "res_shunt",
    ]:
        if hasattr(net, name):
            try:
                delattr(net, name)
            except Exception:
                pass

    if hasattr(net, "res_cost"):
        try:
            delattr(net, "res_cost")
        except Exception:
            pass

    pp.runopp(
        net,
        verbose=False,
        numba=numba,
        suppress_warnings=True,
    )

    return net


def compare_tables(old_net, new_net):
    out = {}

    # cost
    old_cost = safe_float(getattr(old_net, "res_cost", 0.0))
    new_cost = safe_float(getattr(new_net, "res_cost", 0.0))
    out["cost_old"] = old_cost
    out["cost_new"] = new_cost
    out["cost_abs_err"] = abs_err(old_cost, new_cost)
    out["cost_rel_err"] = rel_err(old_cost, new_cost)

    # bus voltage
    if hasattr(old_net, "res_bus") and hasattr(new_net, "res_bus"):
        common = old_net.res_bus.index.intersection(new_net.res_bus.index)
        if len(common) > 0:
            old_vm = old_net.res_bus.loc[common, "vm_pu"].to_numpy(dtype=float)
            new_vm = new_net.res_bus.loc[common, "vm_pu"].to_numpy(dtype=float)

            old_va = old_net.res_bus.loc[common, "va_degree"].to_numpy(dtype=float)
            new_va = new_net.res_bus.loc[common, "va_degree"].to_numpy(dtype=float)

            out["bus_vm_mae"] = float(np.mean(np.abs(old_vm - new_vm)))
            out["bus_va_mae_deg"] = float(np.mean(np.abs(old_va - new_va)))
        else:
            out["bus_vm_mae"] = None
            out["bus_va_mae_deg"] = None
    else:
        out["bus_vm_mae"] = None
        out["bus_va_mae_deg"] = None

    # generator output
    if hasattr(old_net, "res_gen") and hasattr(new_net, "res_gen"):
        common = old_net.res_gen.index.intersection(new_net.res_gen.index)
        if len(common) > 0:
            old_p = old_net.res_gen.loc[common, "p_mw"].to_numpy(dtype=float)
            new_p = new_net.res_gen.loc[common, "p_mw"].to_numpy(dtype=float)
            old_q = old_net.res_gen.loc[common, "q_mvar"].to_numpy(dtype=float)
            new_q = new_net.res_gen.loc[common, "q_mvar"].to_numpy(dtype=float)

            out["gen_p_mae_mw"] = float(np.mean(np.abs(old_p - new_p)))
            out["gen_q_mae_mvar"] = float(np.mean(np.abs(old_q - new_q)))
        else:
            out["gen_p_mae_mw"] = None
            out["gen_q_mae_mvar"] = None
    else:
        out["gen_p_mae_mw"] = None
        out["gen_q_mae_mvar"] = None

    # line flow
    if hasattr(old_net, "res_line") and hasattr(new_net, "res_line"):
        common = old_net.res_line.index.intersection(new_net.res_line.index)
        if len(common) > 0:
            old_pf = old_net.res_line.loc[common, "p_from_mw"].to_numpy(dtype=float)
            new_pf = new_net.res_line.loc[common, "p_from_mw"].to_numpy(dtype=float)
            old_qf = old_net.res_line.loc[common, "q_from_mvar"].to_numpy(dtype=float)
            new_qf = new_net.res_line.loc[common, "q_from_mvar"].to_numpy(dtype=float)

            out["line_p_from_mae_mw"] = float(np.mean(np.abs(old_pf - new_pf)))
            out["line_q_from_mae_mvar"] = float(np.mean(np.abs(old_qf - new_qf)))
        else:
            out["line_p_from_mae_mw"] = None
            out["line_q_from_mae_mvar"] = None
    else:
        out["line_p_from_mae_mw"] = None
        out["line_q_from_mae_mvar"] = None

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out-csv", type=str, default="reports/resolve_check.csv")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--numba", action="store_true")
    parser.add_argument("--cost-rel-tol", type=float, default=1e-5)
    parser.add_argument("--cost-abs-tol", type=float, default=1e-3)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    with open(args.data, "rb") as f:
        obj = pickle.load(f)

    samples = obj.get("samples", [])
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    old_feasible_count = 0
    old_infeasible_count = 0
    new_feasible_count = 0
    new_infeasible_count = 0
    agree_count = 0

    feasible_resolve_success = 0
    infeasible_resolve_failed = 0

    cost_rel_errs = []
    cost_abs_errs = []

    print("=== MiniGridSFM30 re-solve integrity check ===")
    print("data:", args.data)
    print("samples:", len(samples))
    print("out_csv:", out_csv)
    print()

    for i, sample in enumerate(samples):
        old_feasible = bool(sample.get("feasible", False))
        if old_feasible:
            old_feasible_count += 1
        else:
            old_infeasible_count += 1

        sample_id = sample.get("sample_id", -1)
        merged_index = sample.get("merged_index", -1)
        mode = sample.get("perturb_mode", obj.get("mode", "unknown"))

        error_msg = ""
        new_feasible = False
        comp = {}

        try:
            old_net = sample_to_net(sample)
            new_net = try_resolve(sample, numba=args.numba)
            new_feasible = True
            comp = compare_tables(old_net, new_net)

            if old_feasible:
                cost_rel_errs.append(comp["cost_rel_err"])
                cost_abs_errs.append(comp["cost_abs_err"])

        except Exception as e:
            new_feasible = False
            error_msg = f"{type(e).__name__}: {e}"

        if new_feasible:
            new_feasible_count += 1
        else:
            new_infeasible_count += 1

        if old_feasible == new_feasible:
            agree_count += 1

        if old_feasible and new_feasible:
            feasible_resolve_success += 1

        if (not old_feasible) and (not new_feasible):
            infeasible_resolve_failed += 1

        cost_rel = comp.get("cost_rel_err", None)
        cost_abs = comp.get("cost_abs_err", None)

        cost_ok = None
        if old_feasible and new_feasible:
            cost_ok = (
                safe_float(cost_rel) <= args.cost_rel_tol
                or safe_float(cost_abs) <= args.cost_abs_tol
            )

        row = {
            "idx": i,
            "sample_id": sample_id,
            "merged_index": merged_index,
            "perturb_mode": mode,
            "old_feasible": int(old_feasible),
            "new_feasible": int(new_feasible),
            "label_agree": int(old_feasible == new_feasible),
            "cost_ok": "" if cost_ok is None else int(bool(cost_ok)),
            "cost_old": comp.get("cost_old", ""),
            "cost_new": comp.get("cost_new", ""),
            "cost_abs_err": comp.get("cost_abs_err", ""),
            "cost_rel_err": comp.get("cost_rel_err", ""),
            "bus_vm_mae": comp.get("bus_vm_mae", ""),
            "bus_va_mae_deg": comp.get("bus_va_mae_deg", ""),
            "gen_p_mae_mw": comp.get("gen_p_mae_mw", ""),
            "gen_q_mae_mvar": comp.get("gen_q_mae_mvar", ""),
            "line_p_from_mae_mw": comp.get("line_p_from_mae_mw", ""),
            "line_q_from_mae_mvar": comp.get("line_q_from_mae_mvar", ""),
            "error_msg": error_msg,
        }

        rows.append(row)

        if (i + 1) % args.progress_every == 0:
            print(f"checked {i + 1}/{len(samples)}")

    fieldnames = [
        "idx",
        "sample_id",
        "merged_index",
        "perturb_mode",
        "old_feasible",
        "new_feasible",
        "label_agree",
        "cost_ok",
        "cost_old",
        "cost_new",
        "cost_abs_err",
        "cost_rel_err",
        "bus_vm_mae",
        "bus_va_mae_deg",
        "gen_p_mae_mw",
        "gen_q_mae_mvar",
        "line_p_from_mae_mw",
        "line_q_from_mae_mvar",
        "error_msg",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(samples)
    label_agreement = agree_count / max(total, 1)
    feas_resolve_rate = feasible_resolve_success / max(old_feasible_count, 1)
    infeas_still_fail_rate = infeasible_resolve_failed / max(old_infeasible_count, 1)

    print()
    print("=== Summary ===")
    print("total:", total)
    print("old feasible:", old_feasible_count)
    print("old infeasible:", old_infeasible_count)
    print("new feasible:", new_feasible_count)
    print("new infeasible:", new_infeasible_count)
    print("label agreement:", f"{label_agreement:.2%}")
    print("feasible re-solve success rate:", f"{feas_resolve_rate:.2%}")
    print("infeasible still-fail rate:", f"{infeas_still_fail_rate:.2%}")

    if cost_rel_errs:
        print()
        print("=== Feasible cost error ===")
        print("cost rel err mean:", float(np.mean(cost_rel_errs)))
        print("cost rel err median:", float(np.median(cost_rel_errs)))
        print("cost rel err max:", float(np.max(cost_rel_errs)))
        print("cost abs err mean:", float(np.mean(cost_abs_errs)))
        print("cost abs err median:", float(np.median(cost_abs_errs)))
        print("cost abs err max:", float(np.max(cost_abs_errs)))

    bad_rows = [
        r for r in rows
        if int(r["label_agree"]) == 0
        or str(r["cost_ok"]) == "0"
    ]

    print()
    print("bad rows:", len(bad_rows))
    print("csv saved:", out_csv)

    if bad_rows:
        print()
        print("First bad rows:")
        for r in bad_rows[:10]:
            print(
                "idx=", r["idx"],
                "sample_id=", r["sample_id"],
                "merged_index=", r["merged_index"],
                "mode=", r["perturb_mode"],
                "old_feasible=", r["old_feasible"],
                "new_feasible=", r["new_feasible"],
                "cost_rel_err=", r["cost_rel_err"],
                "error=", r["error_msg"],
            )


if __name__ == "__main__":
    main()
