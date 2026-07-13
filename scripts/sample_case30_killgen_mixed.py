from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import pandapower as pp

from scripts.sample_case30 import (
    build_base_net,
    perturb_loads,
    perturb_costs,
    perturb_derate,
    perturb_killgen,
    perturb_vsqueeze,
    export_sample,
    set_seed,
)


def run_opf(net, numba: bool = True):
    pp.runopp(
        net,
        verbose=False,
        numba=numba,
        suppress_warnings=True,
    )


def apply_killgen_mixed(net, args):
    """
    Stage11 perturbation:
      always apply killgen first,
      then optionally apply continuous operating-condition perturbations.

    This creates meaningful outage samples:
      gen0/gen3 outage + load/cost/derate/vsqueeze variation.

    Without the continuous perturbations, killgen only has two templates
    and nearest-neighbor can trivially get zero error.
    """
    params = {
        "mode": "killgen_mixed",
        "components": {},
    }

    params["components"]["killgen"] = perturb_killgen(
        net,
        args.killgen_n,
        args.killgen_keep_min,
        args.killgen_candidates,
    )

    if args.apply_loads:
        params["components"]["loads"] = perturb_loads(
            net,
            args.load_min,
            args.load_max,
            args.load_jitter,
        )

    if args.apply_costs:
        params["components"]["costs"] = perturb_costs(
            net,
            args.cost_min,
            args.cost_max,
        )

    if args.apply_derate:
        params["components"]["derate"] = perturb_derate(
            net,
            args.derate_prob,
            args.derate_min,
            args.derate_max,
        )

    if args.apply_vsqueeze:
        params["components"]["vsqueeze"] = perturb_vsqueeze(
            net,
            args.vsqueeze_prob,
            args.vsqueeze_eps,
            args.vm_margin,
        )

    return params


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--case", type=str, default="case30", choices=["case30", "case_ieee30"])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--killgen-n", type=int, default=1)
    parser.add_argument("--killgen-keep-min", type=int, default=1)
    parser.add_argument("--killgen-candidates", type=str, default="0,3")

    parser.add_argument("--apply-loads", action="store_true")
    parser.add_argument("--load-min", type=float, default=0.95)
    parser.add_argument("--load-max", type=float, default=1.05)
    parser.add_argument("--load-jitter", type=float, default=0.00)

    parser.add_argument("--apply-costs", action="store_true")
    parser.add_argument("--cost-min", type=float, default=0.8)
    parser.add_argument("--cost-max", type=float, default=1.2)

    parser.add_argument("--apply-derate", action="store_true")
    parser.add_argument("--derate-prob", type=float, default=0.1)
    parser.add_argument("--derate-min", type=float, default=0.8)
    parser.add_argument("--derate-max", type=float, default=0.98)

    parser.add_argument("--apply-vsqueeze", action="store_true")
    parser.add_argument("--vsqueeze-prob", type=float, default=0.1)
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

    print("=== MiniGridSFM30-v2 Stage11 killgen-mixed sampling ===")
    print("case:", args.case)
    print("n:", args.n)
    print("out:", args.out)
    print("seed:", args.seed)
    print("killgen_candidates:", args.killgen_candidates)
    print("apply_loads:", args.apply_loads)
    print("apply_costs:", args.apply_costs)
    print("apply_derate:", args.apply_derate)
    print("apply_vsqueeze:", args.apply_vsqueeze)
    print("keep_failed:", args.keep_failed)
    print()

    for i in range(args.n):
        net = build_base_net(args.case)
        perturb_params = {}

        try:
            perturb_params = apply_killgen_mixed(net, args)
            run_opf(net, numba=args.numba)

            sample = export_sample(
                net=net,
                sample_id=i,
                feasible=True,
                perturb_mode="killgen_mixed",
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
                    perturb_mode="killgen_mixed",
                    perturb_params=perturb_params,
                    error_msg=err,
                )
                samples.append(sample)

        if (i + 1) % 50 == 0 or i + 1 == args.n:
            print(
                f"[{i + 1}/{args.n}] "
                f"success={n_success} failed={n_failed} saved={len(samples)}"
            )

    dataset = {
        "case": args.case,
        "mode": "killgen_mixed",
        "n_requested": args.n,
        "n_success": n_success,
        "n_failed": n_failed,
        "n_saved": len(samples),
        "seed": args.seed,
        "settings": vars(args),
        "sample_format": {
            "bus_y": ["theta_rad", "vm_pu"],
            "generator_y": ["pg_pu", "qg_pu"],
            "branch_ac_y": ["p_from_pu", "q_from_pu", "p_to_pu", "q_to_pu"],
            "metadata": ["perturb_mode", "perturb_params", "feasible", "res_cost"],
            "note": "Stage11 killgen-mixed samples: selected generator outage plus continuous perturbations.",
        },
        "samples": samples,
    }

    with open(out_path, "wb") as f:
        pickle.dump(dataset, f)

    print()
    print("=== Sampling finished ===")
    print("case:      ", args.case)
    print("mode:      ", "killgen_mixed")
    print("requested: ", args.n)
    print("success:   ", n_success)
    print("failed:    ", n_failed)
    print("saved:     ", len(samples))
    print("out:       ", args.out)


if __name__ == "__main__":
    main()
