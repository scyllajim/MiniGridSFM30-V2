from __future__ import annotations

import argparse
import pickle
import random
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("inputs", nargs="+")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    all_samples = []
    input_infos = []
    mode_counter = Counter()
    feasible_counter = Counter()

    case_name = None

    for path_str in args.inputs:
        path = Path(path_str)

        if not path.exists():
            raise FileNotFoundError(path)

        with open(path, "rb") as f:
            obj = pickle.load(f)

        if case_name is None:
            case_name = obj.get("case", "unknown")

        samples = obj.get("samples", [])
        mode = obj.get("mode", "unknown")

        input_infos.append({
            "path": str(path),
            "case": obj.get("case", "unknown"),
            "mode": mode,
            "n_requested": obj.get("n_requested", 0),
            "n_success": obj.get("n_success", 0),
            "n_failed": obj.get("n_failed", 0),
            "n_saved": obj.get("n_saved", len(samples)),
        })

        for s in samples:
            s = dict(s)
            if "source_file" not in s:
                s["source_file"] = str(path)

            mode_i = s.get("perturb_mode", mode)
            mode_counter[mode_i] += 1
            feasible_counter["feasible" if s.get("feasible", False) else "infeasible"] += 1

            all_samples.append(s)

    if args.shuffle:
        rng.shuffle(all_samples)

    for i, s in enumerate(all_samples):
        s["merged_index"] = i

    out_obj = {
        "case": case_name,
        "mode": "merged_pure_modes",
        "n_requested": len(all_samples),
        "n_success": feasible_counter["feasible"],
        "n_failed": feasible_counter["infeasible"],
        "n_saved": len(all_samples),
        "seed": args.seed,
        "shuffle": bool(args.shuffle),
        "input_files": input_infos,
        "mode_counts": dict(mode_counter),
        "feasible_counts": dict(feasible_counter),
        "sample_format": {
            "bus_y": ["theta_rad", "vm_pu"],
            "generator_y": ["pg_pu", "qg_pu"],
            "branch_ac_y": ["p_from_pu", "q_from_pu", "p_to_pu", "q_to_pu"],
            "metadata": ["perturb_mode", "perturb_params", "feasible", "res_cost"],
            "note": "Merged pure-mode MiniGridSFM30 raw dataset. Infeasible samples are retained for feasibility classification.",
        },
        "samples": all_samples,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(out_obj, f)

    print("=== Merged raw datasets ===")
    print("out:", out_path)
    print("case:", case_name)
    print("n_saved:", len(all_samples))
    print("feasible:", feasible_counter["feasible"])
    print("infeasible:", feasible_counter["infeasible"])
    print()
    print("mode counts:")
    for k, v in sorted(mode_counter.items()):
        print(f"  {k}: {v}")
    print()
    print("input files:")
    for info in input_infos:
        print(
            f"  {info['path']} | mode={info['mode']} "
            f"success={info['n_success']} failed={info['n_failed']} saved={info['n_saved']}"
        )


if __name__ == "__main__":
    main()
