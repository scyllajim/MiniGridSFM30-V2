from __future__ import annotations

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="raw .pkl dataset files")
    args = parser.parse_args()

    print("=== MiniGridSFM30 raw dataset analysis ===")
    print()

    total_requested = 0
    total_success = 0
    total_failed = 0
    total_saved = 0

    mode_summary = defaultdict(lambda: {
        "files": 0,
        "requested": 0,
        "success": 0,
        "failed": 0,
        "saved": 0,
        "feasible_saved": 0,
        "infeasible_saved": 0,
    })

    killgen_counter = Counter()
    error_counter = Counter()

    for path_str in args.paths:
        path = Path(path_str)

        if not path.exists():
            print(f"[MISSING] {path}")
            continue

        with open(path, "rb") as f:
            obj = pickle.load(f)

        mode = obj.get("mode", "unknown")
        n_req = int(obj.get("n_requested", 0))
        n_succ = int(obj.get("n_success", 0))
        n_fail = int(obj.get("n_failed", 0))
        n_saved = int(obj.get("n_saved", len(obj.get("samples", []))))

        samples = obj.get("samples", [])
        feasible_saved = sum(1 for s in samples if s.get("feasible", False))
        infeasible_saved = sum(1 for s in samples if not s.get("feasible", False))

        total_requested += n_req
        total_success += n_succ
        total_failed += n_fail
        total_saved += n_saved

        ms = mode_summary[mode]
        ms["files"] += 1
        ms["requested"] += n_req
        ms["success"] += n_succ
        ms["failed"] += n_fail
        ms["saved"] += n_saved
        ms["feasible_saved"] += feasible_saved
        ms["infeasible_saved"] += infeasible_saved

        for s in samples:
            if not s.get("feasible", False):
                err = s.get("error_msg", "")
                if err:
                    error_counter[err.split(":")[0]] += 1

            pp = s.get("perturb_params", {})
            if s.get("perturb_mode") == "killgen":
                killed = pp.get("killed_gen_indices", [])
                for g in killed:
                    killgen_counter[int(g)] += 1

        rate = n_succ / max(n_req, 1)

        print(f"file: {path}")
        print(f"  mode:      {mode}")
        print(f"  requested: {n_req}")
        print(f"  success:   {n_succ}")
        print(f"  failed:    {n_fail}")
        print(f"  saved:     {n_saved}")
        print(f"  success rate: {rate:.2%}")
        print(f"  feasible saved:   {feasible_saved}")
        print(f"  infeasible saved: {infeasible_saved}")
        print()

    print("=" * 70)
    print("Summary by mode")
    print("=" * 70)

    print(
        f"{'mode':<12} {'files':>5} {'requested':>10} {'success':>8} "
        f"{'failed':>8} {'rate':>10} {'saved':>8} {'inf_saved':>10}"
    )

    for mode, ms in sorted(mode_summary.items()):
        rate = ms["success"] / max(ms["requested"], 1)
        print(
            f"{mode:<12} {ms['files']:>5} {ms['requested']:>10} {ms['success']:>8} "
            f"{ms['failed']:>8} {rate:>9.2%} {ms['saved']:>8} {ms['infeasible_saved']:>10}"
        )

    print()
    print("=" * 70)
    print("Total")
    print("=" * 70)
    print("requested:", total_requested)
    print("success:  ", total_success)
    print("failed:   ", total_failed)
    print("saved:    ", total_saved)
    print("success rate:", f"{total_success / max(total_requested, 1):.2%}")

    if killgen_counter:
        print()
        print("=" * 70)
        print("Killgen killed generator frequency")
        print("=" * 70)
        for g, c in sorted(killgen_counter.items()):
            print(f"gen {g}: {c}")

    if error_counter:
        print()
        print("=" * 70)
        print("Error type frequency")
        print("=" * 70)
        for err, c in error_counter.most_common():
            print(f"{err}: {c}")


if __name__ == "__main__":
    main()
