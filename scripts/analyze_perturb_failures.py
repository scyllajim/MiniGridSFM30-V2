from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path


def bucket(x: float, step: float = 0.05):
    lo = int(x / step) * step
    hi = lo + step
    return f"[{lo:.2f},{hi:.2f})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()

    load_rows = []
    killgen_stats = defaultdict(lambda: {"success": 0, "failed": 0, "total": 0})
    mode_stats = defaultdict(lambda: {"success": 0, "failed": 0, "total": 0})

    for path_str in args.paths:
        path = Path(path_str)
        with open(path, "rb") as f:
            obj = pickle.load(f)

        for s in obj.get("samples", []):
            mode = s.get("perturb_mode", obj.get("mode", "unknown"))
            feasible = bool(s.get("feasible", False))
            pp = s.get("perturb_params", {})

            mode_stats[mode]["total"] += 1
            if feasible:
                mode_stats[mode]["success"] += 1
            else:
                mode_stats[mode]["failed"] += 1

            if mode == "loads":
                load_rows.append({
                    "feasible": feasible,
                    "global_scale": pp.get("global_scale"),
                    "scale_min": pp.get("scale_min"),
                    "scale_max": pp.get("scale_max"),
                    "scale_mean": pp.get("scale_mean"),
                })

            if mode == "killgen":
                killed = pp.get("killed_gen_indices", [])
                for g in killed:
                    g = int(g)
                    killgen_stats[g]["total"] += 1
                    if feasible:
                        killgen_stats[g]["success"] += 1
                    else:
                        killgen_stats[g]["failed"] += 1

    print("=== Mode summary ===")
    print(f"{'mode':<12} {'total':>8} {'success':>8} {'failed':>8} {'rate':>10}")
    for mode, st in sorted(mode_stats.items()):
        rate = st["success"] / max(st["total"], 1)
        print(f"{mode:<12} {st['total']:>8} {st['success']:>8} {st['failed']:>8} {rate:>9.2%}")

    if load_rows:
        print()
        print("=== Loads global_scale diagnosis ===")
        valid = [r for r in load_rows if r["global_scale"] is not None]
        succ = [r for r in valid if r["feasible"]]
        fail = [r for r in valid if not r["feasible"]]

        def avg(rows, key):
            vals = [r[key] for r in rows if r[key] is not None]
            return sum(vals) / max(len(vals), 1)

        print("n loads:", len(valid))
        print("success:", len(succ))
        print("failed:", len(fail))
        print("avg global_scale success:", f"{avg(succ, 'global_scale'):.4f}")
        print("avg global_scale failed: ", f"{avg(fail, 'global_scale'):.4f}")
        print("avg scale_max success:   ", f"{avg(succ, 'scale_max'):.4f}")
        print("avg scale_max failed:    ", f"{avg(fail, 'scale_max'):.4f}")

        bucket_stats = defaultdict(lambda: {"success": 0, "failed": 0, "total": 0})
        for r in valid:
            b = bucket(float(r["global_scale"]), step=0.05)
            bucket_stats[b]["total"] += 1
            if r["feasible"]:
                bucket_stats[b]["success"] += 1
            else:
                bucket_stats[b]["failed"] += 1

        print()
        print(f"{'global_scale':<16} {'total':>8} {'success':>8} {'failed':>8} {'rate':>10}")
        for b, st in sorted(bucket_stats.items()):
            rate = st["success"] / max(st["total"], 1)
            print(f"{b:<16} {st['total']:>8} {st['success']:>8} {st['failed']:>8} {rate:>9.2%}")

    if killgen_stats:
        print()
        print("=== Killgen per-generator diagnosis ===")
        print(f"{'gen':<6} {'total':>8} {'success':>8} {'failed':>8} {'rate':>10}")
        for g, st in sorted(killgen_stats.items()):
            rate = st["success"] / max(st["total"], 1)
            print(f"{g:<6} {st['total']:>8} {st['success']:>8} {st['failed']:>8} {rate:>9.2%}")


if __name__ == "__main__":
    main()
