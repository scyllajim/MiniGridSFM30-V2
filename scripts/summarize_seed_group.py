from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any

import torch


PHYSICAL_METRICS = {
    "val_pg_mae": ("Pg MAE", "MW", 100.0),
    "val_qg_mae": ("Qg MAE", "MVAr", 100.0),
    "val_branch_p_mae": ("Branch P MAE", "MW", 100.0),
    "val_branch_q_mae": ("Branch Q MAE", "MVAr", 100.0),
    "val_kcl_p_mae": ("KCL P MAE", "MW", 100.0),
    "val_kcl_q_mae": ("KCL Q MAE", "MVAr", 100.0),
    "val_balance_p_mae": ("Balance P MAE", "MW", 100.0),
    "val_balance_q_mae": ("Balance Q MAE", "MVAr", 100.0),
    "val_cost_mape": ("Cost MAPE", "%", 100.0),
}

RAW_METRICS = {
    "val_theta_mae": ("Theta MAE", "", 1.0),
    "val_v_mae": ("V MAE", "p.u.", 1.0),
    "val_loss_total": ("Validation loss", "", 1.0),
}


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_best_row(run_dir: Path) -> tuple[int | None, dict[str, str]]:
    ckpt_path = run_dir / "best_model.pt"
    metrics_path = run_dir / "metrics.csv"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics: {metrics_path}")

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )
    best_epoch = checkpoint.get("epoch")
    if best_epoch is not None:
        best_epoch = int(best_epoch)

    with metrics_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(f"No rows in {metrics_path}")

    if best_epoch is not None:
        for row in rows:
            if int(float(row["epoch"])) == best_epoch:
                return best_epoch, row

    def score(row: dict[str, str]) -> float:
        for key in ("val_loss_total", "val_loss"):
            value = as_float(row.get(key))
            if value is not None:
                return value
        return float("inf")

    row = min(rows, key=score)
    return int(float(row["epoch"])), row


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--out-csv",
        default="reports/stage13_seed_summary.csv",
    )
    parser.add_argument(
        "--out-md",
        default="reports/stage13_seed_summary.md",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    records: list[dict[str, Any]] = []

    all_metrics = {**RAW_METRICS, **PHYSICAL_METRICS}

    for seed in args.seeds:
        run_dir = runs_dir / f"{args.prefix}{seed}"
        epoch, row = load_best_row(run_dir)

        record: dict[str, Any] = {
            "seed": seed,
            "run": str(run_dir),
            "best_epoch": epoch,
        }

        for key, (_, _, scale) in all_metrics.items():
            value = as_float(row.get(key))
            record[key] = None if value is None else value * scale

        records.append(record)

    aggregate: list[dict[str, Any]] = []
    for key, (label, unit, _) in all_metrics.items():
        values = [
            float(record[key])
            for record in records
            if record.get(key) is not None
        ]
        if not values:
            continue

        mean, std = mean_std(values)
        aggregate.append(
            {
                "metric": key,
                "label": label,
                "unit": unit,
                "n": len(values),
                "mean": mean,
                "std": std,
                "min": min(values),
                "max": max(values),
            }
        )

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    per_seed_csv = out_csv.with_name(
        out_csv.stem + "_per_seed.csv"
    )

    per_seed_fields = [
        "seed",
        "run",
        "best_epoch",
        *all_metrics.keys(),
    ]
    with per_seed_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_seed_fields)
        writer.writeheader()
        writer.writerows(records)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "label",
                "unit",
                "n",
                "mean",
                "std",
                "min",
                "max",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregate)

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Stage13 Five-Seed Summary\n\n")
        f.write(
            f"Run prefix: `{args.prefix}`  \n"
            f"Seeds: {', '.join(map(str, args.seeds))}\n\n"
        )

        f.write("## Aggregate results\n\n")
        f.write("| Metric | Mean ± Std | Min | Max | Unit | N |\n")
        f.write("|---|---:|---:|---:|---|---:|\n")
        for item in aggregate:
            f.write(
                "| {label} | {mean:.4f} ± {std:.4f} | "
                "{min:.4f} | {max:.4f} | {unit} | {n} |\n".format(
                    **item
                )
            )

        f.write("\n## Per-seed results\n\n")
        f.write(
            "| Seed | Epoch | Pg MW | Qg MVAr | BrP MW | "
            "BrQ MVAr | KCL P MW | Balance P MW | Cost % |\n"
        )
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for record in records:
            def fmt(key: str) -> str:
                value = record.get(key)
                return "" if value is None else f"{value:.4f}"

            f.write(
                f"| {record['seed']} | {record['best_epoch']} | "
                f"{fmt('val_pg_mae')} | {fmt('val_qg_mae')} | "
                f"{fmt('val_branch_p_mae')} | "
                f"{fmt('val_branch_q_mae')} | "
                f"{fmt('val_kcl_p_mae')} | "
                f"{fmt('val_balance_p_mae')} | "
                f"{fmt('val_cost_mape')} |\n"
            )

    print("=== Five-seed summary ===")
    print("runs:", len(records))
    print("aggregate csv:", out_csv)
    print("per-seed csv:", per_seed_csv)
    print("markdown:", out_md)


if __name__ == "__main__":
    main()
