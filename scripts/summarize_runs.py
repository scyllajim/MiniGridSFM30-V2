from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch

UNIT_SCALE = 100.0


def safe_float(value: Any):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_best_metrics(metrics_path: Path, best_epoch: int | None):
    if not metrics_path.exists():
        return {}

    with metrics_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return {}

    if best_epoch is not None:
        for row in rows:
            try:
                if int(float(row.get("epoch", -1))) == best_epoch:
                    return row
            except (TypeError, ValueError):
                continue

    def val_loss(row):
        for key in ("val_loss_total", "val_loss"):
            value = safe_float(row.get(key))
            if value is not None:
                return value
        return float("inf")

    return min(rows, key=val_loss)


def collect_run(run_dir: Path):
    ckpt_path = run_dir / "best_model.pt"
    metrics_path = run_dir / "metrics.csv"

    if not ckpt_path.exists() and not metrics_path.exists():
        return None

    checkpoint = {}
    if ckpt_path.exists():
        try:
            checkpoint = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            checkpoint = {"load_error": str(exc)}

    best_epoch = checkpoint.get("epoch")
    if best_epoch is not None:
        try:
            best_epoch = int(best_epoch)
        except (TypeError, ValueError):
            best_epoch = None

    row = load_best_metrics(metrics_path, best_epoch)
    args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    loss_kwargs = checkpoint.get("loss_kwargs", {}) if isinstance(checkpoint, dict) else {}

    result = {
        "run": str(run_dir),
        "best_epoch": best_epoch,
        "best_val": safe_float(checkpoint.get("best_val")),
        "data": args.get("data"),
        "train_data": args.get("train_data"),
        "val_data": args.get("val_data"),
        "seed": args.get("seed"),
        "hidden_dim": args.get("hidden_dim"),
        "num_layers": args.get("num_layers"),
        "batch_size": args.get("batch_size"),
        "lr": args.get("lr"),
        "lambda_kcl_p": loss_kwargs.get("lambda_kcl_p"),
        "lambda_kcl_q": loss_kwargs.get("lambda_kcl_q"),
        "lambda_balance_p": loss_kwargs.get("lambda_balance_p"),
        "lambda_balance_q": loss_kwargs.get("lambda_balance_q"),
        "lambda_cost": loss_kwargs.get("lambda_cost"),
    }

    metric_keys = [
        "val_theta_mae",
        "val_v_mae",
        "val_pg_mae",
        "val_qg_mae",
        "val_branch_p_mae",
        "val_branch_q_mae",
        "val_kcl_p_mae",
        "val_kcl_q_mae",
        "val_balance_p_mae",
        "val_balance_q_mae",
        "val_cost_mape",
        "val_loss_total",
        "val_loss",
    ]

    for key in metric_keys:
        result[key] = safe_float(row.get(key))

    for key in (
        "val_pg_mae",
        "val_qg_mae",
        "val_branch_p_mae",
        "val_branch_q_mae",
        "val_kcl_p_mae",
        "val_kcl_q_mae",
        "val_balance_p_mae",
        "val_balance_q_mae",
    ):
        value = result.get(key)
        result[f"{key}_physical"] = None if value is None else value * UNIT_SCALE

    cost = result.get("val_cost_mape")
    result["val_cost_mape_percent"] = None if cost is None else cost * 100.0

    return result


def markdown_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(path: Path, rows):
    columns = [
        ("run", "Run"),
        ("best_epoch", "Epoch"),
        ("seed", "Seed"),
        ("val_pg_mae_physical", "Pg MW"),
        ("val_qg_mae_physical", "Qg MVAr"),
        ("val_branch_p_mae_physical", "BrP MW"),
        ("val_branch_q_mae_physical", "BrQ MVAr"),
        ("val_kcl_p_mae_physical", "KCL P MW"),
        ("val_balance_p_mae_physical", "Balance P MW"),
        ("val_cost_mape_percent", "Cost %"),
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write("# Experiment Summary\n\n")
        f.write("| " + " | ".join(title for _, title in columns) + " |\n")
        f.write("|" + "|".join("---" for _ in columns) + "|\n")

        for row in rows:
            values = [markdown_value(row.get(key)) for key, _ in columns]
            f.write("| " + " | ".join(values) + " |\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-csv", default="reports/experiment_summary.csv")
    parser.add_argument("--out-md", default="reports/experiment_summary.md")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)

    run_dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir())

    rows = []
    for run_dir in run_dirs:
        result = collect_run(run_dir)
        if result is not None:
            rows.append(result)

    if not rows:
        raise SystemExit(f"No runs found under: {runs_dir}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(out_md, rows)

    print("=== Experiment summary ===")
    print("runs:", len(rows))
    print("csv:", out_csv)
    print("markdown:", out_md)


if __name__ == "__main__":
    main()
