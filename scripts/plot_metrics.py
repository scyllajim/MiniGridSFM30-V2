from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_one(df, ycols, title, ylabel, out):
    plt.figure(figsize=(8, 5))

    for c in ycols:
        if c in df.columns:
            plt.plot(df["epoch"], df[c], label=c)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="runs/v2_baseline/metrics.csv")
    parser.add_argument("--out-dir", type=str, default="runs/v2_baseline/figures")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    plot_one(
        df,
        ["train_loss_total", "val_loss_total"],
        "Total Loss",
        "Loss",
        out_dir / "loss_total.png",
    )

    plot_one(
        df,
        ["train_theta_mae", "val_theta_mae"],
        "Theta MAE",
        "rad",
        out_dir / "theta_mae.png",
    )

    plot_one(
        df,
        ["train_v_mae", "val_v_mae"],
        "Voltage Magnitude MAE",
        "p.u.",
        out_dir / "v_mae.png",
    )

    plot_one(
        df,
        ["train_pg_mae", "val_pg_mae", "train_qg_mae", "val_qg_mae"],
        "Generator Dispatch MAE",
        "p.u.",
        out_dir / "generator_mae.png",
    )

    plot_one(
        df,
        ["train_branch_p_mae", "val_branch_p_mae", "train_branch_q_mae", "val_branch_q_mae"],
        "Branch Flow MAE",
        "p.u.",
        out_dir / "branch_flow_mae.png",
    )

    plot_one(
        df,
        ["train_kcl_p_mae", "val_kcl_p_mae", "train_kcl_q_mae", "val_kcl_q_mae"],
        "Bus KCL Residual MAE",
        "p.u.",
        out_dir / "kcl_mae.png",
    )

    plot_one(
        df,
        ["train_balance_p_mae", "val_balance_p_mae", "train_balance_q_mae", "val_balance_q_mae"],
        "Graph Power Balance MAE",
        "p.u.",
        out_dir / "balance_mae.png",
    )

    plot_one(
        df,
        ["train_cost_mape", "val_cost_mape"],
        "Generation Cost MAPE",
        "MAPE",
        out_dir / "cost_mape.png",
    )

    print("saved figures to:", out_dir)


if __name__ == "__main__":
    main()
