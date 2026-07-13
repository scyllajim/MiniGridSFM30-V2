from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from minigridsfm30.dataset import Case30OPFDataset


def flatten_input(g):
    parts = [
        g["bus"].x.flatten(),
        g["generator"].x.flatten(),
        g["load"].x.flatten(),
        g["branch_ac"].x.flatten(),
        g["cycle"].x.flatten(),
    ]
    return torch.cat(parts).float()


def flatten_target(g):
    return {
        "bus": g["bus"].y.float(),
        "generator": g["generator"].y.float(),
        "branch_ac": g["branch_ac"].y.float(),
        "cost": g.res_cost.float().view(1),
    }


def stack_targets(graphs):
    return {
        "bus": torch.stack([flatten_target(g)["bus"] for g in graphs], dim=0),
        "generator": torch.stack([flatten_target(g)["generator"] for g in graphs], dim=0),
        "branch_ac": torch.stack([flatten_target(g)["branch_ac"] for g in graphs], dim=0),
        "cost": torch.stack([flatten_target(g)["cost"] for g in graphs], dim=0),
    }


def compute_metrics(pred, true):
    bus_err = (pred["bus"] - true["bus"]).abs()
    gen_err = (pred["generator"] - true["generator"]).abs()
    br_err = (pred["branch_ac"] - true["branch_ac"]).abs()

    theta = bus_err[..., 0].mean().item()
    v = bus_err[..., 1].mean().item()

    pg = gen_err[..., 0].mean().item() * 100.0
    qg = gen_err[..., 1].mean().item() * 100.0

    brp = torch.cat(
        [
            br_err[..., 0].reshape(-1),
            br_err[..., 2].reshape(-1),
        ],
        dim=0,
    ).mean().item() * 100.0

    brq = torch.cat(
        [
            br_err[..., 1].reshape(-1),
            br_err[..., 3].reshape(-1),
        ],
        dim=0,
    ).mean().item() * 100.0

    cost_true = true["cost"].clamp_min(1e-6)
    cost_mape = ((pred["cost"] - true["cost"]).abs() / cost_true).mean().item() * 100.0

    return {
        "theta": theta,
        "V": v,
        "Pg(MW)": pg,
        "Qg(MVAr)": qg,
        "BrP(MW)": brp,
        "BrQ(MVAr)": brq,
        "cost%": cost_mape,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    train_ds = Case30OPFDataset(args.train_data, only_feasible=True)
    val_ds = Case30OPFDataset(args.val_data, only_feasible=True)

    train_graphs = [train_ds[i] for i in range(len(train_ds))]
    val_graphs = [val_ds[i] for i in range(len(val_ds))]

    train_x = torch.stack([flatten_input(g) for g in train_graphs], dim=0)
    val_x = torch.stack([flatten_input(g) for g in val_graphs], dim=0)

    train_y = stack_targets(train_graphs)
    val_y = stack_targets(val_graphs)

    mean_pred = {
        k: v.mean(dim=0, keepdim=True).expand_as(val_y[k])
        for k, v in train_y.items()
    }

    dists = torch.cdist(val_x, train_x)
    nn_idx = dists.argmin(dim=1)

    nn_pred = {
        k: v[nn_idx]
        for k, v in train_y.items()
    }

    rows = []
    rows.append({"model": "mean_baseline", **compute_metrics(mean_pred, val_y)})
    rows.append({"model": "nearest_neighbor", **compute_metrics(nn_pred, val_y)})

    print("=== MiniGridSFM30 holdout baseline comparison ===")
    print("train_data:", args.train_data)
    print("val_data:", args.val_data)
    print("train size:", len(train_ds))
    print("val size:", len(val_ds))

    print("\nmodel                         theta          V     Pg(MW)   Qg(MVAr)    BrP(MW)  BrQ(MVAr)      cost%")
    for r in rows:
        print(
            f"{r['model']:<24}"
            f"{r['theta']:>10.6f}"
            f"{r['V']:>11.6f}"
            f"{r['Pg(MW)']:>11.4f}"
            f"{r['Qg(MVAr)']:>11.4f}"
            f"{r['BrP(MW)']:>11.4f}"
            f"{r['BrQ(MVAr)']:>11.4f}"
            f"{r['cost%']:>11.4f}"
        )

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model",
                    "theta",
                    "V",
                    "Pg(MW)",
                    "Qg(MVAr)",
                    "BrP(MW)",
                    "BrQ(MVAr)",
                    "cost%",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print("\ncsv saved:", out)


if __name__ == "__main__":
    main()
