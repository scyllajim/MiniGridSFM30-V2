from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from minigridsfm30.dataset import Case30OPFDataset


def to_np(x):
    return x.detach().cpu().numpy()


def graph_input_vector(g):
    """
    Flatten graph-level input features for nearest-neighbor matching.
    This baseline does not use graph structure; it only compares input tensors.
    """
    parts = [
        to_np(g["bus"].x).reshape(-1),
        to_np(g["generator"].x).reshape(-1),
        to_np(g["load"].x).reshape(-1),
        to_np(g["branch_ac"].x).reshape(-1),
    ]
    return np.concatenate(parts, axis=0).astype(np.float32)


def graph_target_dict(g):
    return {
        "bus": to_np(g["bus"].y).astype(np.float32),
        "generator": to_np(g["generator"].y).astype(np.float32),
        "branch_ac": to_np(g["branch_ac"].y).astype(np.float32),
        "res_cost": float(g.res_cost.item()),
    }


def make_mean_target(train_graphs):
    bus = np.stack([graph_target_dict(g)["bus"] for g in train_graphs], axis=0)
    gen = np.stack([graph_target_dict(g)["generator"] for g in train_graphs], axis=0)
    br = np.stack([graph_target_dict(g)["branch_ac"] for g in train_graphs], axis=0)
    cost = np.asarray([graph_target_dict(g)["res_cost"] for g in train_graphs], dtype=np.float32)

    return {
        "bus": bus.mean(axis=0),
        "generator": gen.mean(axis=0),
        "branch_ac": br.mean(axis=0),
        "res_cost": float(cost.mean()),
    }


def compute_one_metrics(pred, true):
    bus_pred = pred["bus"]
    bus_true = true["bus"]

    gen_pred = pred["generator"]
    gen_true = true["generator"]

    br_pred = pred["branch_ac"]
    br_true = true["branch_ac"]

    cost_pred = float(pred["res_cost"])
    cost_true = float(true["res_cost"])

    out = {}

    out["theta_mae"] = float(np.mean(np.abs(bus_pred[:, 0] - bus_true[:, 0])))
    out["v_mae"] = float(np.mean(np.abs(bus_pred[:, 1] - bus_true[:, 1])))

    out["pg_mae"] = float(np.mean(np.abs(gen_pred[:, 0] - gen_true[:, 0])))
    out["qg_mae"] = float(np.mean(np.abs(gen_pred[:, 1] - gen_true[:, 1])))

    out["branch_p_mae"] = float(
        0.5
        * (
            np.mean(np.abs(br_pred[:, 0] - br_true[:, 0]))
            + np.mean(np.abs(br_pred[:, 2] - br_true[:, 2]))
        )
    )
    out["branch_q_mae"] = float(
        0.5
        * (
            np.mean(np.abs(br_pred[:, 1] - br_true[:, 1]))
            + np.mean(np.abs(br_pred[:, 3] - br_true[:, 3]))
        )
    )

    out["cost_mape"] = float(abs(cost_pred - cost_true) / max(abs(cost_true), 1e-9))

    return out


def average_metrics(metrics_list):
    keys = metrics_list[0].keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}


def split_dataset(ds, train_ratio: float, seed: int):
    n = len(ds)
    idx = list(range(n))

    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    train_graphs = [ds[i] for i in train_idx]
    val_graphs = [ds[i] for i in val_idx]

    return train_graphs, val_graphs, train_idx, val_idx


def run_mean_baseline(train_graphs, val_graphs):
    mean_target = make_mean_target(train_graphs)

    rows = []
    for g in val_graphs:
        true = graph_target_dict(g)
        rows.append(compute_one_metrics(mean_target, true))

    return average_metrics(rows)


def run_nearest_neighbor_baseline(train_graphs, val_graphs):
    train_x = np.stack([graph_input_vector(g) for g in train_graphs], axis=0)

    train_targets = [graph_target_dict(g) for g in train_graphs]

    rows = []

    for g in val_graphs:
        x = graph_input_vector(g)

        # normalized L2 distance
        diff = train_x - x[None, :]
        dist = np.mean(diff * diff, axis=1)

        j = int(np.argmin(dist))
        pred = train_targets[j]
        true = graph_target_dict(g)

        rows.append(compute_one_metrics(pred, true))

    return average_metrics(rows)


def write_csv(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "theta_mae",
        "v_mae",
        "pg_mae",
        "qg_mae",
        "branch_p_mae",
        "branch_q_mae",
        "cost_mape",
        "pg_mae_mw",
        "qg_mae_mvar",
        "branch_p_mae_mw",
        "branch_q_mae_mvar",
        "cost_mape_percent",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for name, m in results.items():
            row = {"model": name}
            row.update(m)

            row["pg_mae_mw"] = m["pg_mae"] * 100.0
            row["qg_mae_mvar"] = m["qg_mae"] * 100.0
            row["branch_p_mae_mw"] = m["branch_p_mae"] * 100.0
            row["branch_q_mae_mvar"] = m["branch_q_mae"] * 100.0
            row["cost_mape_percent"] = m["cost_mape"] * 100.0

            writer.writerow(row)


def print_result_table(results):
    print()
    print("=== Baseline comparison on validation split ===")
    print(
        f"{'model':<24} "
        f"{'theta':>10} "
        f"{'V':>10} "
        f"{'Pg(MW)':>10} "
        f"{'Qg(MVAr)':>10} "
        f"{'BrP(MW)':>10} "
        f"{'BrQ(MVAr)':>10} "
        f"{'cost%':>10}"
    )

    for name, m in results.items():
        print(
            f"{name:<24} "
            f"{m['theta_mae']:10.6f} "
            f"{m['v_mae']:10.6f} "
            f"{m['pg_mae'] * 100.0:10.4f} "
            f"{m['qg_mae'] * 100.0:10.4f} "
            f"{m['branch_p_mae'] * 100.0:10.4f} "
            f"{m['branch_q_mae'] * 100.0:10.4f} "
            f"{m['cost_mape'] * 100.0:10.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out-csv", type=str, default="reports/baseline_compare.csv")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = Case30OPFDataset(args.data, only_feasible=True)
    train_graphs, val_graphs, train_idx, val_idx = split_dataset(
        ds,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )

    print("=== MiniGridSFM30 baseline comparison ===")
    print("data:", args.data)
    print("dataset size:", len(ds))
    print("train size:", len(train_graphs))
    print("val size:", len(val_graphs))
    print("seed:", args.seed)

    results = {}
    results["mean_baseline"] = run_mean_baseline(train_graphs, val_graphs)
    results["nearest_neighbor"] = run_nearest_neighbor_baseline(train_graphs, val_graphs)

    write_csv(args.out_csv, results)
    print_result_table(results)

    print()
    print("csv saved:", args.out_csv)


if __name__ == "__main__":
    main()
