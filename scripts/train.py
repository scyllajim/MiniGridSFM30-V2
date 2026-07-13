from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from minigridsfm30.dataset import Case30OPFDataset
from minigridsfm30.model import GridSFM30, count_parameters
from minigridsfm30.losses import compute_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_dataset(ds, train_ratio: float, seed: int):
    n = len(ds)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    return Subset(ds, train_idx), Subset(ds, val_idx)


def move_to_device(batch, device):
    return batch.to(device)


def run_epoch(model, loader, optimizer, device, train: bool, loss_kwargs: dict):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_graphs = 0
    metric_sum = {}

    for batch in loader:
        batch = move_to_device(batch, device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(batch)
            loss, metrics = compute_loss(out, batch, **loss_kwargs)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        ng = int(batch.num_graphs)
        total_loss += float(loss.detach().cpu()) * ng
        total_graphs += ng

        for k, v in metrics.items():
            metric_sum[k] = metric_sum.get(k, 0.0) + float(v) * ng

    avg = {k: v / max(total_graphs, 1) for k, v in metric_sum.items()}
    avg["loss"] = total_loss / max(total_graphs, 1)
    return avg


def print_metrics(prefix, epoch, metrics):
    keys = [
        "loss",
        "loss_total",
        "theta_mae",
        "v_mae",
        "pg_mae",
        "qg_mae",
        "branch_p_mae",
        "branch_q_mae",
        "kcl_p_mae",
        "kcl_q_mae",
        "balance_p_mae",
        "balance_q_mae",
        "cost_mape",
    ]

    parts = [f"{prefix} epoch={epoch:03d}"]
    for k in keys:
        if k in metrics:
            parts.append(f"{k}={metrics[k]:.6f}")
    print(" | ".join(parts))


def save_metrics_csv(path, rows):
    if len(rows) == 0:
        return

    keys = sorted(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--run-dir", type=str, default="runs/v2_baseline")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)

    # ----------------------------
    # Loss weights for Stage 8 physics-loss ablation
    # These names match compute_loss() in minigridsfm30/losses.py
    # ----------------------------
    parser.add_argument("--lambda-theta", type=float, default=1.0)
    parser.add_argument("--lambda-v", type=float, default=1.0)
    parser.add_argument("--lambda-pg", type=float, default=1.0)
    parser.add_argument("--lambda-qg", type=float, default=1.0)
    parser.add_argument("--lambda-branch-p", type=float, default=1.0)
    parser.add_argument("--lambda-branch-q", type=float, default=1.0)
    parser.add_argument("--lambda-balance-p", type=float, default=0.1)
    parser.add_argument("--lambda-balance-q", type=float, default=0.1)
    parser.add_argument("--lambda-kcl-p", type=float, default=1.0)
    parser.add_argument("--lambda-kcl-q", type=float, default=1.0)
    parser.add_argument("--lambda-cost", type=float, default=0.1)
    parser.add_argument("--lambda-feas", type=float, default=0.1)

    args = parser.parse_args()

    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ds = Case30OPFDataset(
        args.data,
        only_feasible=True,
        max_samples=args.max_samples,
    )

    train_ds, val_ds = split_dataset(ds, args.train_ratio, args.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = GridSFM30(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    loss_kwargs = {
        "lambda_theta": args.lambda_theta,
        "lambda_v": args.lambda_v,
        "lambda_pg": args.lambda_pg,
        "lambda_qg": args.lambda_qg,
        "lambda_branch_p": args.lambda_branch_p,
        "lambda_branch_q": args.lambda_branch_q,
        "lambda_balance_p": args.lambda_balance_p,
        "lambda_balance_q": args.lambda_balance_q,
        "lambda_kcl_p": args.lambda_kcl_p,
        "lambda_kcl_q": args.lambda_kcl_q,
        "lambda_cost": args.lambda_cost,
        "lambda_feas": args.lambda_feas,
    }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("=== MiniGridSFM30-v2 Training ===")
    print("device:", device)
    print("data:", args.data)
    print("dataset size:", len(ds))
    print("train size:", len(train_ds))
    print("val size:", len(val_ds))
    print("batch size:", args.batch_size)
    print("hidden_dim:", args.hidden_dim)
    print("num_layers:", args.num_layers)
    print("parameters:", count_parameters(model))
    print("run_dir:", run_dir)
    print("loss weights:", loss_kwargs)
    print()

    best_val = float("inf")
    rows = []

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            loss_kwargs=loss_kwargs,
        )

        val_m = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            train=False,
            loss_kwargs=loss_kwargs,
        )

        print_metrics("train", epoch, train_m)
        print_metrics("val  ", epoch, val_m)
        print()

        row = {"epoch": epoch}
        for k, v in train_m.items():
            row[f"train_{k}"] = v
        for k, v in val_m.items():
            row[f"val_{k}"] = v
        rows.append(row)

        val_loss = val_m.get("loss_total", val_m.get("loss", float("inf")))

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "loss_kwargs": loss_kwargs,
                "best_val": best_val,
            }
            torch.save(ckpt, run_dir / "best_model.pt")

        save_metrics_csv(run_dir / "metrics.csv", rows)

    print("=== Training finished ===")
    print("best_val:", best_val)
    print("best_model:", run_dir / "best_model.pt")
    print("metrics:", run_dir / "metrics.csv")


if __name__ == "__main__":
    main()