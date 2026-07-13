from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from minigridsfm30.dataset import Case30OPFDataset
from minigridsfm30.model import GridSFM30, count_parameters
from minigridsfm30.losses import compute_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def graph_count(batch) -> int:
    if hasattr(batch, "num_graphs"):
        return int(batch.num_graphs)
    if hasattr(batch["bus"], "batch"):
        b = batch["bus"].batch
        return int(b.max().item()) + 1 if b.numel() > 0 else 1
    return 1


def run_epoch(model, loader, optimizer, device, loss_kwargs, train: bool):
    model.train(train)

    total_loss = 0.0
    total_graphs = 0
    metrics_sum = {}

    for batch in loader:
        batch = batch.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        out = model(batch)
        loss, metrics = compute_loss(out, batch, **loss_kwargs)

        if train:
            loss.backward()
            optimizer.step()

        n = graph_count(batch)
        total_loss += float(loss.detach().cpu()) * n
        total_graphs += n

        for k, v in metrics.items():
            if torch.is_tensor(v):
                v = float(v.detach().cpu())
            else:
                v = float(v)
            metrics_sum[k] = metrics_sum.get(k, 0.0) + v * n

    avg = {"loss": total_loss / max(total_graphs, 1)}
    for k, v in metrics_sum.items():
        avg[k] = v / max(total_graphs, 1)
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--run-dir", required=True)

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)

    ap.add_argument("--lambda-theta", type=float, default=1.0)
    ap.add_argument("--lambda-v", type=float, default=1.0)
    ap.add_argument("--lambda-pg", type=float, default=1.0)
    ap.add_argument("--lambda-qg", type=float, default=1.0)
    ap.add_argument("--lambda-branch-p", type=float, default=1.0)
    ap.add_argument("--lambda-branch-q", type=float, default=1.0)
    ap.add_argument("--lambda-balance-p", type=float, default=0.1)
    ap.add_argument("--lambda-balance-q", type=float, default=0.1)
    ap.add_argument("--lambda-kcl-p", type=float, default=1.0)
    ap.add_argument("--lambda-kcl-q", type=float, default=1.0)
    ap.add_argument("--lambda-cost", type=float, default=0.1)
    ap.add_argument("--lambda-feas", type=float, default=0.1)

    args = ap.parse_args()

    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_ds = Case30OPFDataset(args.train_data, only_feasible=True)
    val_ds = Case30OPFDataset(args.val_data, only_feasible=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = GridSFM30(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

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

    print("=== MiniGridSFM30-v2 Split Training ===")
    print("device:", device)
    print("train_data:", args.train_data)
    print("val_data:", args.val_data)
    print("train size:", len(train_ds))
    print("val size:", len(val_ds))
    print("parameters:", count_parameters(model))
    print("loss weights:", json.dumps(loss_kwargs, indent=2))

    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    metrics_path = run_dir / "metrics.csv"
    best_val = float("inf")
    best_epoch = -1

    fieldnames = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_kwargs,
            train=True,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            loss_kwargs,
            train=False,
        )

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})

        if fieldnames is None:
            fieldnames = list(row.keys())
            with open(metrics_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

        with open(metrics_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        val_loss = val_metrics["loss"]

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            ckpt = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "model_args": {
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                },
                "loss_kwargs": loss_kwargs,
                "train_data": args.train_data,
                "val_data": args.val_data,
                "seed": args.seed,
            }
            torch.save(ckpt, run_dir / "best_model.pt")

        print(
            f"epoch {epoch:03d} | "
            f"train {train_metrics['loss']:.6f} | "
            f"val {val_metrics['loss']:.6f} | "
            f"best {best_val:.6f}"
        )

    print("=== Training finished ===")
    print("best_epoch:", best_epoch)
    print("best_val:", best_val)
    print("best_model:", run_dir / "best_model.pt")
    print("metrics:", metrics_path)


if __name__ == "__main__":
    main()
