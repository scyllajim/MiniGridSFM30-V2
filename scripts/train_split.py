from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from minigridsfm30.dataset import Case30OPFDataset
from minigridsfm30.losses import compute_loss
from minigridsfm30.model import GridSFM30, count_parameters
from minigridsfm30.training_utils import (
    resolve_device,
    save_json,
    set_seed,
    validate_batch_finite,
    validate_metrics_finite,
    validate_output_finite,
)


def graph_count(batch) -> int:
    if hasattr(batch, "num_graphs"):
        return int(batch.num_graphs)
    if hasattr(batch["bus"], "batch"):
        values = batch["bus"].batch
        return int(values.max().item()) + 1 if values.numel() > 0 else 1
    return 1


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    loss_kwargs,
    train: bool,
    check_finite: bool,
):
    model.train(train)

    total_loss = 0.0
    total_graphs = 0
    metrics_sum = {}

    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for batch_index, batch in enumerate(loader):
            if check_finite:
                validate_batch_finite(batch)

            batch = batch.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            output = model(batch)

            if check_finite:
                validate_output_finite(output)

            loss, metrics = compute_loss(output, batch, **loss_kwargs)

            if check_finite:
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at batch {batch_index}: {loss}"
                    )
                validate_metrics_finite(metrics)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            num_graphs = graph_count(batch)
            total_loss += float(loss.detach().cpu()) * num_graphs
            total_graphs += num_graphs

            for key, value in metrics.items():
                if torch.is_tensor(value):
                    value = float(value.detach().cpu())
                else:
                    value = float(value)
                metrics_sum[key] = (
                    metrics_sum.get(key, 0.0) + value * num_graphs
                )

    avg = {"loss": total_loss / max(total_graphs, 1)}
    for key, value in metrics_sum.items():
        avg[key] = value / max(total_graphs, 1)
    return avg


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--run-dir", required=True)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or cuda:N",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="0 disables early stopping.",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--no-finite-check",
        action="store_true",
        help="Disable NaN/Inf checks for inputs, outputs, loss, and metrics.",
    )

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

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience cannot be negative.")

    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    check_finite = not args.no_finite_check

    train_ds = Case30OPFDataset(args.train_data, only_feasible=True)
    val_ds = Case30OPFDataset(args.val_data, only_feasible=True)

    save_json(run_dir / "args.json", vars(args))
    save_json(
        run_dir / "split_manifest.json",
        {
            "train_data": args.train_data,
            "val_data": args.val_data,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "seed": args.seed,
        },
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
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
    print("num workers:", args.num_workers)
    print("finite checks:", check_finite)
    print("early stopping patience:", args.early_stopping_patience)
    print("parameters:", count_parameters(model))
    print("loss weights:", json.dumps(loss_kwargs, indent=2))

    metrics_path = run_dir / "metrics.csv"
    best_val = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    fieldnames = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_kwargs,
            train=True,
            check_finite=check_finite,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            loss_kwargs,
            train=False,
            check_finite=check_finite,
        )

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})

        if fieldnames is None:
            fieldnames = list(row.keys())
            with metrics_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        val_loss = val_metrics.get(
            "loss_total",
            val_metrics.get("loss", float("inf")),
        )

        improved = val_loss < (
            best_val - args.early_stopping_min_delta
        )

        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "model_args": {
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                },
                "args": vars(args),
                "loss_kwargs": loss_kwargs,
                "train_data": args.train_data,
                "val_data": args.val_data,
                "seed": args.seed,
                "split_manifest_file": str(
                    run_dir / "split_manifest.json"
                ),
            }
            torch.save(checkpoint, run_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch:03d} | "
            f"train {train_metrics['loss']:.6f} | "
            f"val {val_metrics['loss']:.6f} | "
            f"best {best_val:.6f}"
        )

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(
                "Early stopping triggered: "
                f"no improvement for {epochs_without_improvement} epoch(s)."
            )
            break

    print("=== Training finished ===")
    print("best_epoch:", best_epoch)
    print("best_val:", best_val)
    print("best_model:", run_dir / "best_model.pt")
    print("metrics:", metrics_path)
    print("split_manifest:", run_dir / "split_manifest.json")


if __name__ == "__main__":
    main()
