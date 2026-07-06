from __future__ import annotations

import argparse
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
    return Subset(ds, idx[:n_train]), Subset(ds, idx[n_train:])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    total_graphs = 0
    total_loss = 0.0
    metric_sum = {}

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss, metrics = compute_loss(out, batch)

        ng = int(batch.num_graphs)
        total_graphs += ng
        total_loss += float(loss.detach().cpu()) * ng

        for k, v in metrics.items():
            metric_sum[k] = metric_sum.get(k, 0.0) + float(v) * ng

    avg = {k: v / max(total_graphs, 1) for k, v in metric_sum.items()}
    avg["loss"] = total_loss / max(total_graphs, 1)
    return avg


def print_block(name, metrics):
    print(f"=== {name} ===")
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
        "loss_theta",
        "loss_v",
        "loss_pg",
        "loss_qg",
        "loss_branch_p",
        "loss_branch_q",
        "loss_kcl_p",
        "loss_kcl_q",
        "loss_balance_p",
        "loss_balance_q",
        "loss_cost",
        "loss_feas",
    ]

    for k in keys:
        if k in metrics:
            print(f"{k}: {metrics[k]:.8f}")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/case30_1000_v2_graphs.pt")
    parser.add_argument("--ckpt", type=str, default="runs/v2_baseline/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ds = Case30OPFDataset(args.data, only_feasible=True)
    train_ds, val_ds = split_dataset(ds, args.train_ratio, args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = GridSFM30(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("=== MiniGridSFM30-v2 Evaluation ===")
    print("device:", device)
    print("data:", args.data)
    print("ckpt:", args.ckpt)
    print("checkpoint epoch:", ckpt.get("epoch"))
    print("checkpoint best_val:", ckpt.get("best_val"))
    print("dataset size:", len(ds))
    print("train size:", len(train_ds))
    print("val size:", len(val_ds))
    print("parameters:", count_parameters(model))
    print()

    train_m = evaluate(model, train_loader, device)
    val_m = evaluate(model, val_loader, device)

    print_block("train", train_m)
    print_block("val", val_m)

    print("=== Unit conversion, baseMVA=100 ===")
    print(f"val Pg MAE MW:        {val_m['pg_mae'] * 100.0:.4f}")
    print(f"val Qg MAE MVAr:      {val_m['qg_mae'] * 100.0:.4f}")
    print(f"val branch P MAE MW:  {val_m['branch_p_mae'] * 100.0:.4f}")
    print(f"val branch Q MAE MVAr:{val_m['branch_q_mae'] * 100.0:.4f}")
    print(f"val KCL P MAE MW:     {val_m['kcl_p_mae'] * 100.0:.4f}")
    print(f"val KCL Q MAE MVAr:   {val_m['kcl_q_mae'] * 100.0:.4f}")
    print(f"val balance P MAE MW: {val_m['balance_p_mae'] * 100.0:.4f}")
    print(f"val balance Q MAE MVAr:{val_m['balance_q_mae'] * 100.0:.4f}")
    print(f"val cost MAPE percent:{val_m['cost_mape'] * 100.0:.4f}%")


if __name__ == "__main__":
    main()
