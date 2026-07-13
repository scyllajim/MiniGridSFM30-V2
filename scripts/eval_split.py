from __future__ import annotations

import argparse

import torch
from torch_geometric.loader import DataLoader

from minigridsfm30.dataset import Case30OPFDataset
from minigridsfm30.model import GridSFM30, count_parameters
from minigridsfm30.losses import compute_loss


def graph_count(batch) -> int:
    if hasattr(batch, "num_graphs"):
        return int(batch.num_graphs)
    if hasattr(batch["bus"], "batch"):
        b = batch["bus"].batch
        return int(b.max().item()) + 1 if b.numel() > 0 else 1
    return 1


@torch.no_grad()
def evaluate(model, loader, device, loss_kwargs):
    model.eval()

    total_loss = 0.0
    total_graphs = 0
    metrics_sum = {}

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss, metrics = compute_loss(out, batch, **loss_kwargs)

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


def print_metrics(title, m):
    print(f"\n=== {title} ===")
    for k, v in m.items():
        print(f"{k}: {v:.8f}")

    print("\n=== Unit conversion, baseMVA=100 ===")
    print(f"Pg MAE MW:        {m.get('pg_mae', 0.0) * 100:.4f}")
    print(f"Qg MAE MVAr:      {m.get('qg_mae', 0.0) * 100:.4f}")
    print(f"branch P MAE MW:  {m.get('branch_p_mae', 0.0) * 100:.4f}")
    print(f"branch Q MAE MVAr:{m.get('branch_q_mae', 0.0) * 100:.4f}")
    print(f"KCL P MAE MW:     {m.get('kcl_p_mae', 0.0) * 100:.4f}")
    print(f"KCL Q MAE MVAr:   {m.get('kcl_q_mae', 0.0) * 100:.4f}")
    print(f"balance P MAE MW: {m.get('balance_p_mae', 0.0) * 100:.4f}")
    print(f"balance Q MAE MVAr:{m.get('balance_q_mae', 0.0) * 100:.4f}")
    print(f"cost MAPE percent:{m.get('cost_mape', 0.0) * 100:.4f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model_args = ckpt.get(
        "model_args",
        {
            "hidden_dim": 128,
            "num_layers": 3,
            "dropout": 0.0,
        },
    )
    loss_kwargs = ckpt.get(
        "loss_kwargs",
        {
            "lambda_theta": 1.0,
            "lambda_v": 1.0,
            "lambda_pg": 1.0,
            "lambda_qg": 1.0,
            "lambda_branch_p": 1.0,
            "lambda_branch_q": 1.0,
            "lambda_balance_p": 0.1,
            "lambda_balance_q": 0.1,
            "lambda_kcl_p": 1.0,
            "lambda_kcl_q": 1.0,
            "lambda_cost": 0.1,
            "lambda_feas": 0.1,
        },
    )

    model = GridSFM30(**model_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    train_ds = Case30OPFDataset(args.train_data, only_feasible=True)
    val_ds = Case30OPFDataset(args.val_data, only_feasible=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("=== MiniGridSFM30-v2 Split Evaluation ===")
    print("device:", device)
    print("train_data:", args.train_data)
    print("val_data:", args.val_data)
    print("ckpt:", args.ckpt)
    print("checkpoint epoch:", ckpt.get("epoch"))
    print("checkpoint best_val:", ckpt.get("best_val"))
    print("train size:", len(train_ds))
    print("val size:", len(val_ds))
    print("parameters:", count_parameters(model))

    train_metrics = evaluate(model, train_loader, device, loss_kwargs)
    val_metrics = evaluate(model, val_loader, device, loss_kwargs)

    print_metrics("train", train_metrics)
    print_metrics("val", val_metrics)


if __name__ == "__main__":
    main()
