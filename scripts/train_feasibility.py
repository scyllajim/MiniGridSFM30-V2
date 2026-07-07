from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from minigridsfm30.dataset import Case30OPFDataset
from minigridsfm30.model import GridSFM30, count_parameters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_graph_label(g) -> int:
    return int(float(g.feasible.item()) >= 0.5)


def stratified_split(ds, train_ratio: float, seed: int):
    pos_idx = []
    neg_idx = []

    for i in range(len(ds)):
        y = get_graph_label(ds[i])
        if y == 1:
            pos_idx.append(i)
        else:
            neg_idx.append(i)

    rng = random.Random(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_pos_train = int(len(pos_idx) * train_ratio)
    n_neg_train = int(len(neg_idx) * train_ratio)

    train_idx = pos_idx[:n_pos_train] + neg_idx[:n_neg_train]
    val_idx = pos_idx[n_pos_train:] + neg_idx[n_neg_train:]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return Subset(ds, train_idx), Subset(ds, val_idx), {
        "pos_total": len(pos_idx),
        "neg_total": len(neg_idx),
        "pos_train": n_pos_train,
        "neg_train": n_neg_train,
        "pos_val": len(pos_idx) - n_pos_train,
        "neg_val": len(neg_idx) - n_neg_train,
    }


def get_label(batch, device):
    return batch.feasible.view(-1).to(device=device, dtype=torch.float32)


def weighted_bce_loss(logit, y, pos_weight_value: float, neg_weight_value: float):
    per_item = F.binary_cross_entropy_with_logits(
        logit,
        y,
        reduction="none",
    )

    weights = torch.where(
        y >= 0.5,
        torch.full_like(y, float(pos_weight_value)),
        torch.full_like(y, float(neg_weight_value)),
    )

    return (per_item * weights).mean()


@torch.no_grad()
def compute_metrics_from_logits(logit, y, threshold: float):
    prob_feas = torch.sigmoid(logit)

    pred_feas = (prob_feas >= threshold).float()
    true_feas = y

    correct = int((pred_feas == true_feas).sum().detach().cpu())
    total = int(y.numel())

    tp_feas = int(((pred_feas == 1) & (true_feas == 1)).sum().detach().cpu())
    tn_infeas = int(((pred_feas == 0) & (true_feas == 0)).sum().detach().cpu())
    fp_feas = int(((pred_feas == 1) & (true_feas == 0)).sum().detach().cpu())
    fn_infeas = int(((pred_feas == 0) & (true_feas == 1)).sum().detach().cpu())

    # Treat infeasible as the important alarm class.
    # pred_infeas = 1 means model predicts infeasible.
    pred_infeas = 1.0 - pred_feas
    true_infeas = 1.0 - true_feas

    tp_infeas = int(((pred_infeas == 1) & (true_infeas == 1)).sum().detach().cpu())
    fp_infeas = int(((pred_infeas == 1) & (true_infeas == 0)).sum().detach().cpu())
    fn_infeas_cls = int(((pred_infeas == 0) & (true_infeas == 1)).sum().detach().cpu())

    precision_infeas = tp_infeas / max(tp_infeas + fp_infeas, 1)
    recall_infeas = tp_infeas / max(tp_infeas + fn_infeas_cls, 1)

    f1_infeas = (
        2 * precision_infeas * recall_infeas
        / max(precision_infeas + recall_infeas, 1e-12)
    )

    return {
        "acc": correct / max(total, 1),
        "total": total,
        "pos": int((y == 1).sum().detach().cpu()),
        "neg": int((y == 0).sum().detach().cpu()),
        "tp_feas": tp_feas,
        "tn_infeas": tn_infeas,
        "fp_feas": fp_feas,
        "fn_infeas": fn_infeas,
        "infeas_precision": precision_infeas,
        "infeas_recall": recall_infeas,
        "infeas_f1": f1_infeas,
        "prob_mean": float(prob_feas.mean().detach().cpu()),
        "prob_min": float(prob_feas.min().detach().cpu()),
        "prob_max": float(prob_feas.max().detach().cpu()),
    }


@torch.no_grad()
def eval_pass(model, loader, device, pos_weight_value: float, neg_weight_value: float, threshold: float):
    model.eval()

    loss_sum = 0.0
    total = 0

    all_logit = []
    all_y = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)

        logit = out["feas_logit"].view(-1)
        y = get_label(batch, device)

        loss = weighted_bce_loss(logit, y, pos_weight_value, neg_weight_value)

        loss_sum += float(loss.detach().cpu()) * y.numel()
        total += y.numel()

        all_logit.append(logit.detach().cpu())
        all_y.append(y.detach().cpu())

    all_logit = torch.cat(all_logit, dim=0)
    all_y = torch.cat(all_y, dim=0)

    metrics = compute_metrics_from_logits(all_logit, all_y, threshold)
    metrics["loss"] = loss_sum / max(total, 1)

    return metrics


def train_one_epoch(model, loader, optimizer, device, pos_weight_value: float, neg_weight_value: float, threshold: float):
    model.train()

    loss_sum = 0.0
    total = 0

    all_logit = []
    all_y = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)

        logit = out["feas_logit"].view(-1)
        y = get_label(batch, device)

        loss = weighted_bce_loss(logit, y, pos_weight_value, neg_weight_value)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        loss_sum += float(loss.detach().cpu()) * y.numel()
        total += y.numel()

        all_logit.append(logit.detach().cpu())
        all_y.append(y.detach().cpu())

    all_logit = torch.cat(all_logit, dim=0)
    all_y = torch.cat(all_y, dim=0)

    metrics = compute_metrics_from_logits(all_logit, all_y, threshold)
    metrics["loss"] = loss_sum / max(total, 1)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed/case30_pure_modes_stable_mixed_graphs.pt")
    parser.add_argument("--run-dir", type=str, default="runs/v2_feasibility_weighted")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--neg-weight", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ds = Case30OPFDataset(args.data, only_feasible=False)
    train_ds, val_ds, split_info = stratified_split(ds, args.train_ratio, args.seed)

    # feasible label = 1, infeasible label = 0
    # Infeasible is rare, so give y=0 samples larger weight.
    if args.neg_weight > 0:
        neg_weight_value = float(args.neg_weight)
    else:
        neg_weight_value = split_info["pos_train"] / max(split_info["neg_train"], 1)

    pos_weight_value = 1.0

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

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

    best_score = -1.0
    metrics_path = run_dir / "metrics.csv"

    print("=== MiniGridSFM30 weighted feasibility training ===")
    print("device:", device)
    print("data:", args.data)
    print("dataset size:", len(ds))
    print("train size:", len(train_ds))
    print("val size:", len(val_ds))
    print("split:", split_info)
    print("parameters:", count_parameters(model))
    print("pos_weight:", pos_weight_value)
    print("neg_weight:", neg_weight_value)
    print("threshold:", args.threshold)
    print("run_dir:", run_dir)
    print()

    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "train_infeas_precision",
        "train_infeas_recall",
        "train_infeas_f1",
        "train_tp_feas",
        "train_tn_infeas",
        "train_fp_feas",
        "train_fn_infeas",
        "train_prob_mean",
        "val_loss",
        "val_acc",
        "val_infeas_precision",
        "val_infeas_recall",
        "val_infeas_f1",
        "val_tp_feas",
        "val_tn_infeas",
        "val_fp_feas",
        "val_fn_infeas",
        "val_prob_mean",
    ]

    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_m = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                pos_weight_value,
                neg_weight_value,
                args.threshold,
            )

            val_m = eval_pass(
                model,
                val_loader,
                device,
                pos_weight_value,
                neg_weight_value,
                args.threshold,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_m["loss"],
                "train_acc": train_m["acc"],
                "train_infeas_precision": train_m["infeas_precision"],
                "train_infeas_recall": train_m["infeas_recall"],
                "train_infeas_f1": train_m["infeas_f1"],
                "train_tp_feas": train_m["tp_feas"],
                "train_tn_infeas": train_m["tn_infeas"],
                "train_fp_feas": train_m["fp_feas"],
                "train_fn_infeas": train_m["fn_infeas"],
                "train_prob_mean": train_m["prob_mean"],
                "val_loss": val_m["loss"],
                "val_acc": val_m["acc"],
                "val_infeas_precision": val_m["infeas_precision"],
                "val_infeas_recall": val_m["infeas_recall"],
                "val_infeas_f1": val_m["infeas_f1"],
                "val_tp_feas": val_m["tp_feas"],
                "val_tn_infeas": val_m["tn_infeas"],
                "val_fp_feas": val_m["fp_feas"],
                "val_fn_infeas": val_m["fn_infeas"],
                "val_prob_mean": val_m["prob_mean"],
            }

            writer.writerow(row)
            f.flush()

            print(
                f"epoch {epoch:03d} | "
                f"train loss {train_m['loss']:.6f} acc {train_m['acc']:.4f} "
                f"infeas P/R/F1 {train_m['infeas_precision']:.4f}/"
                f"{train_m['infeas_recall']:.4f}/{train_m['infeas_f1']:.4f} "
                f"CM(feas_tp={train_m['tp_feas']}, infeas_tn={train_m['tn_infeas']}, "
                f"fp_feas={train_m['fp_feas']}, fn_infeas={train_m['fn_infeas']}) | "
                f"val loss {val_m['loss']:.6f} acc {val_m['acc']:.4f} "
                f"infeas P/R/F1 {val_m['infeas_precision']:.4f}/"
                f"{val_m['infeas_recall']:.4f}/{val_m['infeas_f1']:.4f} "
                f"CM(feas_tp={val_m['tp_feas']}, infeas_tn={val_m['tn_infeas']}, "
                f"fp_feas={val_m['fp_feas']}, fn_infeas={val_m['fn_infeas']})"
            )

            # Use infeasible F1 as the model selection score.
            score = val_m["infeas_f1"]

            if score > best_score:
                best_score = score
                torch.save(
                    {
                        "epoch": epoch,
                        "best_val_infeas_f1": best_score,
                        "model_state_dict": model.state_dict(),
                        "args": vars(args),
                        "split_info": split_info,
                        "pos_weight": pos_weight_value,
                        "neg_weight": neg_weight_value,
                    },
                    run_dir / "best_model.pt",
                )

    print()
    print("best val infeasible F1:", best_score)
    print("saved:", run_dir / "best_model.pt")
    print("metrics:", metrics_path)


if __name__ == "__main__":
    main()