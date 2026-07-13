from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from collections import Counter

import torch

from minigridsfm30.graph_builder import sample_to_heterodata


def load_raw(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--holdout-mode", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--only-feasible", action="store_true")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    args = ap.parse_args()

    raw = load_raw(args.raw)
    samples = raw["samples"]

    mode_counter_all = Counter()
    mode_counter_used = Counter()
    mode_counter_train = Counter()
    mode_counter_val = Counter()

    train_samples = []
    val_samples = []

    for s in samples:
        mode = s.get("perturb_mode", "unknown")
        mode_counter_all[mode] += 1

        if args.only_feasible and not bool(s.get("feasible", False)):
            continue

        mode_counter_used[mode] += 1

        if mode == args.holdout_mode:
            val_samples.append(s)
            mode_counter_val[mode] += 1
        else:
            train_samples.append(s)
            mode_counter_train[mode] += 1

    if args.max_train is not None:
        train_samples = train_samples[:args.max_train]
    if args.max_val is not None:
        val_samples = val_samples[:args.max_val]

    print("=== Mode split ===")
    print("raw:", args.raw)
    print("holdout_mode:", args.holdout_mode)
    print("only_feasible:", args.only_feasible)
    print("all modes:", dict(mode_counter_all))
    print("used modes:", dict(mode_counter_used))
    print("train modes:", dict(mode_counter_train))
    print("val modes:", dict(mode_counter_val))
    print("train samples:", len(train_samples))
    print("val samples:", len(val_samples))

    if len(train_samples) == 0:
        raise RuntimeError("No train samples after split.")
    if len(val_samples) == 0:
        raise RuntimeError("No val samples after split.")

    train_graphs = []
    for i, s in enumerate(train_samples):
        g = sample_to_heterodata(s)
        g.split_index = int(i)
        train_graphs.append(g)

    val_graphs = []
    for i, s in enumerate(val_samples):
        g = sample_to_heterodata(s)
        g.split_index = int(i)
        val_graphs.append(g)

    train_payload = {
        "case": raw.get("case", "case30"),
        "split": "mode_holdout_train",
        "holdout_mode": args.holdout_mode,
        "graphs": train_graphs,
        "raw_info": {
            "raw": args.raw,
            "only_feasible": args.only_feasible,
            "n_train": len(train_graphs),
            "n_val": len(val_graphs),
            "mode_counter_all": dict(mode_counter_all),
            "mode_counter_used": dict(mode_counter_used),
            "mode_counter_train": dict(mode_counter_train),
            "mode_counter_val": dict(mode_counter_val),
        },
    }

    val_payload = {
        "case": raw.get("case", "case30"),
        "split": "mode_holdout_val",
        "holdout_mode": args.holdout_mode,
        "graphs": val_graphs,
        "raw_info": train_payload["raw_info"],
    }

    train_out = Path(args.train_out)
    val_out = Path(args.val_out)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out.parent.mkdir(parents=True, exist_ok=True)

    torch.save(train_payload, train_out)
    torch.save(val_payload, val_out)

    print("saved train:", train_out)
    print("saved val:", val_out)


if __name__ == "__main__":
    main()
