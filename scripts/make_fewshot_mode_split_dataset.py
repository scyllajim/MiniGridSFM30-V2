from __future__ import annotations

import argparse
import pickle
import random
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
    ap.add_argument("--mode", required=True)
    ap.add_argument("--mode-train-frac", type=float, required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--only-feasible", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (0.0 < args.mode_train_frac < 1.0):
        raise ValueError("--mode-train-frac must be between 0 and 1")

    rng = random.Random(args.seed)

    raw = load_raw(args.raw)
    samples = raw["samples"]

    mode_samples = []
    other_samples = []

    mode_counter_all = Counter()
    mode_counter_used = Counter()

    for s in samples:
        mode = s.get("perturb_mode", "unknown")
        mode_counter_all[mode] += 1

        if args.only_feasible and not bool(s.get("feasible", False)):
            continue

        mode_counter_used[mode] += 1

        if mode == args.mode:
            mode_samples.append(s)
        else:
            other_samples.append(s)

    if len(mode_samples) == 0:
        raise RuntimeError(f"No samples found for mode={args.mode}")

    rng.shuffle(mode_samples)

    n_mode_train = int(round(len(mode_samples) * args.mode_train_frac))
    n_mode_train = max(1, min(n_mode_train, len(mode_samples) - 1))

    mode_train = mode_samples[:n_mode_train]
    mode_val = mode_samples[n_mode_train:]

    train_samples = other_samples + mode_train
    val_samples = mode_val

    print("=== Few-shot mode split ===")
    print("raw:", args.raw)
    print("mode:", args.mode)
    print("mode_train_frac:", args.mode_train_frac)
    print("seed:", args.seed)
    print("only_feasible:", args.only_feasible)
    print("all modes:", dict(mode_counter_all))
    print("used modes:", dict(mode_counter_used))
    print("other train samples:", len(other_samples))
    print(f"{args.mode} total:", len(mode_samples))
    print(f"{args.mode} train:", len(mode_train))
    print(f"{args.mode} val:", len(mode_val))
    print("train samples:", len(train_samples))
    print("val samples:", len(val_samples))

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

    raw_info = {
        "raw": args.raw,
        "mode": args.mode,
        "mode_train_frac": args.mode_train_frac,
        "seed": args.seed,
        "only_feasible": args.only_feasible,
        "n_other_train": len(other_samples),
        "n_mode_total": len(mode_samples),
        "n_mode_train": len(mode_train),
        "n_mode_val": len(mode_val),
        "n_train": len(train_graphs),
        "n_val": len(val_graphs),
        "mode_counter_all": dict(mode_counter_all),
        "mode_counter_used": dict(mode_counter_used),
    }

    train_payload = {
        "case": raw.get("case", "case30"),
        "split": "fewshot_mode_train",
        "fewshot_mode": args.mode,
        "graphs": train_graphs,
        "raw_info": raw_info,
    }

    val_payload = {
        "case": raw.get("case", "case30"),
        "split": "fewshot_mode_val",
        "fewshot_mode": args.mode,
        "graphs": val_graphs,
        "raw_info": raw_info,
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
