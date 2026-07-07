from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)


MODES = ["base", "loads", "costs", "derate", "killgen", "vsqueeze"]


def get_float(d, k, default=0.0):
    try:
        v = d.get(k, default)
        if v is None:
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def make_features(sample: dict):
    mode = sample.get("perturb_mode", "unknown")
    pp = sample.get("perturb_params", {}) or {}

    feats = []

    # one-hot perturb mode
    for m in MODES:
        feats.append(1.0 if mode == m else 0.0)

    # generic parameters
    feats.append(get_float(pp, "global_scale", 1.0))
    feats.append(get_float(pp, "scale_min", 1.0))
    feats.append(get_float(pp, "scale_max", 1.0))
    feats.append(get_float(pp, "scale_mean", 1.0))

    feats.append(get_float(pp, "n_derated", 0.0))
    feats.append(get_float(pp, "n_line", 0.0))

    factors = pp.get("factors", [])
    if factors:
        factors = [float(x) for x in factors]
        feats.append(float(np.min(factors)))
        feats.append(float(np.mean(factors)))
    else:
        feats.append(1.0)
        feats.append(1.0)

    feats.append(get_float(pp, "n_killed", 0.0))
    killed = pp.get("killed_gen_indices", [])
    for g in range(5):
        feats.append(1.0 if g in killed else 0.0)

    feats.append(get_float(pp, "n_squeezed", 0.0))
    feats.append(get_float(pp, "vsqueeze_eps", 0.0))

    return feats


def stratified_split(y, train_ratio: float, seed: int):
    pos = [i for i, v in enumerate(y) if v == 1]
    neg = [i for i, v in enumerate(y) if v == 0]

    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_pos_train = int(len(pos) * train_ratio)
    n_neg_train = int(len(neg) * train_ratio)

    train_idx = pos[:n_pos_train] + neg[:n_neg_train]
    val_idx = pos[n_pos_train:] + neg[n_neg_train:]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return train_idx, val_idx


def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    # y=0 means infeasible. Treat infeasible as positive alarm class.
    y_true_infeas = 1 - np.asarray(y_true)
    y_pred_infeas = 1 - np.asarray(y_pred)

    p, r, f1, _ = precision_recall_fscore_support(
        y_true_infeas,
        y_pred_infeas,
        average="binary",
        zero_division=0,
    )

    print()
    print("===", name, "===")
    print("accuracy:", f"{acc:.4f}")
    print("confusion matrix labels=[feasible(1), infeasible(0)]")
    print(cm)
    print("infeasible precision:", f"{p:.4f}")
    print("infeasible recall:   ", f"{r:.4f}")
    print("infeasible f1:       ", f"{f1:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["infeasible", "feasible"], zero_division=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/case30_pure_modes_stable.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    with open(args.data, "rb") as f:
        obj = pickle.load(f)

    samples = obj["samples"]

    X = np.asarray([make_features(s) for s in samples], dtype=np.float32)
    y = np.asarray([1 if s.get("feasible", False) else 0 for s in samples], dtype=np.int64)

    train_idx, val_idx = stratified_split(y, args.train_ratio, args.seed)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    print("data:", args.data)
    print("n:", len(y))
    print("train:", len(train_idx), "val:", len(val_idx))
    print("train feasible/infeasible:", int((y_train == 1).sum()), int((y_train == 0).sum()))
    print("val feasible/infeasible:", int((y_val == 1).sum()), int((y_val == 0).sum()))

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=args.seed,
    )
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        class_weight="balanced",
        random_state=args.seed,
    )
    rf.fit(X_train, y_train)

    report("LogisticRegression train", y_train, lr.predict(X_train))
    report("LogisticRegression val", y_val, lr.predict(X_val))

    report("RandomForest train", y_train, rf.predict(X_train))
    report("RandomForest val", y_val, rf.predict(X_val))

    print()
    print("RandomForest feature importances:")
    names = (
        [f"mode_{m}" for m in MODES]
        + [
            "global_scale",
            "scale_min",
            "scale_max",
            "scale_mean",
            "n_derated",
            "n_line",
            "derate_factor_min",
            "derate_factor_mean",
            "n_killed",
        ]
        + [f"killed_gen_{g}" for g in range(5)]
        + ["n_squeezed", "vsqueeze_eps"]
    )

    order = np.argsort(rf.feature_importances_)[::-1]
    for i in order[:20]:
        print(f"{names[i]:<24} {rf.feature_importances_[i]:.6f}")


if __name__ == "__main__":
    main()
