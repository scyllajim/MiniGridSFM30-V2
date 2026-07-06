from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from minigridsfm30.graph_builder import sample_to_heterodata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, default="data/raw/case30_1000_v2.pkl")
    parser.add_argument("--out", type=str, default="data/processed/case30_1000_v2_graphs.pt")
    parser.add_argument("--only-feasible", action="store_true")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(raw_path, "rb") as f:
        raw = pickle.load(f)

    samples = raw["samples"]

    if args.only_feasible:
        samples = [s for s in samples if s.get("feasible", False)]

    print("raw:", raw_path)
    print("out:", out_path)
    print("samples:", len(samples))

    graphs = []

    for s in tqdm(samples, desc="convert samples to HeteroData"):
        data = sample_to_heterodata(s)
        graphs.append(data)

    obj = {
        "case": raw.get("case"),
        "n_graphs": len(graphs),
        "raw_info": {
            "n_requested": raw.get("n_requested"),
            "n_success": raw.get("n_success"),
            "n_failed": raw.get("n_failed"),
            "n_saved": raw.get("n_saved"),
            "settings": raw.get("settings"),
            "sample_format": raw.get("sample_format"),
        },
        "graphs": graphs,
    }

    torch.save(obj, out_path)

    print()
    print("saved:", out_path)
    print("n_graphs:", len(graphs))


if __name__ == "__main__":
    main()
