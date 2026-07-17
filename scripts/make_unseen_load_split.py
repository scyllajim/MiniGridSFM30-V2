from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def get_total_active_load_mw(graph: Any) -> float:
    if "bus" not in graph.node_types:
        raise ValueError("Graph does not contain bus nodes.")

    bus_x = graph["bus"].x

    if bus_x.ndim != 2 or bus_x.shape[1] < 5:
        raise ValueError(
            "Expected bus.x to contain active load in feature index 4."
        )

    pd_pu = bus_x[:, 4]

    if not torch.isfinite(pd_pu).all():
        raise FloatingPointError(
            "Non-finite values found in bus active-load features."
        )

    sn_mva = getattr(graph, "sn_mva", 100.0)

    if torch.is_tensor(sn_mva):
        sn_mva = float(sn_mva.reshape(-1)[0].item())
    else:
        sn_mva = float(sn_mva)

    return float(pd_pu.sum().item() * sn_mva)


def describe(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)

    return {
        "count": int(array.size),
        "min_mw": float(array.min()),
        "max_mw": float(array.max()),
        "mean_mw": float(array.mean()),
        "std_mw": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "q10_mw": float(np.quantile(array, 0.10)),
        "q25_mw": float(np.quantile(array, 0.25)),
        "q50_mw": float(np.quantile(array, 0.50)),
        "q70_mw": float(np.quantile(array, 0.70)),
        "q75_mw": float(np.quantile(array, 0.75)),
        "q85_mw": float(np.quantile(array, 0.85)),
        "q90_mw": float(np.quantile(array, 0.90)),
        "q95_mw": float(np.quantile(array, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--manifest-out", required=True)

    parser.add_argument(
        "--train-max-quantile",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--val-min-quantile",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--min-train-samples",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--min-val-samples",
        type=int,
        default=300,
    )

    args = parser.parse_args()

    if not 0.0 < args.train_max_quantile < 1.0:
        raise ValueError(
            "--train-max-quantile must be between 0 and 1."
        )

    if not 0.0 < args.val_min_quantile < 1.0:
        raise ValueError(
            "--val-min-quantile must be between 0 and 1."
        )

    if args.train_max_quantile >= args.val_min_quantile:
        raise ValueError(
            "Training quantile must be lower than validation quantile."
        )

    source_path = Path(args.source)
    train_path = Path(args.train_out)
    val_path = Path(args.val_out)
    manifest_path = Path(args.manifest_out)

    payload = torch.load(
        source_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise TypeError(
            "Expected the processed dataset to be a dictionary."
        )

    graphs = payload.get("graphs")

    if not isinstance(graphs, list) or not graphs:
        raise TypeError(
            "Expected payload['graphs'] to be a non-empty list."
        )

    total_loads = [
        get_total_active_load_mw(graph)
        for graph in graphs
    ]

    loads_array = np.asarray(
        total_loads,
        dtype=np.float64,
    )

    train_threshold = float(
        np.quantile(
            loads_array,
            args.train_max_quantile,
        )
    )

    val_threshold = float(
        np.quantile(
            loads_array,
            args.val_min_quantile,
        )
    )

    train_graphs = []
    val_graphs = []

    train_indices = []
    val_indices = []
    gap_indices = []

    train_loads = []
    val_loads = []
    gap_loads = []

    for index, (graph, total_load) in enumerate(
        zip(graphs, total_loads)
    ):
        if total_load <= train_threshold:
            train_graphs.append(graph)
            train_indices.append(index)
            train_loads.append(total_load)

        elif total_load >= val_threshold:
            val_graphs.append(graph)
            val_indices.append(index)
            val_loads.append(total_load)

        else:
            gap_indices.append(index)
            gap_loads.append(total_load)

    if len(train_graphs) < args.min_train_samples:
        raise RuntimeError(
            f"Training split has only {len(train_graphs)} samples; "
            f"minimum required is {args.min_train_samples}."
        )

    if len(val_graphs) < args.min_val_samples:
        raise RuntimeError(
            f"Validation split has only {len(val_graphs)} samples; "
            f"minimum required is {args.min_val_samples}."
        )

    if max(train_loads) >= min(val_loads):
        raise AssertionError(
            "Load-range leakage detected: training and validation "
            "load ranges overlap."
        )

    common_metadata = {
        "case": payload.get("case", "case30"),
        "source": str(source_path),
        "split_type": "unseen_high_load_range",
        "train_max_quantile": args.train_max_quantile,
        "val_min_quantile": args.val_min_quantile,
        "train_max_load_mw": train_threshold,
        "val_min_load_mw": val_threshold,
        "gap_width_mw": val_threshold - train_threshold,
    }

    train_payload = {
        "case": payload.get("case", "case30"),
        "graphs": train_graphs,
        "raw_info": {
            **payload.get("raw_info", {}),
            **common_metadata,
            "partition": "train",
            "source_indices": train_indices,
        },
    }

    val_payload = {
        "case": payload.get("case", "case30"),
        "graphs": val_graphs,
        "raw_info": {
            **payload.get("raw_info", {}),
            **common_metadata,
            "partition": "validation",
            "source_indices": val_indices,
        },
    }

    train_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    val_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(train_payload, train_path)
    torch.save(val_payload, val_path)

    manifest = {
        **common_metadata,
        "source_graphs": len(graphs),
        "train_graphs": len(train_graphs),
        "gap_graphs": len(gap_indices),
        "validation_graphs": len(val_graphs),
        "source_load_statistics": describe(total_loads),
        "train_load_statistics": describe(train_loads),
        "gap_load_statistics": describe(gap_loads),
        "validation_load_statistics": describe(val_loads),
        "train_file": str(train_path),
        "validation_file": str(val_path),
        "train_source_indices": train_indices,
        "gap_source_indices": gap_indices,
        "validation_source_indices": val_indices,
        "leakage_check_passed": True,
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("=== Unseen-load-range split created ===")
    print("source:", source_path)
    print("source graphs:", len(graphs))
    print(
        "training quantile:",
        args.train_max_quantile,
    )
    print(
        "validation quantile:",
        args.val_min_quantile,
    )
    print(
        "train max load:",
        f"{train_threshold:.4f} MW",
    )
    print(
        "validation min load:",
        f"{val_threshold:.4f} MW",
    )
    print(
        "load gap:",
        f"{val_threshold - train_threshold:.4f} MW",
    )
    print("train graphs:", len(train_graphs))
    print("gap graphs:", len(gap_indices))
    print("validation graphs:", len(val_graphs))
    print(
        "actual train range:",
        f"{min(train_loads):.4f}",
        "to",
        f"{max(train_loads):.4f} MW",
    )
    print(
        "actual validation range:",
        f"{min(val_loads):.4f}",
        "to",
        f"{max(val_loads):.4f} MW",
    )
    print("leakage check: passed")
    print("train file:", train_path)
    print("validation file:", val_path)
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
