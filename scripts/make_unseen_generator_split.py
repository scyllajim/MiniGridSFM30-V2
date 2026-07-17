from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


def generator_online_mask(graph: Any) -> torch.Tensor:
    if "generator" not in graph.node_types:
        raise ValueError("Graph does not contain generator nodes.")

    x = graph["generator"].x

    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError(
            "generator.x must be a two-dimensional tensor "
            "whose first feature is is_online."
        )

    return x[:, 0] > 0.5


def inspect_graphs(graphs: list[Any]) -> tuple[int, Counter]:
    if not graphs:
        raise ValueError("The source dataset contains no graphs.")

    first_mask = generator_online_mask(graphs[0])
    num_generators = int(first_mask.numel())

    offline_counts: Counter[int] = Counter()

    for graph_index, graph in enumerate(graphs):
        mask = generator_online_mask(graph)

        if int(mask.numel()) != num_generators:
            raise ValueError(
                f"Graph {graph_index} has {mask.numel()} generators, "
                f"expected {num_generators}."
            )

        offline_indices = torch.where(~mask)[0].tolist()

        for generator_index in offline_indices:
            offline_counts[int(generator_index)] += 1

    return num_generators, offline_counts


def choose_holdout_generator(
    offline_counts: Counter,
    requested: int | None,
) -> int:
    if requested is not None:
        if offline_counts.get(requested, 0) <= 0:
            raise ValueError(
                f"Generator {requested} has no offline samples."
            )
        return requested

    candidates = [
        (count, generator_index)
        for generator_index, count in offline_counts.items()
        if count > 0
    ]

    if not candidates:
        raise ValueError(
            "No generator outage samples were found. "
            "Check generator.x[:, 0] encoding."
        )

    # Choose the most frequently offline generator.
    # On ties, choose the smaller generator index.
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--manifest-out", required=True)

    parser.add_argument(
        "--holdout-generator",
        type=int,
        default=None,
        help=(
            "Generator node index to hold out. "
            "Default: generator with the most outage samples."
        ),
    )

    parser.add_argument(
        "--min-val-samples",
        type=int,
        default=100,
    )

    args = parser.parse_args()

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
            "Expected processed dataset to be a dictionary."
        )

    graphs = payload.get("graphs")

    if not isinstance(graphs, list):
        raise TypeError(
            "Expected processed dataset payload['graphs'] to be a list."
        )

    num_generators, offline_counts = inspect_graphs(graphs)

    holdout_generator = choose_holdout_generator(
        offline_counts,
        args.holdout_generator,
    )

    train_graphs = []
    val_graphs = []

    train_source_indices = []
    val_source_indices = []

    for source_index, graph in enumerate(graphs):
        mask = generator_online_mask(graph)

        if bool(mask[holdout_generator]):
            train_graphs.append(graph)
            train_source_indices.append(source_index)
        else:
            val_graphs.append(graph)
            val_source_indices.append(source_index)

    if len(train_graphs) == 0:
        raise RuntimeError("Generated training split is empty.")

    if len(val_graphs) < args.min_val_samples:
        raise RuntimeError(
            f"Validation split has only {len(val_graphs)} samples; "
            f"minimum required is {args.min_val_samples}."
        )

    # Strict leakage check.
    for graph_index, graph in enumerate(train_graphs):
        mask = generator_online_mask(graph)
        if not bool(mask[holdout_generator]):
            raise AssertionError(
                f"Leakage: train graph {graph_index} contains "
                f"held-out generator {holdout_generator} offline."
            )

    for graph_index, graph in enumerate(val_graphs):
        mask = generator_online_mask(graph)
        if bool(mask[holdout_generator]):
            raise AssertionError(
                f"Invalid validation graph {graph_index}: "
                f"held-out generator {holdout_generator} is online."
            )

    offline_counts_dict = {
        str(index): int(offline_counts.get(index, 0))
        for index in range(num_generators)
    }

    common_metadata = {
        "case": payload.get("case", "case30"),
        "source": str(source_path),
        "split_type": "unseen_generator",
        "held_out_generator_index": holdout_generator,
        "num_generators": num_generators,
        "generator_offline_counts": offline_counts_dict,
    }

    train_payload = {
        "case": payload.get("case", "case30"),
        "graphs": train_graphs,
        "raw_info": {
            **payload.get("raw_info", {}),
            **common_metadata,
            "partition": "train",
            "source_indices": train_source_indices,
        },
    }

    val_payload = {
        "case": payload.get("case", "case30"),
        "graphs": val_graphs,
        "raw_info": {
            **payload.get("raw_info", {}),
            **common_metadata,
            "partition": "validation",
            "source_indices": val_source_indices,
        },
    }

    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(train_payload, train_path)
    torch.save(val_payload, val_path)

    manifest = {
        **common_metadata,
        "source_graphs": len(graphs),
        "train_graphs": len(train_graphs),
        "validation_graphs": len(val_graphs),
        "train_file": str(train_path),
        "validation_file": str(val_path),
        "train_source_indices": train_source_indices,
        "validation_source_indices": val_source_indices,
        "leakage_check_passed": True,
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("=== Unseen-generator split created ===")
    print("source:", source_path)
    print("source graphs:", len(graphs))
    print("number of generators:", num_generators)
    print("offline sample counts:")

    for generator_index in range(num_generators):
        count = offline_counts.get(generator_index, 0)
        selected = (
            "  <-- HELD OUT"
            if generator_index == holdout_generator
            else ""
        )
        print(
            f"  generator {generator_index}: "
            f"{count} offline samples{selected}"
        )

    print("train graphs:", len(train_graphs))
    print("validation graphs:", len(val_graphs))
    print("train file:", train_path)
    print("validation file:", val_path)
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
