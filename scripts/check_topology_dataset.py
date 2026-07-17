from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader


def scalar_int(value) -> int:
    if torch.is_tensor(value):
        return int(
            value.reshape(-1)[0].item()
        )

    return int(value)


def check_dataset(
    path: Path,
    expected_partition: str,
    heldout_position: int,
) -> tuple[int, set[int]]:
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    graphs = payload["graphs"]

    if not graphs:
        raise RuntimeError(
            f"{path} is empty."
        )

    positions = {
        scalar_int(
            getattr(
                graph,
                "topology_outage_line_position",
            )
        )
        for graph in graphs
    }

    branch_counts = sorted(
        {
            int(
                graph["branch_ac"].x.shape[0]
            )
            for graph in graphs
        }
    )

    cycle_counts = sorted(
        {
            int(
                graph["cycle"].x.shape[0]
            )
            for graph in graphs
        }
    )

    print()
    print(path)
    print("graphs:", len(graphs))
    print(
        "topology positions:",
        sorted(positions),
    )
    print(
        "branch counts:",
        branch_counts,
    )
    print(
        "cycle counts:",
        cycle_counts,
    )

    if expected_partition == "train":
        if heldout_position in positions:
            raise AssertionError(
                "Held-out topology leaked into train."
            )
    else:
        if positions != {heldout_position}:
            raise AssertionError(
                "Validation contains unexpected topology."
            )

    loader = DataLoader(
        graphs[: min(16, len(graphs))],
        batch_size=min(
            8,
            len(graphs),
        ),
        shuffle=False,
    )

    batch = next(iter(loader))

    for node_type in (
        "bus",
        "generator",
        "branch_ac",
    ):
        if not torch.isfinite(
            batch[node_type].x
        ).all():
            raise FloatingPointError(
                f"Non-finite batched {node_type}.x"
            )

    print(
        "PyG batch check: passed"
    )

    return len(graphs), positions


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True,
    )

    parser.add_argument(
        "--validation",
        required=True,
    )

    parser.add_argument(
        "--manifest",
        required=True,
    )

    args = parser.parse_args()

    manifest_path = Path(args.manifest)

    with manifest_path.open(
        encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    heldout_position = int(
        manifest["heldout_line_position"]
    )

    train_count, _ = check_dataset(
        Path(args.train),
        "train",
        heldout_position,
    )

    val_count, _ = check_dataset(
        Path(args.validation),
        "validation",
        heldout_position,
    )

    if train_count != int(
        manifest["train_graphs"]
    ):
        raise AssertionError(
            "Train count differs from manifest."
        )

    if val_count != int(
        manifest["validation_graphs"]
    ):
        raise AssertionError(
            "Validation count differs from manifest."
        )

    print()
    print(
        "held-out line position:",
        heldout_position,
    )
    print(
        "held-out line index:",
        manifest["heldout_line_index"],
    )
    print(
        "held-out buses:",
        f'{manifest["heldout_from_bus"]}'
        f'->{manifest["heldout_to_bus"]}',
    )
    print(
        "leakage check:",
        manifest["leakage_check_passed"],
    )
    print(
        "dataset validation complete"
    )


if __name__ == "__main__":
    main()
