from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import pickle
import random
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minigridsfm30.graph_builder import (
    build_case30_net,
    net_to_heterodata,
    run_ac_opf,
)


@dataclass(frozen=True)
class Scenario:
    scenario_id: int
    seed: int
    outage_line_position: int | None
    load_scale: float
    reactive_scale: float
    generator_vm_scale: float


@dataclass
class ScenarioResult:
    scenario_id: int
    success: bool
    outage_line_position: int | None
    outage_line_index: int | None
    from_bus: int | None
    to_bus: int | None
    load_scale: float
    reactive_scale: float
    generator_vm_scale: float
    graph: Any | None
    error_type: str | None
    error_message: str | None


def is_connected_after_line_outage(
    line_position: int,
) -> bool:
    net = build_case30_net()

    line_indices = list(net.line.index)
    line_index = int(line_indices[line_position])

    active_edges: list[tuple[int, int]] = []

    for idx, row in net.line.iterrows():
        if int(idx) == line_index:
            continue

        if "in_service" in row and not bool(row["in_service"]):
            continue

        active_edges.append(
            (
                int(row["from_bus"]),
                int(row["to_bus"]),
            )
        )

    buses = [int(x) for x in net.bus.index]

    if not buses:
        return False

    adjacency: dict[int, set[int]] = {
        bus: set()
        for bus in buses
    }

    for left, right in active_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    visited = set()
    stack = [buses[0]]

    while stack:
        bus = stack.pop()

        if bus in visited:
            continue

        visited.add(bus)
        stack.extend(adjacency[bus] - visited)

    return len(visited) == len(buses)


def apply_random_operating_point(
    net: Any,
    scenario: Scenario,
) -> None:
    rng = np.random.default_rng(scenario.seed)

    if len(net.load) > 0:
        local_variation = rng.uniform(
            0.96,
            1.04,
            size=len(net.load),
        )

        p_scale = (
            scenario.load_scale
            * local_variation
        )

        q_scale = (
            scenario.reactive_scale
            * local_variation
        )

        net.load.loc[:, "p_mw"] = (
            net.load["p_mw"].to_numpy(dtype=float)
            * p_scale
        )

        net.load.loc[:, "q_mvar"] = (
            net.load["q_mvar"].to_numpy(dtype=float)
            * q_scale
        )

    if len(net.gen) > 0 and "vm_pu" in net.gen.columns:
        vm_noise = rng.uniform(
            0.995,
            1.005,
            size=len(net.gen),
        )

        net.gen.loc[:, "vm_pu"] = np.clip(
            net.gen["vm_pu"].to_numpy(dtype=float)
            * scenario.generator_vm_scale
            * vm_noise,
            0.95,
            1.05,
        )

    if (
        len(net.ext_grid) > 0
        and "vm_pu" in net.ext_grid.columns
    ):
        net.ext_grid.loc[:, "vm_pu"] = np.clip(
            net.ext_grid["vm_pu"].to_numpy(dtype=float)
            * scenario.generator_vm_scale,
            0.97,
            1.03,
        )


def solve_scenario(
    scenario: Scenario,
) -> ScenarioResult:
    try:
        net = build_case30_net()

        outage_line_index = None
        from_bus = None
        to_bus = None

        if scenario.outage_line_position is not None:
            line_indices = list(net.line.index)

            outage_line_index = int(
                line_indices[
                    scenario.outage_line_position
                ]
            )

            from_bus = int(
                net.line.loc[
                    outage_line_index,
                    "from_bus",
                ]
            )

            to_bus = int(
                net.line.loc[
                    outage_line_index,
                    "to_bus",
                ]
            )

            net.line.loc[
                outage_line_index,
                "in_service",
            ] = False

        apply_random_operating_point(
            net,
            scenario,
        )

        run_ac_opf(
            net,
            verbose=False,
        )

        graph = net_to_heterodata(
            net,
            require_solution=True,
        )

        graph.scenario_id = int(
            scenario.scenario_id
        )

        graph.topology_outage_line_position = (
            -1
            if scenario.outage_line_position is None
            else int(
                scenario.outage_line_position
            )
        )

        graph.topology_outage_line_index = (
            -1
            if outage_line_index is None
            else int(outage_line_index)
        )

        graph.load_scale = float(
            scenario.load_scale
        )

        graph.reactive_scale = float(
            scenario.reactive_scale
        )

        graph.generator_vm_scale = float(
            scenario.generator_vm_scale
        )

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            success=True,
            outage_line_position=(
                scenario.outage_line_position
            ),
            outage_line_index=outage_line_index,
            from_bus=from_bus,
            to_bus=to_bus,
            load_scale=scenario.load_scale,
            reactive_scale=scenario.reactive_scale,
            generator_vm_scale=(
                scenario.generator_vm_scale
            ),
            graph=graph,
            error_type=None,
            error_message=None,
        )

    except Exception as exc:
        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            success=False,
            outage_line_position=(
                scenario.outage_line_position
            ),
            outage_line_index=None,
            from_bus=None,
            to_bus=None,
            load_scale=scenario.load_scale,
            reactive_scale=scenario.reactive_scale,
            generator_vm_scale=(
                scenario.generator_vm_scale
            ),
            graph=None,
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
        )


def create_scenarios(
    seed: int,
    base_samples: int,
    samples_per_line: int,
    eligible_line_positions: list[int],
) -> list[Scenario]:
    rng = random.Random(seed)
    scenarios: list[Scenario] = []
    scenario_id = 0

    def add_scenario(
        outage_line_position: int | None,
    ) -> None:
        nonlocal scenario_id

        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                seed=rng.randrange(
                    0,
                    2**31 - 1,
                ),
                outage_line_position=(
                    outage_line_position
                ),
                load_scale=rng.uniform(
                    0.88,
                    1.12,
                ),
                reactive_scale=rng.uniform(
                    0.88,
                    1.12,
                ),
                generator_vm_scale=rng.uniform(
                    0.995,
                    1.005,
                ),
            )
        )

        scenario_id += 1

    for _ in range(base_samples):
        add_scenario(None)

    for line_position in eligible_line_positions:
        for _ in range(samples_per_line):
            add_scenario(line_position)

    rng.shuffle(scenarios)

    return scenarios


def graph_outage_position(
    graph: Any,
) -> int:
    value = getattr(
        graph,
        "topology_outage_line_position",
        -1,
    )

    if torch.is_tensor(value):
        return int(
            value.reshape(-1)[0].item()
        )

    return int(value)


def validate_graph(
    graph: Any,
) -> None:
    num_branches = int(
        graph["branch_ac"].x.shape[0]
    )

    if num_branches <= 0:
        raise ValueError(
            "Graph contains no branch_ac nodes."
        )

    for node_type in (
        "bus",
        "generator",
        "branch_ac",
    ):
        x = graph[node_type].x

        if not torch.isfinite(x).all():
            raise FloatingPointError(
                f"Non-finite {node_type}.x"
            )

        if hasattr(
            graph[node_type],
            "y",
        ):
            y = graph[node_type].y

            if not torch.isfinite(y).all():
                raise FloatingPointError(
                    f"Non-finite {node_type}.y"
                )

    bus_to_branch = graph[
        ("bus", "endpoint_of", "branch_ac")
    ].edge_index

    branch_to_bus = graph[
        ("branch_ac", "endpoint_of", "bus")
    ].edge_index

    if int(bus_to_branch[1].max()) >= num_branches:
        raise IndexError(
            "bus->branch edge references invalid node."
        )

    if int(branch_to_bus[0].max()) >= num_branches:
        raise IndexError(
            "branch->bus edge references invalid node."
        )


def strip_graph_metadata(
    graph: Any,
) -> Any:
    # Keep only scalar attributes with consistent keys so PyG batching
    # cannot fail because different graph objects expose different fields.
    required = {
        "scenario_id": int(
            getattr(graph, "scenario_id")
        ),
        "topology_outage_line_position": int(
            graph_outage_position(graph)
        ),
        "topology_outage_line_index": int(
            getattr(
                graph,
                "topology_outage_line_index",
                -1,
            )
        ),
        "load_scale": float(
            getattr(graph, "load_scale")
        ),
        "reactive_scale": float(
            getattr(graph, "reactive_scale")
        ),
        "generator_vm_scale": float(
            getattr(
                graph,
                "generator_vm_scale",
            )
        ),
    }

    for key, value in required.items():
        setattr(graph, key, value)

    return graph


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-out",
        required=True,
    )

    parser.add_argument(
        "--train-out",
        required=True,
    )

    parser.add_argument(
        "--val-out",
        required=True,
    )

    parser.add_argument(
        "--manifest-out",
        required=True,
    )

    parser.add_argument(
        "--base-samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--samples-per-line",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--min-heldout-samples",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260717,
    )

    args = parser.parse_args()

    base_net = build_case30_net()

    line_records = []

    for position, (line_index, row) in enumerate(
        base_net.line.iterrows()
    ):
        connected = is_connected_after_line_outage(
            position
        )

        line_records.append(
            {
                "line_position": position,
                "line_index": int(line_index),
                "from_bus": int(
                    row["from_bus"]
                ),
                "to_bus": int(
                    row["to_bus"]
                ),
                "connected_after_outage": connected,
            }
        )

    eligible_positions = [
        record["line_position"]
        for record in line_records
        if record["connected_after_outage"]
    ]

    print(
        "case30 lines:",
        len(line_records),
    )

    print(
        "connected N-1 candidates:",
        len(eligible_positions),
    )

    scenarios = create_scenarios(
        seed=args.seed,
        base_samples=args.base_samples,
        samples_per_line=(
            args.samples_per_line
        ),
        eligible_line_positions=(
            eligible_positions
        ),
    )

    print(
        "total OPF attempts:",
        len(scenarios),
    )

    results: list[ScenarioResult] = []

    if args.workers <= 1:
        print(
            "execution mode: serial",
            flush=True,
        )

        for completed, scenario in enumerate(
            scenarios,
            start=1,
        ):
            result = solve_scenario(
                scenario
            )

            results.append(result)

            if (
                completed % 100 == 0
                or completed == len(scenarios)
            ):
                successes = sum(
                    item.success
                    for item in results
                )

                print(
                    f"completed {completed}/"
                    f"{len(scenarios)}, "
                    f"success={successes}, "
                    f"failed={completed - successes}",
                    flush=True,
                )

    else:
        print(
            f"execution mode: process pool, "
            f"workers={args.workers}",
            flush=True,
        )

        # Use spawn instead of Linux fork.
        mp_context = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp_context,
        ) as executor:
            future_to_id = {
                executor.submit(
                    solve_scenario,
                    scenario,
                ): scenario.scenario_id
                for scenario in scenarios
            }

            completed = 0

            for future in as_completed(
                future_to_id
            ):
                scenario_id = future_to_id[
                    future
                ]

                try:
                    result = future.result()
                except BaseException as exc:
                    print(
                        f"worker failure for scenario "
                        f"{scenario_id}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    raise

                results.append(result)
                completed += 1

                if (
                    completed % 100 == 0
                    or completed == len(scenarios)
                ):
                    successes = sum(
                        item.success
                        for item in results
                    )

                    print(
                        f"completed {completed}/"
                        f"{len(scenarios)}, "
                        f"success={successes}, "
                        f"failed={completed - successes}",
                        flush=True,
                    )

    results.sort(
        key=lambda item: item.scenario_id
    )

    successful = [
        item
        for item in results
        if item.success
        and item.graph is not None
    ]

    failed = [
        item
        for item in results
        if not item.success
    ]

    for item in successful:
        validate_graph(item.graph)
        strip_graph_metadata(item.graph)

    success_by_outage: Counter[int] = Counter()

    for item in successful:
        position = (
            -1
            if item.outage_line_position is None
            else int(
                item.outage_line_position
            )
        )

        success_by_outage[position] += 1

    heldout_candidates = [
        (
            count,
            position,
        )
        for position, count
        in success_by_outage.items()
        if (
            position >= 0
            and count >= args.min_heldout_samples
        )
    ]

    if not heldout_candidates:
        raise RuntimeError(
            "No line has enough successful outage samples "
            f"for held-out validation. Required "
            f"{args.min_heldout_samples}."
        )

    heldout_candidates.sort(
        key=lambda pair: (
            -pair[0],
            pair[1],
        )
    )

    heldout_count, heldout_position = (
        heldout_candidates[0]
    )

    heldout_record = next(
        record
        for record in line_records
        if record["line_position"]
        == heldout_position
    )

    train_graphs = []
    val_graphs = []

    train_scenario_ids = []
    val_scenario_ids = []

    for item in successful:
        position = (
            -1
            if item.outage_line_position is None
            else int(
                item.outage_line_position
            )
        )

        if position == heldout_position:
            val_graphs.append(item.graph)
            val_scenario_ids.append(
                item.scenario_id
            )
        else:
            train_graphs.append(item.graph)
            train_scenario_ids.append(
                item.scenario_id
            )

    train_positions = {
        graph_outage_position(graph)
        for graph in train_graphs
    }

    val_positions = {
        graph_outage_position(graph)
        for graph in val_graphs
    }

    if heldout_position in train_positions:
        raise AssertionError(
            "Held-out line outage leaked into training."
        )

    if val_positions != {heldout_position}:
        raise AssertionError(
            "Validation contains unexpected topologies: "
            f"{sorted(val_positions)}"
        )

    if not val_graphs:
        raise RuntimeError(
            "Validation split is empty."
        )

    raw_path = Path(args.raw_out)
    train_path = Path(args.train_out)
    val_path = Path(args.val_out)
    manifest_path = Path(
        args.manifest_out
    )

    for path in (
        raw_path,
        train_path,
        val_path,
        manifest_path,
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    raw_payload = {
        "case": "case30",
        "settings": {
            "base_samples": args.base_samples,
            "samples_per_line": (
                args.samples_per_line
            ),
            "workers": args.workers,
            "seed": args.seed,
            "load_scale_range": [
                0.88,
                1.12,
            ],
            "reactive_scale_range": [
                0.88,
                1.12,
            ],
        },
        "line_records": line_records,
        "results": [
            {
                key: value
                for key, value
                in asdict(item).items()
                if key != "graph"
            }
            for item in results
        ],
    }

    with raw_path.open("wb") as f:
        pickle.dump(
            raw_payload,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    common = {
        "case": "case30",
        "source": str(raw_path),
        "split_type": (
            "held_out_single_line_outage"
        ),
        "heldout_line_position": (
            heldout_position
        ),
        "heldout_line_index": (
            heldout_record["line_index"]
        ),
        "heldout_from_bus": (
            heldout_record["from_bus"]
        ),
        "heldout_to_bus": (
            heldout_record["to_bus"]
        ),
    }

    torch.save(
        {
            "case": "case30",
            "graphs": train_graphs,
            "raw_info": {
                **common,
                "partition": "train",
                "scenario_ids": (
                    train_scenario_ids
                ),
            },
        },
        train_path,
    )

    torch.save(
        {
            "case": "case30",
            "graphs": val_graphs,
            "raw_info": {
                **common,
                "partition": "validation",
                "scenario_ids": (
                    val_scenario_ids
                ),
            },
        },
        val_path,
    )

    failure_types = Counter(
        item.error_type or "Unknown"
        for item in failed
    )

    manifest = {
        **common,
        "attempted_scenarios": (
            len(results)
        ),
        "successful_scenarios": (
            len(successful)
        ),
        "failed_scenarios": len(failed),
        "base_successful_graphs": (
            success_by_outage.get(-1, 0)
        ),
        "train_graphs": len(train_graphs),
        "validation_graphs": (
            len(val_graphs)
        ),
        "heldout_successful_graphs": (
            heldout_count
        ),
        "eligible_line_positions": (
            eligible_positions
        ),
        "line_records": line_records,
        "successful_graphs_by_outage": {
            str(position): int(count)
            for position, count
            in sorted(
                success_by_outage.items()
            )
        },
        "failure_types": dict(
            failure_types
        ),
        "train_topology_positions": (
            sorted(train_positions)
        ),
        "validation_topology_positions": (
            sorted(val_positions)
        ),
        "leakage_check_passed": True,
        "train_file": str(train_path),
        "validation_file": str(val_path),
        "raw_file": str(raw_path),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "=== Topology dataset complete ==="
    )

    print(
        "attempted:",
        len(results),
    )

    print(
        "successful:",
        len(successful),
    )

    print(
        "failed:",
        len(failed),
    )

    print(
        "held-out line position:",
        heldout_position,
    )

    print(
        "held-out line index:",
        heldout_record["line_index"],
    )

    print(
        "held-out buses:",
        f'{heldout_record["from_bus"]}'
        f'->{heldout_record["to_bus"]}',
    )

    print(
        "train graphs:",
        len(train_graphs),
    )

    print(
        "validation graphs:",
        len(val_graphs),
    )

    print(
        "train topology positions:",
        sorted(train_positions),
    )

    print(
        "validation topology positions:",
        sorted(val_positions),
    )

    print(
        "leakage check: passed"
    )

    print("raw:", raw_path)
    print("train:", train_path)
    print("validation:", val_path)
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
