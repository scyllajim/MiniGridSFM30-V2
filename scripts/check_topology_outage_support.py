from __future__ import annotations

from minigridsfm30.graph_builder import (
    build_case30_net,
    net_to_heterodata,
)


def main() -> None:
    base_net = build_case30_net()
    base_graph = net_to_heterodata(
        base_net,
        require_solution=False,
    )

    print(
        "base lines:",
        len(base_net.line),
    )
    print(
        "base branch_ac nodes:",
        base_graph["branch_ac"].x.shape[0],
    )

    candidate_positions = [
        0,
        1,
        2,
        3,
        5,
        7,
        10,
        15,
        20,
        25,
    ]

    print()
    print("=== Single-line outage graph checks ===")

    passed = 0

    for position in candidate_positions:
        if position >= len(base_net.line):
            continue

        net = build_case30_net()
        line_index = int(net.line.index[position])

        from_bus = int(
            net.line.loc[line_index, "from_bus"]
        )
        to_bus = int(
            net.line.loc[line_index, "to_bus"]
        )

        net.line.loc[
            line_index,
            "in_service",
        ] = False

        try:
            graph = net_to_heterodata(
                net,
                require_solution=False,
            )

            num_branches = int(
                graph["branch_ac"].x.shape[0]
            )

            bus_to_branch = graph[
                ("bus", "endpoint_of", "branch_ac")
            ].edge_index

            branch_to_bus = graph[
                ("branch_ac", "endpoint_of", "bus")
            ].edge_index

            max_forward = int(
                bus_to_branch[1].max()
            )
            max_reverse = int(
                branch_to_bus[0].max()
            )

            valid = (
                num_branches == len(net.line) - 1
                and max_forward < num_branches
                and max_reverse < num_branches
            )

            status = "PASS" if valid else "FAIL"

            print(
                f"{status}: "
                f"line_position={position}, "
                f"line_index={line_index}, "
                f"buses={from_bus}->{to_bus}, "
                f"branch_nodes={num_branches}, "
                f"max_edge_branch="
                f"{max(max_forward, max_reverse)}"
            )

            if valid:
                passed += 1

        except Exception as exc:
            print(
                "FAIL:",
                f"line_position={position},",
                f"line_index={line_index},",
                f"error={type(exc).__name__}: {exc}",
            )

    print()
    print(
        f"passed {passed} / "
        f"{len(candidate_positions)} checks"
    )

    if passed != len(candidate_positions):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
