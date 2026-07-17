from __future__ import annotations

import torch

from minigridsfm30.graph_builder import (
    build_case30_net,
    net_to_heterodata,
)


def test_offline_line_uses_compact_branch_indices() -> None:
    net = build_case30_net()

    original_line_count = len(net.line)

    # Disable a line near the beginning so that an implementation based on
    # enumerate(...), followed by continue, would leave an index gap.
    disabled_line_index = int(net.line.index[0])
    net.line.loc[disabled_line_index, "in_service"] = False

    graph = net_to_heterodata(
        net,
        require_solution=False,
    )

    num_branches = int(graph["branch_ac"].x.shape[0])

    assert num_branches == original_line_count - 1

    bus_to_branch = graph[
        ("bus", "endpoint_of", "branch_ac")
    ].edge_index

    branch_to_bus = graph[
        ("branch_ac", "endpoint_of", "bus")
    ].edge_index

    assert bus_to_branch.shape[1] == 2 * num_branches
    assert branch_to_bus.shape[1] == 2 * num_branches

    assert int(bus_to_branch[1].min()) == 0
    assert int(bus_to_branch[1].max()) == num_branches - 1

    assert int(branch_to_bus[0].min()) == 0
    assert int(branch_to_bus[0].max()) == num_branches - 1

    expected = torch.arange(
        num_branches,
        dtype=torch.long,
    ).repeat_interleave(2)

    assert torch.equal(
        bus_to_branch[1],
        expected,
    )

    assert torch.equal(
        branch_to_bus[0],
        expected,
    )


def test_multiple_offline_lines_keep_valid_endpoint_edges() -> None:
    net = build_case30_net()

    disabled_indices = [
        int(net.line.index[0]),
        int(net.line.index[3]),
        int(net.line.index[7]),
    ]

    net.line.loc[disabled_indices, "in_service"] = False

    graph = net_to_heterodata(
        net,
        require_solution=False,
    )

    num_branches = int(graph["branch_ac"].x.shape[0])

    assert num_branches == len(net.line) - len(disabled_indices)

    for edge_type, branch_axis in (
        (
            ("bus", "endpoint_of", "branch_ac"),
            1,
        ),
        (
            ("branch_ac", "endpoint_of", "bus"),
            0,
        ),
    ):
        edge_index = graph[edge_type].edge_index
        branch_indices = edge_index[branch_axis]

        assert torch.isfinite(
            branch_indices.to(torch.float32)
        ).all()

        assert int(branch_indices.min()) >= 0
        assert int(branch_indices.max()) < num_branches
