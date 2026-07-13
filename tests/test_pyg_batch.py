import pandapower.networks as pn
from torch_geometric.loader import DataLoader

from minigridsfm30.graph_builder import run_ac_opf, net_to_heterodata


def make_graph(killed_gen=None):
    net = pn.case30()

    if killed_gen is not None:
        net.gen.loc[killed_gen, "in_service"] = False

    run_ac_opf(net)
    return net_to_heterodata(net, require_solution=True)


def test_graphs_can_be_batched():
    graphs = [
        make_graph(None),
        make_graph(0),
        make_graph(3),
    ]

    for graph in graphs:
        for key in graph.keys():
            assert not isinstance(graph[key], dict)

    loader = DataLoader(graphs, batch_size=3, shuffle=False)
    batch = next(iter(loader))

    assert int(batch.num_graphs) == 3
    assert batch["bus"].x.shape[0] == 90
    assert batch["generator"].x.shape[0] == 18
