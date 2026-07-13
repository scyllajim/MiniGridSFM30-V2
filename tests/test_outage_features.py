import pandapower.networks as pn

from minigridsfm30.graph_builder import run_ac_opf, net_to_heterodata


def test_offline_generator_is_encoded_explicitly():
    net = pn.case30()

    net.gen.loc[0, "in_service"] = False
    run_ac_opf(net)

    g = net_to_heterodata(net, require_solution=True)

    gen_x = g["generator"].x
    gen_y = g["generator"].y

    assert gen_x.shape[1] == 12
    assert float(gen_x[0, 0]) == 0.0

    # effective P/Q limits of offline generator
    assert float(gen_x[0, 2]) == 0.0
    assert float(gen_x[0, 3]) == 0.0
    assert float(gen_x[0, 5]) == 0.0
    assert float(gen_x[0, 6]) == 0.0

    # solved target is forced to zero
    assert float(gen_y[0, 0]) == 0.0
    assert float(gen_y[0, 1]) == 0.0
