import pandapower.networks as pn

from minigridsfm30.graph_builder import _line_pu_params
from minigridsfm30.cycle_basis import build_cycle_basis_from_lines


def main():
    net = pn.case30()
    sn_mva = float(net.sn_mva)

    bus_indices = list(net.bus.index.astype(int))
    bus_to_pos = {b: i for i, b in enumerate(bus_indices)}

    endpoints = []
    branch_x = []

    for _, row in net.line.iterrows():
        if not bool(row.get("in_service", True)):
            continue

        fb = int(row["from_bus"])
        tb = int(row["to_bus"])

        fpos = bus_to_pos[fb]
        tpos = bus_to_pos[tb]

        r_pu, x_pu, b_fr, b_to, rate = _line_pu_params(net, row, fb, sn_mva)

        endpoints.append((fpos, tpos))
        branch_x.append([
            -3.1415926,
            3.1415926,
            b_fr,
            b_to,
            r_pu,
            x_pu,
            rate,
        ])

    out = build_cycle_basis_from_lines(
        n_bus=len(bus_indices),
        line_endpoints=endpoints,
        branch_x=branch_x,
    )

    cycle_x, c2b_ei, c2b_ea, b2c_ei, b2c_ea = out

    print("n_bus:", len(bus_indices))
    print("n_branch:", len(branch_x))
    print("cycle_x:", cycle_x.shape)
    print("cycle -> branch edge_index:", c2b_ei.shape)
    print("cycle -> branch edge_attr:", c2b_ea.shape)
    print("branch -> cycle edge_index:", b2c_ei.shape)
    print("branch -> cycle edge_attr:", b2c_ea.shape)

    print()
    print("first 5 cycle.x = [length, sum_r, sum_x, mean_rate]:")
    print(cycle_x[:5])

    print()
    print("first 20 cycle->branch edges:")
    print(c2b_ei[:, :20])
    print(c2b_ea[:20])


if __name__ == "__main__":
    main()
