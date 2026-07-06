from minigridsfm30.graph_builder import build_solved_case30_heterodata


def main():
    data = build_solved_case30_heterodata(verbose=False)

    print(data)
    print()
    print("node types:", data.node_types)
    print("edge types:")
    for et in data.edge_types:
        print(" ", et)

    print()
    print("node shapes:")
    for nt in data.node_types:
        x = data[nt].x
        print(f"  {nt}.x:", tuple(x.shape))
        if hasattr(data[nt], "y"):
            print(f"  {nt}.y:", tuple(data[nt].y.shape))

    print()
    print("edge shapes:")
    for et in data.edge_types:
        ei = data[et].edge_index
        print(f"  {et}.edge_index:", tuple(ei.shape))
        if hasattr(data[et], "edge_attr"):
            print(f"  {et}.edge_attr:", tuple(data[et].edge_attr.shape))

    print()
    print("first 5 bus.x:")
    print(data["bus"].x[:5])

    print()
    print("first 5 bus.y = [theta_rad, vm_pu]:")
    print(data["bus"].y[:5])

    print()
    print("generator.x shape:", data["generator"].x.shape)
    print("generator.y = [Pg_pu, Qg_pu]:")
    print(data["generator"].y)

    print()
    print("branch_ac.x first 5 = [angmin, angmax, b_fr, b_to, r, x, rate]:")
    print(data["branch_ac"].x[:5])

    print()
    print("branch_ac.y first 5 = [p_from, q_from, p_to, q_to]:")
    print(data["branch_ac"].y[:5])

    print()
    print("feasible:", data.feasible)
    print("res_cost:", data.res_cost)
    print("sn_mva:", data.sn_mva)


if __name__ == "__main__":
    main()
