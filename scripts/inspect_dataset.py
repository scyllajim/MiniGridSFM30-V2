import argparse
import pickle

from minigridsfm30.graph_builder import sample_to_heterodata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/case30_10_v2.pkl")
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()

    with open(args.data, "rb") as f:
        d = pickle.load(f)

    print("=== dataset info ===")
    print("case:", d["case"])
    print("requested:", d["n_requested"])
    print("success:", d["n_success"])
    print("failed:", d["n_failed"])
    print("saved:", d["n_saved"])
    print("settings:", d["settings"])
    print("sample_format:", d["sample_format"])

    sample = d["samples"][args.idx]
    print()
    print("=== sample info ===")
    print("sample_id:", sample["sample_id"])
    print("feasible:", sample["feasible"])
    print("sn_mva:", sample["sn_mva"])
    print("res_cost:", sample["res_cost"])
    print("net tables:", list(sample["net_tables"].keys()))
    print("res tables:", list(sample["res_tables"].keys()))

    data = sample_to_heterodata(sample)

    print()
    print("=== heterodata ===")
    print(data)

    print()
    print("node shapes:")
    for nt in data.node_types:
        print(f"  {nt}.x:", tuple(data[nt].x.shape))
        if hasattr(data[nt], "y"):
            print(f"  {nt}.y:", tuple(data[nt].y.shape))

    print()
    print("edge shapes:")
    for et in data.edge_types:
        print(f"  {et}.edge_index:", tuple(data[et].edge_index.shape))
        if hasattr(data[et], "edge_attr"):
            print(f"  {et}.edge_attr:", tuple(data[et].edge_attr.shape))

    print()
    print("bus.y first 5 = [theta, V]:")
    print(data["bus"].y[:5])

    print()
    print("generator.y = [Pg, Qg]:")
    print(data["generator"].y)

    print()
    print("branch_ac.y first 5:")
    print(data["branch_ac"].y[:5])


if __name__ == "__main__":
    main()
