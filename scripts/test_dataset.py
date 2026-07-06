import argparse

from torch_geometric.loader import DataLoader

from minigridsfm30.dataset import Case30OPFDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/case30_10_v2.pkl")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    ds = Case30OPFDataset(args.data, only_feasible=True)

    print("=== dataset info ===")
    for k, v in ds.info.items():
        print(k, ":", v)

    print()
    print("len:", len(ds))

    data0 = ds[0]
    print()
    print("=== single sample ===")
    print(data0)
    print("bus.x:", data0["bus"].x.shape)
    print("bus.y:", data0["bus"].y.shape)
    print("generator.x:", data0["generator"].x.shape)
    print("generator.y:", data0["generator"].y.shape)
    print("branch_ac.x:", data0["branch_ac"].x.shape)
    print("branch_ac.y:", data0["branch_ac"].y.shape)
    print("cycle.x:", data0["cycle"].x.shape)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    batch = next(iter(loader))

    print()
    print("=== batch sample ===")
    print(batch)
    print("batch.num_graphs:", batch.num_graphs)
    print("bus.x:", batch["bus"].x.shape)
    print("bus.y:", batch["bus"].y.shape)
    print("generator.x:", batch["generator"].x.shape)
    print("generator.y:", batch["generator"].y.shape)
    print("branch_ac.x:", batch["branch_ac"].x.shape)
    print("branch_ac.y:", batch["branch_ac"].y.shape)
    print("cycle.x:", batch["cycle"].x.shape)

    if hasattr(batch["bus"], "batch"):
        print("bus.batch:", batch["bus"].batch.shape)

    if hasattr(batch["generator"], "batch"):
        print("generator.batch:", batch["generator"].batch.shape)

    if hasattr(batch["branch_ac"], "batch"):
        print("branch_ac.batch:", batch["branch_ac"].batch.shape)


if __name__ == "__main__":
    main()
