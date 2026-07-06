import torch

from minigridsfm30.graph_builder import build_solved_case30_heterodata
from minigridsfm30.model import GridSFM30
from minigridsfm30.losses import compute_loss


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data = build_solved_case30_heterodata(verbose=False).to(device)

    model = GridSFM30(
        hidden_dim=128,
        num_layers=3,
        dropout=0.0,
    ).to(device)

    model.train()

    out = model(data)
    loss, metrics = compute_loss(out, data)

    print("device:", device)
    print("loss:", float(loss.detach().cpu()))
    print()

    for k in sorted(metrics.keys()):
        print(f"{k}: {metrics[k]:.8f}")


if __name__ == "__main__":
    main()
