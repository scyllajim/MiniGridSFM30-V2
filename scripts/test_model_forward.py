import torch

from minigridsfm30.graph_builder import build_solved_case30_heterodata
from minigridsfm30.model import GridSFM30, count_parameters


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data = build_solved_case30_heterodata(verbose=False).to(device)

    model = GridSFM30(
        hidden_dim=128,
        num_layers=3,
        dropout=0.0,
    ).to(device)

    model.eval()

    with torch.no_grad():
        out = model(data)

    print("device:", device)
    print("model:", model.__class__.__name__)
    print("parameters:", count_parameters(model))

    print()
    print("input shapes:")
    print("bus.x:", data["bus"].x.shape)
    print("generator.x:", data["generator"].x.shape)
    print("load.x:", data["load"].x.shape)
    print("branch_ac.x:", data["branch_ac"].x.shape)
    print("cycle.x:", data["cycle"].x.shape)

    print()
    print("target shapes:")
    print("bus.y:", data["bus"].y.shape, " = [theta_rad, vm_pu]")
    print("generator.y:", data["generator"].y.shape, " = [Pg_pu, Qg_pu]")
    print("branch_ac.y:", data["branch_ac"].y.shape, " = [p_from, q_from, p_to, q_to]")

    print()
    print("prediction shapes:")
    print("bus_pred:", out["bus_pred"].shape)
    print("generator_pred:", out["generator_pred"].shape)
    print("branch_ac_pred:", out["branch_ac_pred"].shape)
    print("feas_logit:", out["feas_logit"].shape)

    print()
    print("first 5 bus_pred = [theta_rad, vm_pu]:")
    print(out["bus_pred"][:5])

    print()
    print("first 5 bus target = [theta_rad, vm_pu]:")
    print(data["bus"].y[:5])

    print()
    print("generator_pred = [Pg_pu, Qg_pu]:")
    print(out["generator_pred"])

    print()
    print("generator target = [Pg_pu, Qg_pu]:")
    print(data["generator"].y)

    print()
    print("first 5 branch_ac_pred = [p_from, q_from, p_to, q_to]:")
    print(out["branch_ac_pred"][:5])

    print()
    print("first 5 branch_ac target = [p_from, q_from, p_to, q_to]:")
    print(data["branch_ac"].y[:5])

    print()
    print("feas_logit:")
    print(out["feas_logit"])


if __name__ == "__main__":
    main()
