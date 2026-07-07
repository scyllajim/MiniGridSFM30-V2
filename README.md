# MiniGridSFM30-V2

MiniGridSFM30-V2 is a small-scale **GridSFM-style AC-OPF surrogate learning project** built on `pandapower` case30.

The goal is not to directly run the official Microsoft GridSFM-Open checkpoint on case30. Instead, this project reproduces the main ideas of GridSFM on a 30-bus system:

- AC-OPF scenario generation
- GridSFM-style heterogeneous graph construction
- PyTorch Geometric `HeteroData` dataset
- Heterogeneous GNN surrogate model
- Multi-target OPF prediction
- Physics-aware composite loss

---

## 1. Relation to Microsoft GridSFM

Microsoft GridSFM is an open-source framework for AC Optimal Power Flow (AC-OPF). Its official repository has two major parts:

- `power_grid/`: data pipeline that turns grid topologies into solved AC-OPF `.pyg.json` scenarios
- `model/`: neural surrogate model package that loads GridSFM-Open and runs fast AC-OPF inference

The official GridSFM data pipeline is roughly:

```text
raw grid topology
    -> cold-strict solvable PowerModels JSON
    -> perturbed solved .pyg.json scenarios
    -> GridSFM model training / inference
```

MiniGridSFM30-V2 follows the same idea in a smaller setting:

```text
pandapower case30
    -> perturbed pandapower OPF samples
    -> PyTorch Geometric HeteroData graphs
    -> MiniGridSFM30 heterogeneous GNN
```

Important difference:

The official GridSFM-Open checkpoint is trained on grids with at least 500 buses. Small cases such as case30 are out of distribution for the released checkpoint. Therefore, this project trains its own case30-specific MiniGridSFM model instead of using the official checkpoint directly.

---

## 2. Current status

Implemented:

- Project environment initialized
- GridSFM-style schema created
- `pandapower` case30 converted to PyG `HeteroData`
- Line is represented as `branch_ac` node
- `bus <-> branch_ac` endpoint graph is constructed
- `generator`, `load`, `shunt`, `branch_tr`, `cycle` node types are reserved
- AC-OPF labels are extracted from `pandapower.runopp`
- MiniGridSFM30 heterogeneous GNN implemented
- Composite loss implemented:
  - supervised bus state loss
  - generator output loss
  - branch flow loss
  - graph-level power balance loss
  - bus-level KCL loss
  - generation cost loss
  - feasibility loss

Current graph shape for case30:

```text
bus:        30 nodes
generator:   6 nodes = 5 regular generators + 1 ext_grid
load:       20 nodes
branch_ac:  41 nodes
endpoint edges: 82
```

---

## 3. Project structure

```text
MiniGridSFM30-V2/
├── README.md
├── activate_project.sh
├── minigridsfm30/
│   ├── schema.py          # GridSFM-style node/edge/feature schema
│   ├── graph_builder.py   # pandapower net -> HeteroData
│   ├── cycle_basis.py     # cycle node construction
│   ├── dataset.py         # Case30OPFDataset
│   ├── model.py           # MiniGridSFM30 heterogeneous GNN
│   └── losses.py          # GridSFM-style composite loss
├── scripts/
│   ├── sample_case30.py       # generate perturbed OPF samples
│   ├── preprocess_dataset.py  # raw pkl -> processed HeteroData .pt
│   ├── train.py               # train MiniGridSFM30
│   ├── eval.py                # evaluate checkpoint
│   ├── inspect_dataset.py
│   ├── test_dataset.py
│   ├── test_graph_builder.py
│   ├── test_model_forward.py
│   └── test_loss.py
├── data/
│   ├── raw/
│   └── processed/
└── runs/
```

---

## 4. Pipeline

### Stage 1: Base topology

The project uses `pandapower.networks.case30()` or `pandapower.networks.case_ieee30()` as the raw topology source.

This corresponds to the raw-topology stage in Microsoft GridSFM, but in this project the topology comes from a standard 30-bus test case instead of OSM or utility data.

### Stage 2: AC-OPF solving

Each sample is solved using:

```python
pandapower.runopp(net)
```

The OPF solution provides labels:

```text
bus.y        = [theta_rad, vm_pu]
generator.y  = [pg_pu, qg_pu]
branch_ac.y  = [p_from_pu, q_from_pu, p_to_pu, q_to_pu]
res_cost     = OPF objective
feasible     = OPF success flag
```

### Stage 3: Scenario generation

Current perturbations:

- load perturbation: randomly scale load `p_mw` and `q_mvar`
- cost perturbation: randomly scale polynomial cost coefficients
- optional line derating: randomly scale line `max_i_ka`

Not yet fully implemented:

- generator outage / `killgen`
- voltage bound squeezing / `vsqueeze`
- official-style pure-mode split

### Stage 4: Graph conversion

Solved pandapower samples are converted into PyTorch Geometric `HeteroData`.

Main node types:

```text
bus
generator
load
branch_ac
cycle
```

Reserved node types:

```text
shunt
branch_tr
```

Main edge types:

```text
bus <-> generator
bus <-> load
bus <-> branch_ac
cycle <-> branch_ac
```

### Stage 5: Model training

The `GridSFM30` model is a 3-layer heterogeneous GNN using:

- node-type-specific encoders
- `HeteroConv`
- `SAGEConv`
- node-type-specific prediction heads

Model outputs:

```text
bus_pred        = [theta_rad, vm_pu]
generator_pred  = [pg_pu, qg_pu]
branch_ac_pred  = [p_from_pu, q_from_pu, p_to_pu, q_to_pu]
feas_logit      = graph-level feasibility logit
```

---

## 5. Installation

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies manually or from your environment file.

Core dependencies:

```text
torch
torch-geometric
pandapower
numpy
pandas
tqdm
```

If needed:

```bash
pip install torch torch-geometric pandapower numpy pandas tqdm
```

---

## 6. Generate raw OPF samples

Generate 1000 perturbed case30 OPF samples:

```bash
python scripts/sample_case30.py \
  --case case30 \
  --n 1000 \
  --out data/raw/case30_1000_v2.pkl \
  --seed 42 \
  --load-min 0.95 \
  --load-max 1.05 \
  --cost-min 0.98 \
  --cost-max 1.02
```

Optional line derating can be enabled:

```bash
python scripts/sample_case30.py \
  --case case30 \
  --n 1000 \
  --out data/raw/case30_1000_v2_derate.pkl \
  --derate-prob 0.1 \
  --derate-min 0.8 \
  --derate-max 1.0
```

---

## 7. Preprocess dataset

Convert raw samples into PyG `HeteroData` graphs:

```bash
python scripts/preprocess_dataset.py \
  --raw data/raw/case30_1000_v2.pkl \
  --out data/processed/case30_1000_v2_graphs.pt \
  --only-feasible
```

---

## 8. Train model

Train MiniGridSFM30:

```bash
python scripts/train.py \
  --data data/processed/case30_1000_v2_graphs.pt \
  --run-dir runs/v2_baseline \
  --epochs 100 \
  --batch-size 16 \
  --hidden-dim 128 \
  --num-layers 3 \
  --lr 1e-3
```

Training outputs:

```text
runs/v2_baseline/best_model.pt
runs/v2_baseline/metrics.csv
```

---

## 9. Evaluate model

Evaluate a trained checkpoint:

```bash
python scripts/eval.py \
  --data data/processed/case30_1000_v2_graphs.pt \
  --ckpt runs/v2_baseline/best_model.pt \
  --batch-size 32
```

Evaluation metrics include:

```text
theta_mae
v_mae
pg_mae
qg_mae
branch_p_mae
branch_q_mae
kcl_p_mae
kcl_q_mae
balance_p_mae
balance_q_mae
cost_mape
```

The evaluation script also reports MW / MVAr converted values using baseMVA = 100.

---

## 10. Comparison with Microsoft GridSFM

| Component | Microsoft GridSFM | MiniGridSFM30-V2 |
|---|---|---|
| Target scale | large grids, >=500 buses | pandapower case30, 30 buses |
| Raw data | OSM / utility / open grid data | pandapower case30 |
| OPF solver | PowerModels.jl / Julia | pandapower.runopp / Python |
| Solvable format | `.solvable.json` | solved pandapower sample dict |
| Scenario format | `.pyg.json` | `.pkl` raw samples + `.pt` HeteroData |
| Perturbations | loads, costs, killgen, derate, vsqueeze | loads, costs, optional derate |
| Model | released GridSFM-Open checkpoint | self-trained GridSFM30 |
| Graph format | GridSFM HeteroData after preprocessing | PyG HeteroData |
| Outputs | V, theta, Pg, Qg, flows, feasibility | theta/V, Pg/Qg, branch flows, feasibility |

---

## 11. Current limitations

This project is still a small-scale experimental implementation.

Current limitations:

1. It does not implement the full Microsoft `topology_solver_pipeline`.
2. It does not implement L0/AC1/L1-L5 parameter relaxation, because case30 is already a standard solvable test system.
3. It does not export official `.pyg.json`; it uses PyG `HeteroData` directly.
4. Current perturbations do not yet fully match the official five-mode scenario generator.
5. `killgen` and `vsqueeze` are not yet implemented.
6. The feasibility classifier is currently limited because most saved samples are feasible OPF samples.
7. The current model does not yet include official GridSFM-style Hodge PE or DC prior.
8. The project currently focuses on case30 and does not yet test larger cases such as case118, case300, or case500.

---

## 12. Roadmap

Planned improvements:

- Add official-style pure perturbation modes:
  - `base`
  - `loads`
  - `costs`
  - `derate`
  - `killgen`
  - `vsqueeze`
- Save `perturb_mode` and `perturb_params` in each sample
- Add infeasible samples for meaningful feasibility classification
- Add sample re-solve verification script
- Add MLP baseline and mean baseline
- Add AC branch physics residual loss
- Add Laplacian positional encoding or DC power-flow prior
- Extend from case30 to case118 / case300 / case500
- Optionally add `.pyg.json` export compatibility

---

## 13. Summary

MiniGridSFM30-V2 is not a full reproduction of Microsoft GridSFM. It is a compact case30 implementation that reproduces the main learning pipeline idea:

```text
standard power grid case
    -> perturbed AC-OPF samples
    -> heterogeneous graph data
    -> surrogate GNN model
    -> OPF state / flow / feasibility prediction
```

The project is intended as a small, understandable, and extensible research prototype for learning GridSFM-style AC-OPF surrogate modeling.
