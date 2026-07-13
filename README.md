# MiniGridSFM30-V2

A compact, reproducible **GridSFM-style heterogeneous GNN surrogate for AC Optimal Power Flow on `pandapower` case30**.

MiniGridSFM30-V2 studies whether a heterogeneous graph neural network can approximate AC-OPF solutions under:

- load variations;
- generation-cost variations;
- line derating;
- generator outage;
- voltage-bound squeezing;
- combined outage and operating-condition perturbations;
- random, mode-holdout, and few-shot evaluation settings.

> This project is not an official Microsoft GridSFM checkpoint. It is a small-scale research prototype for understanding, testing, and extending GridSFM-style AC-OPF surrogate learning.

---

## 1. Project overview

Traditional AC-OPF repeatedly solves a constrained nonlinear optimization problem:

```text
grid topology + operating condition
                    |
                    v
             AC-OPF solver
                    |
                    v
       voltage, generation, flows, cost
```

MiniGridSFM30-V2 learns a surrogate mapping:

```text
pandapower case30 scenario
          |
          v
PyG heterogeneous graph
          |
          v
GridSFM30 heterogeneous GNN
          |
          +--> bus voltage angle and magnitude
          +--> generator active/reactive power
          +--> branch active/reactive power flow
          +--> feasibility score
```

The surrogate is designed for:

- fast scenario screening;
- contingency pre-evaluation;
- large-batch OPF approximation;
- warm-start research;
- studying graph-based power-system representations.

It should not replace a trusted AC-OPF solver for final operational decisions.

---

## 2. Relation to Microsoft GridSFM

MiniGridSFM30-V2 follows the same high-level idea:

```text
grid scenario
    -> solved AC-OPF sample
    -> heterogeneous graph
    -> graph neural surrogate
    -> OPF state and flow prediction
```

Important differences:

| Item | Microsoft GridSFM | MiniGridSFM30-V2 |
|---|---|---|
| Main scale | Large grids | `pandapower` case30 |
| Solver | PowerModels / Julia pipeline | `pandapower.runopp` |
| Dataset format | GridSFM scenario format | raw `.pkl` + processed PyG `.pt` |
| Model | GridSFM-Open family | self-trained `GridSFM30` |
| Intended use | large-grid foundation-style inference | compact research and teaching prototype |
| Topology | multiple large systems | mainly fixed case30 topology |

The released large-grid checkpoint is not treated as a case30 model. MiniGridSFM30-V2 trains its own case30-specific network.

---

## 3. Implemented capabilities

### Data generation

Implemented perturbation modes:

- `base`
- `loads`
- `costs`
- `derate`
- `killgen`
- `vsqueeze`
- `mixed`
- `killgen_mixed`

Each raw sample can contain:

```text
sample_id
feasible
error_msg
perturb_mode
perturb_params
sn_mva
f_hz
res_cost
net_tables
res_tables
```

### Data integrity

Implemented utilities include:

- raw dataset analysis;
- perturbation failure analysis;
- dataset merging;
- re-solving saved scenarios;
- label agreement and cost consistency checks;
- feasible-only preprocessing;
- explicit train/validation split generation.

### Model and evaluation

Implemented:

- heterogeneous PyG graph construction;
- heterogeneous GraphSAGE model;
- multi-target supervised learning;
- physics-aware composite loss;
- random split evaluation;
- mode holdout evaluation;
- few-shot mode evaluation;
- mean baseline;
- nearest-neighbor baseline;
- outage-aware generator features;
- Stage11 combined-outage experiments.

---

## 4. Graph representation

The `pandapower` network is converted into a PyTorch Geometric `HeteroData` graph.

### Node types

```text
bus
generator
load
branch_ac
cycle
```

Reserved or partially represented types:

```text
shunt
branch_tr
```

Typical case30 graph size:

```text
bus:         30
generator:    6
load:        20
branch_ac:   41
```

The six generator nodes are:

```text
5 regular generators + 1 ext_grid
```

### Edge types

```text
bus <-> generator
bus <-> load
bus <-> branch_ac
cycle <-> branch_ac
```

A transmission line is represented as a `branch_ac` node rather than only as an edge:

```text
bus_from --> branch_ac <-- bus_to
```

This allows each branch to own:

- electrical parameters;
- operating limits;
- endpoint-direction information;
- four branch-flow targets.

---

## 5. Feature schema

### Bus features

The bus encoder expects 7 features:

```text
0  base_kv / 100
1  bus_type
2  min_vm_pu
3  max_vm_pu
4  aggregated active load in p.u.
5  aggregated reactive load in p.u.
6  is_slack
```

Bus labels:

```text
[theta_rad, vm_pu]
```

### Generator features

The generator encoder expects 12 features:

```text
0   is_online
1   reserved
2   effective_pmin_pu
3   effective_pmax_pu
4   reserved
5   effective_qmin_pu
6   effective_qmax_pu
7   vm_set_pu
8   cp2
9   cp1
10  cp0
11  is_ext_grid
```

Generator labels:

```text
[pg_pu, qg_pu]
```

### Outage-aware representation

Offline regular generators are not removed from the graph. Instead:

```text
is_online = 0
effective_pmin = 0
effective_pmax = 0
effective_qmin = 0
effective_qmax = 0
pg label = 0
qg label = 0
```

This preserves a fixed generator-node layout while making the outage explicit. Only online generators contribute to PV-bus classification.

### Load features

```text
[p_pu, q_pu]
```

### Branch features

The branch encoder expects 7 features, including electrical parameters and limits.

Branch labels:

```text
[p_from_pu, q_from_pu, p_to_pu, q_to_pu]
```

### Cycle features

Cycle nodes provide a compact structural description of network cycles and connect to the corresponding `branch_ac` nodes.

---

## 6. Model architecture

The default model is `GridSFM30`.

### Input encoders

```text
bus:         7  -> hidden_dim
generator:  12  -> hidden_dim
load:        2  -> hidden_dim
branch_ac:   7  -> hidden_dim
cycle:       4  -> hidden_dim
```

Each node type uses:

```text
Linear encoder
+ LayerNorm
```

### Message passing

The default model uses 3 heterogeneous GraphSAGE layers.

Relations:

```text
bus -> generator
generator -> bus

bus -> load
load -> bus

bus -> branch_ac
branch_ac -> bus

cycle -> branch_ac
branch_ac -> cycle
```

Each layer uses residual updates:

```text
h_new = LayerNorm(h_old + ReLU(message))
```

### Output heads

```text
bus_head:
    hidden -> [theta_rad, vm_pu]

generator_head:
    hidden -> [pg_pu, qg_pu]

branch_ac_head:
    hidden -> [p_from, q_from, p_to, q_to]

feasibility_head:
    pooled bus embedding -> feasibility logit
```

Default configuration:

```text
hidden_dim: 128
num_layers: 3
parameters: 866,569
```

---

## 7. Composite training loss

The objective supports configurable terms for:

- bus angle;
- bus voltage;
- generator active power;
- generator reactive power;
- branch active flow;
- branch reactive flow;
- system active-power balance;
- system reactive-power balance;
- bus active-power KCL;
- bus reactive-power KCL;
- generation cost;
- feasibility classification.

CLI parameters:

```text
--lambda-theta
--lambda-v
--lambda-pg
--lambda-qg
--lambda-branch-p
--lambda-branch-q
--lambda-balance-p
--lambda-balance-q
--lambda-kcl-p
--lambda-kcl-q
--lambda-cost
--lambda-feas
```

For feasible-only datasets, use:

```bash
--lambda-feas 0
```

because the feasibility head receives almost no meaningful negative supervision.

---

## 8. Repository structure

```text
MiniGridSFM30-V2/
├── README.md
├── activate_project.sh
├── minigridsfm30/
│   ├── schema.py
│   ├── cycle_basis.py
│   ├── graph_builder.py
│   ├── dataset.py
│   ├── model.py
│   └── losses.py
├── scripts/
│   ├── sample_case30.py
│   ├── sample_case30_killgen_mixed.py
│   ├── analyze_raw_dataset.py
│   ├── analyze_perturb_failures.py
│   ├── merge_raw_datasets.py
│   ├── preprocess_dataset.py
│   ├── resolve_check_samples.py
│   ├── train.py
│   ├── eval.py
│   ├── train_split.py
│   ├── eval_split.py
│   ├── baseline_compare.py
│   ├── baseline_split.py
│   ├── make_mode_split_dataset.py
│   └── make_fewshot_mode_split_dataset.py
├── data/
│   ├── raw/
│   └── processed/
├── reports/
└── runs/
```

Large raw datasets, processed datasets, and checkpoints are intentionally excluded from normal Git history.

---

## 9. Environment

Main tested environment:

```text
Python: 3.12
PyTorch: 2.6.0+cu124
torchvision: 0.21.0+cu124
torchaudio: 2.6.0+cu124
CUDA runtime: 12.4
GPU: NVIDIA RTX 4090
```

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install PyTorch CUDA 12.4 wheels:

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124
```

Install the remaining dependencies:

```bash
pip install torch-geometric pandapower numpy pandas scipy \
  networkx tqdm scikit-learn
```

Record exact versions:

```bash
pip freeze > requirements-lock.txt
```

---

## 10. Pure-mode dataset generation

### Base samples

```bash
python scripts/sample_case30.py \
  --mode base \
  --n 50 \
  --out data/raw/case30_base_0050.pkl \
  --seed 9101 \
  --keep-failed
```

### Load perturbations

```bash
python scripts/sample_case30.py \
  --mode loads \
  --n 800 \
  --out data/raw/case30_loads_0800.pkl \
  --seed 9102 \
  --load-min 0.95 \
  --load-max 1.05 \
  --load-jitter 0.00 \
  --keep-failed
```

### Cost perturbations

```bash
python scripts/sample_case30.py \
  --mode costs \
  --n 500 \
  --out data/raw/case30_costs_0500.pkl \
  --seed 9103 \
  --cost-min 0.8 \
  --cost-max 1.2 \
  --keep-failed
```

### Line derating

```bash
python scripts/sample_case30.py \
  --mode derate \
  --n 600 \
  --out data/raw/case30_derate_0600.pkl \
  --seed 9104 \
  --derate-prob 0.1 \
  --derate-min 0.8 \
  --derate-max 0.98 \
  --keep-failed
```

### Generator outage

Current experiments use empirically stable generator candidates 0 and 3:

```bash
python scripts/sample_case30.py \
  --mode killgen \
  --n 400 \
  --out data/raw/case30_killgen_0400.pkl \
  --seed 9105 \
  --killgen-n 1 \
  --killgen-keep-min 1 \
  --killgen-candidates 0,3 \
  --keep-failed
```

### Voltage-limit squeezing

```bash
python scripts/sample_case30.py \
  --mode vsqueeze \
  --n 400 \
  --out data/raw/case30_vsqueeze_0400.pkl \
  --seed 9106 \
  --vsqueeze-prob 0.1 \
  --vsqueeze-eps 0.005 \
  --keep-failed
```

---

## 11. Stage11 outage-combination datasets

### Generator outage plus load variation

```bash
python scripts/sample_case30_killgen_mixed.py \
  --n 3000 \
  --out data/raw/stage11_killgen_loads_3000.pkl \
  --seed 12001 \
  --killgen-candidates 0,3 \
  --apply-loads \
  --load-min 0.95 \
  --load-max 1.05 \
  --load-jitter 0.00 \
  --keep-failed
```

Observed:

```text
requested: 3000
feasible:  2091
failed:     909
success:  69.70%
```

### Generator outage plus line derating

```bash
python scripts/sample_case30_killgen_mixed.py \
  --n 1000 \
  --out data/raw/stage11_killgen_derate_1000.pkl \
  --seed 12002 \
  --killgen-candidates 0,3 \
  --apply-derate \
  --derate-prob 0.05 \
  --derate-min 0.9 \
  --derate-max 0.99 \
  --keep-failed
```

Observed:

```text
requested: 1000
feasible:   964
failed:      36
success:  96.40%
```

---

## 12. Merge and preprocess datasets

```bash
python scripts/merge_raw_datasets.py \
  --out data/raw/case30_merged.pkl \
  --seed 9200 \
  --shuffle \
  data/raw/case30_base_0050.pkl \
  data/raw/case30_loads_0800.pkl \
  data/raw/case30_costs_0500.pkl
```

```bash
python scripts/preprocess_dataset.py \
  --raw data/raw/case30_merged.pkl \
  --out data/processed/case30_merged_feasible_graphs.pt \
  --only-feasible
```

Do not attach raw heterogeneous Python dictionaries such as `perturb_params` directly to each `HeteroData` graph. Different dictionary keys across samples are not batch-safe in PyG.

---

## 13. Standard random-split training

```bash
python scripts/train.py \
  --data data/processed/stage11_case30_outage_augmented_feasible_graphs.pt \
  --run-dir runs/stage11_outage_augmented_default \
  --epochs 100 \
  --batch-size 16 \
  --hidden-dim 128 \
  --num-layers 3 \
  --lr 1e-3 \
  --seed 42 \
  --lambda-feas 0
```

Evaluate:

```bash
python scripts/eval.py \
  --data data/processed/stage11_case30_outage_augmented_feasible_graphs.pt \
  --ckpt runs/stage11_outage_augmented_default/best_model.pt \
  --batch-size 32
```

---

## 14. Explicit split training

```bash
python scripts/train_split.py \
  --train-data data/processed/train_graphs.pt \
  --val-data data/processed/val_graphs.pt \
  --run-dir runs/explicit_split \
  --epochs 100 \
  --batch-size 16 \
  --hidden-dim 128 \
  --num-layers 3 \
  --lr 1e-3 \
  --seed 42
```

Evaluate:

```bash
python scripts/eval_split.py \
  --train-data data/processed/train_graphs.pt \
  --val-data data/processed/val_graphs.pt \
  --ckpt runs/explicit_split/best_model.pt \
  --batch-size 32
```

---

## 15. Baselines

### Random split

```bash
python scripts/baseline_compare.py \
  --data data/processed/case30_graphs.pt \
  --out-csv reports/baseline_random_split.csv \
  --seed 42
```

### Explicit split

```bash
python scripts/baseline_split.py \
  --train-data data/processed/train_graphs.pt \
  --val-data data/processed/val_graphs.pt \
  --out-csv reports/baseline_explicit_split.csv
```

Implemented baselines:

- mean prediction;
- nearest-neighbor retrieval in flattened input-feature space.

Nearest-neighbor is especially strong on fixed-topology case30 because nearby operating scenarios can have nearly identical OPF solutions.

---

## 16. Main experimental results

### 3000-sample pure-mode random split

| Model | Theta | V | Pg MW | Qg MVAr | Branch P MW | Branch Q MVAr | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean | 0.009568 | 0.002890 | 4.4546 | 2.3632 | 1.3004 | 0.6040 | 3.0837% |
| Nearest neighbor | 0.006526 | 0.002382 | 3.4606 | 1.9587 | 0.9283 | 0.4782 | 2.0664% |
| Default GNN | 0.008648 | 0.002659 | 4.0303 | 2.0841 | 1.2703 | 0.5707 | 3.6409% |

### Outage-augmented random split

Dataset size:

```text
5806 feasible graphs
```

| Metric | Result |
|---|---:|
| Pg MAE | 3.7835 MW |
| Qg MAE | 1.8851 MVAr |
| Branch P MAE | 1.4090 MW |
| Branch Q MAE | 0.6421 MVAr |
| KCL P MAE | 0.4021 MW |
| KCL Q MAE | 0.2631 MVAr |
| Balance P MAE | 1.1106 MW |
| Balance Q MAE | 0.4486 MVAr |
| Cost MAPE | 7.4443% |

### Combination holdout

Training data:

```text
pure modes
+
killgen + derate
```

Validation data:

```text
killgen + loads
```

| Model | Pg MW | Qg MVAr | Branch P MW | Branch Q MVAr | Cost |
|---|---:|---:|---:|---:|---:|
| Mean | 11.2375 | 5.8059 | 3.2076 | 1.2545 | 6.2034% |
| Nearest neighbor | 2.9949 | 1.5685 | 1.0476 | 0.5427 | 3.5289% |
| GNN | 4.3918 | 2.2933 | 1.7862 | 0.7721 | 4.8118% |

The GNN substantially outperforms the mean baseline and demonstrates partial combination generalization, but it does not outperform nearest-neighbor.

---

## 17. Interpretation

Main findings:

1. Increasing feasible training data improves GNN prediction accuracy.
2. Explicit generator availability greatly improves generator-outage prediction.
3. Random splits over fixed topology favor nearest-neighbor retrieval.
4. Mode holdout and combination holdout are more meaningful than random split alone.
5. Physics-loss reweighting changes metric trade-offs but has not consistently dominated the default objective.
6. The current GNN generalizes better than a mean predictor, but not consistently better than nearest-neighbor.
7. A stronger graph-learning advantage likely requires multiple topologies or unseen topology changes.

---

## 18. Known limitations

1. The project currently focuses on case30.
2. Most experiments use one fixed topology.
3. Nearest-neighbor is unusually strong under dense random sampling.
4. Only stable generator-outage candidates 0 and 3 are currently used.
5. `OPFNotConverged` is treated as infeasible, although solver failure and mathematical infeasibility are not always identical.
6. The feasibility head is not meaningful when training with feasible-only datasets.
7. The current model does not use branch edge attributes directly inside message passing.
8. The model does not yet include Hodge positional encodings or a DC power-flow prior.
9. Current results are mostly single-seed experiments.
10. No cross-case or cross-topology generalization has yet been demonstrated.

---

## 19. Recommended future work

### Engineering

- add locked dependency files;
- add YAML experiment configurations;
- add formal `pytest` tests;
- add automatic experiment summarization;
- add dataset checksums and manifests;
- add continuous integration.

### Evaluation

- run at least five random seeds;
- report mean and standard deviation;
- test unseen generator outages;
- test unseen load ranges;
- test line-outage topology changes;
- use case118 and case300;
- evaluate across multiple topologies.

### Modeling

- improve graph-level pooling;
- separate feasibility classification from feasible-only regression;
- add edge-attribute-aware message passing;
- add DC power-flow priors;
- add Laplacian or Hodge positional encoding;
- investigate physics projection after neural prediction.

---

## 20. Reproducibility checklist

Before reporting an experiment, record:

```text
git commit
dataset filename
dataset SHA256
sampling seed
split seed
training seed
model configuration
loss weights
best epoch
validation metrics
CUDA/PyTorch environment
```

Useful commands:

```bash
git rev-parse HEAD
sha256sum data/raw/*.pkl
sha256sum data/processed/*.pt
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi
```

---

## 21. Project status

```text
data pipeline                 implemented
pure perturbation modes       implemented
outage-aware graph features   implemented
heterogeneous GNN             implemented
physics-aware loss            implemented
mean/NN baselines             implemented
mode holdout                  implemented
few-shot split                implemented
combined outage datasets      implemented
multi-topology evaluation     planned
large-grid evaluation         planned
```

---

## 22. License and citation

A formal open-source license and `CITATION.cff` should be added before treating the repository as a reusable public research package.

---

## 23. Summary

MiniGridSFM30-V2 is a compact AC-OPF surrogate-learning research framework:

```text
pandapower case30
    -> perturb operating conditions
    -> solve AC-OPF
    -> construct heterogeneous graphs
    -> train a heterogeneous GNN
    -> evaluate state, dispatch, flow, and physics errors
```

Its main contribution is a reproducible small-grid testbed for studying:

- heterogeneous power-grid graph representations;
- physics-aware neural losses;
- generator-outage encoding;
- random versus holdout evaluation;
- nearest-neighbor versus graph-model behavior;
- compositional generalization under combined operating conditions.
