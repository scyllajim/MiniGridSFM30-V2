# Architecture

## 1. Overview

MiniGridSFM30-V2 is a heterogeneous graph neural network surrogate for AC optimal power flow on `pandapower` case30.

The end-to-end pipeline is:

~~~text
pandapower case30
    -> perturb operating conditions
    -> solve AC-OPF with pandapower.runopp
    -> save raw samples
    -> convert samples to PyG HeteroData
    -> train GridSFM30 heterogeneous GNN
    -> evaluate state, dispatch, flow, cost, and physics errors
~~~

## 2. Graph schema

### Node types

- `bus`
- `generator`
- `load`
- `branch_ac`
- `cycle`

Reserved or partially represented types:

- `shunt`
- `branch_tr`

### Edge types

- `bus -> generator`
- `generator -> bus`
- `bus -> load`
- `load -> bus`
- `bus -> branch_ac`
- `branch_ac -> bus`
- `cycle -> branch_ac`
- `branch_ac -> cycle`

Transmission lines are represented as `branch_ac` nodes rather than only graph edges.

~~~text
bus_from --> branch_ac <-- bus_to
~~~

This makes it possible to attach branch parameters, limits, and directional flow labels directly to branch nodes.

## 3. Feature dimensions

| Node type | Input dimension | Main contents |
|---|---:|---|
| bus | 7 | voltage limits, load aggregation, bus type, slack flag |
| generator | 12 | online state, P/Q limits, voltage setpoint, cost, ext-grid flag |
| load | 2 | active and reactive demand |
| branch_ac | 7 | electrical parameters and limits |
| cycle | 4 | structural cycle representation |

## 4. Outage-aware generator representation

Offline generators remain in the graph.

For an offline regular generator:

~~~text
is_online = 0
effective_pmin = 0
effective_pmax = 0
effective_qmin = 0
effective_qmax = 0
pg target = 0
qg target = 0
~~~

Only online generators contribute to PV-bus classification.

This keeps graph dimensions fixed and makes generator outage status explicit.

## 5. Model architecture

The default model is `GridSFM30`.

### Encoders

Each node type has an independent linear encoder followed by `LayerNorm`.

~~~text
bus:         7  -> hidden_dim
generator:  12  -> hidden_dim
load:        2  -> hidden_dim
branch_ac:   7  -> hidden_dim
cycle:       4  -> hidden_dim
~~~

### Message passing

The default model uses three heterogeneous GraphSAGE layers.

Each layer performs:

~~~text
message aggregation
    -> ReLU
    -> dropout
    -> residual addition
    -> LayerNorm
~~~

Conceptually:

~~~text
h_new = LayerNorm(h_old + ReLU(message))
~~~

### Prediction heads

- bus head: `[theta_rad, vm_pu]`
- generator head: `[pg_pu, qg_pu]`
- branch head: `[p_from_pu, q_from_pu, p_to_pu, q_to_pu]`
- feasibility head: graph-level feasibility logit

Default configuration:

~~~text
hidden_dim = 128
num_layers = 3
parameters = 866,569
~~~

## 6. Loss function

The composite objective can include:

- bus angle loss;
- voltage loss;
- generator active-power loss;
- generator reactive-power loss;
- branch active-flow loss;
- branch reactive-flow loss;
- bus-level active KCL;
- bus-level reactive KCL;
- graph-level active balance;
- graph-level reactive balance;
- generation-cost loss;
- feasibility loss.

All weights are configurable from the CLI.

## 7. Current architectural limitations

- fixed case30 topology dominates most experiments;
- branch edge attributes are not used directly inside message passing;
- graph-level pooling uses only bus embeddings;
- no Hodge or Laplacian positional encoding;
- no DC power-flow prior;
- no physics projection after inference;
- feasibility learning is weak on feasible-only datasets.
