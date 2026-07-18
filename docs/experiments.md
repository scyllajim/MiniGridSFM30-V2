# Experiments

## 1. Evaluation philosophy

MiniGridSFM30-V2 uses several evaluation settings:

- random split;
- mode holdout;
- explicit train/validation split;
- few-shot split;
- combined perturbation holdout.

Random split alone is not sufficient because all samples share the same case30 topology and nearby operating points can be extremely similar.

## 2. Main datasets

| Dataset | Requested | Feasible | Failed | Purpose |
|---|---:|---:|---:|---|
| `case30_pure_modes_3000.pkl` | 3000 | 2751 | 249 | pure perturbation modes |
| `stage11_killgen_loads_3000.pkl` | 3000 | 2091 | 909 | generator outage plus load variation |
| `stage11_killgen_derate_1000.pkl` | 1000 | 964 | 36 | generator outage plus weak line derating |
| outage-augmented processed set | — | 5806 | — | random-split outage-augmented training |

## 3. Pure-mode random split

Default GNN validation:

| Metric | Result |
|---|---:|
| Theta MAE | 0.008648 |
| V MAE | 0.002659 |
| Pg MAE | 4.0303 MW |
| Qg MAE | 2.0841 MVAr |
| Branch P MAE | 1.2703 MW |
| Branch Q MAE | 0.5707 MVAr |
| KCL P MAE | 0.2714 MW |
| KCL Q MAE | 0.2233 MVAr |
| Balance P MAE | 1.9261 MW |
| Balance Q MAE | 1.6635 MVAr |
| Cost MAPE | 3.6409% |

Baseline comparison:

| Model | Pg MW | Qg MVAr | Branch P MW | Branch Q MVAr | Cost |
|---|---:|---:|---:|---:|---:|
| Mean | 4.4546 | 2.3632 | 1.3004 | 0.6040 | 3.0837% |
| Nearest neighbor | 3.4606 | 1.9587 | 0.9283 | 0.4782 | 2.0664% |
| Default GNN | 4.0303 | 2.0841 | 1.2703 | 0.5707 | 3.6409% |

Nearest-neighbor is stronger on this dense fixed-topology random split.

## 4. Outage-aware few-shot experiment

Before outage-aware encoding, generator-outage generalization degraded heavily.

After adding explicit generator online state and zeroing effective offline limits:

~~~text
Pg MAE improved to approximately 5.83 MW
Qg MAE improved to approximately 2.28 MVAr
~~~

This demonstrates that generator availability must be explicitly represented.

## 5. Outage-augmented random split

Dataset size:

~~~text
5806 feasible graphs
~~~

Default GNN validation:

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

## 6. Combination holdout

Training set:

~~~text
pure modes
+
killgen + derate
~~~

Validation set:

~~~text
killgen + loads
~~~

Results:

| Model | Pg MW | Qg MVAr | Branch P MW | Branch Q MVAr | Cost |
|---|---:|---:|---:|---:|---:|
| Mean | 11.2375 | 5.8059 | 3.2076 | 1.2545 | 6.2034% |
| Nearest neighbor | 2.9949 | 1.5685 | 1.0476 | 0.5427 | 3.5289% |
| GNN | 4.3918 | 2.2933 | 1.7862 | 0.7721 | 4.8118% |

Interpretation:

- GNN substantially outperforms the mean baseline;
- GNN shows partial compositional generalization;
- nearest-neighbor remains stronger;
- fixed topology and dense scenario coverage strongly favor retrieval methods.

## 7. Physics-loss experiments

Increasing KCL or branch-loss weights changes metric trade-offs but has not produced a universally dominant configuration.

Observed pattern:

- stronger physics weights can improve KCL;
- dispatch or balance metrics may worsen;
- no consistent global winner has emerged.

The default loss remains the primary baseline.

## 8. Experimental cautions

- `OPFNotConverged` is solver failure, not always proof of mathematical infeasibility;
- pure killgen samples can collapse into repeated templates;
- nearest-neighbor zero error in such settings indicates template lookup, not broad generalization;
- cost MAPE should not be compared across datasets with different cost perturbation distributions;
- single-seed results should not be treated as statistically conclusive.

## 9. Recommended next experiments

1. five-seed repeated training;
2. mean and standard deviation reporting;
3. unseen-generator outage split;
4. unseen load-range split;
5. line-outage topology changes;
6. case118 and case300;
7. multi-topology training and validation.

<!-- STAGE13_START -->
## 10. Stage13 five-seed full-100 holdout experiment

To measure training variability, the combination-holdout experiment was repeated with five random seeds:

~~~text
42, 43, 44, 45, 46
~~~

All runs used:

- the same explicit training and validation datasets;
- 100 complete epochs;
- no early stopping;
- `hidden_dim = 128`;
- `num_layers = 3`;
- `batch_size = 16`;
- `lr = 1e-3`;
- identical loss weights;
- explicit CUDA device assignment;
- NaN/Inf checks and gradient clipping.

Training data:

~~~text
pure modes
+
killgen + derate
~~~

Validation data:

~~~text
killgen + loads
~~~

### 10.1 Aggregate GNN results

| Metric | Mean ± Std | Min | Max |
|---|---:|---:|---:|
| Theta MAE | 0.0223 ± 0.0002 | 0.0220 | 0.0225 |
| V MAE | 0.0063 ± 0.0010 p.u. | 0.0052 | 0.0076 |
| Validation loss | 0.0064 ± 0.0002 | 0.0062 | 0.0066 |
| Pg MAE | 4.4268 ± 0.1327 MW | 4.2095 | 4.5639 |
| Qg MAE | 2.3153 ± 0.0269 MVAr | 2.2834 | 2.3449 |
| Branch P MAE | 1.8489 ± 0.1413 MW | 1.7393 | 2.0896 |
| Branch Q MAE | 0.7661 ± 0.0539 MVAr | 0.7202 | 0.8586 |
| KCL P MAE | 0.6074 ± 0.1129 MW | 0.5019 | 0.7632 |
| KCL Q MAE | 0.3303 ± 0.0600 MVAr | 0.2691 | 0.4250 |
| Balance P MAE | 2.8444 ± 0.9664 MW | 2.1804 | 4.4731 |
| Balance Q MAE | 2.0175 ± 1.0098 MVAr | 1.2729 | 3.6722 |
| Cost MAPE | 4.9088 ± 0.3500% | 4.5807 | 5.4193 |

### 10.2 Per-seed results

| Seed | Best epoch | Pg MW | Qg MVAr | Branch P MW | Branch Q MVAr | KCL P MW | Balance P MW | Cost % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 89 | 4.4214 | 2.3037 | 1.7393 | 0.7202 | 0.5408 | 2.9919 | 4.5807 |
| 43 | 49 | 4.2095 | 2.3449 | 2.0896 | 0.8586 | 0.7632 | 2.3164 | 4.8885 |
| 44 | 97 | 4.4505 | 2.3027 | 1.7679 | 0.7623 | 0.5019 | 2.2602 | 4.5951 |
| 45 | 88 | 4.5639 | 2.2834 | 1.7913 | 0.7472 | 0.5411 | 4.4731 | 5.0604 |
| 46 | 94 | 4.4886 | 2.3419 | 1.8566 | 0.7424 | 0.6899 | 2.1804 | 5.4193 |

### 10.3 Comparison with baselines

| Model | Pg MW | Qg MVAr | Branch P MW | Branch Q MVAr | Cost MAPE |
|---|---:|---:|---:|---:|---:|
| Mean baseline | 11.2375 | 5.8059 | 3.2076 | 1.2545 | 6.2034% |
| Nearest neighbor | 2.9949 | 1.5685 | 1.0476 | 0.5427 | 3.5289% |
| GNN, five-seed mean | 4.4268 ± 0.1327 | 2.3153 ± 0.0269 | 1.8489 ± 0.1413 | 0.7661 ± 0.0539 | 4.9088 ± 0.3500% |

### 10.4 Relative improvement over the mean baseline

Using the five-seed GNN mean:

| Metric | Mean baseline | GNN mean | Relative reduction |
|---|---:|---:|---:|
| Pg MAE | 11.2375 MW | 4.4268 MW | 60.61% |
| Qg MAE | 5.8059 MVAr | 2.3153 MVAr | 60.12% |
| Branch P MAE | 3.2076 MW | 1.8489 MW | 42.36% |
| Branch Q MAE | 1.2545 MVAr | 0.7661 MVAr | 38.93% |
| Cost MAPE | 6.2034% | 4.9088% | 20.87% |

### 10.5 Interpretation

The five-seed experiment supports the following conclusions:

1. The GNN consistently and substantially outperforms the mean baseline.
2. The nearest-neighbor baseline remains stronger on this fixed-topology holdout.
3. Generator dispatch errors are relatively stable across random seeds.
4. Branch-flow errors show moderate seed sensitivity.
5. Global active- and reactive-power balance errors are the least stable metrics.
6. The result demonstrates partial compositional generalization, but not superiority over retrieval-based baselines.

The nearest-neighbor advantage is important because all experiments use the same case30 topology. Dense scenario coverage allows retrieval from nearby training operating points.

### 10.6 Early-stopping ablation

A preliminary five-seed experiment used early stopping with patience 25.

Compared with full 100-epoch training:

| Metric | Early-stop mean | Full-100 mean |
|---|---:|---:|
| Pg MAE | 4.4591 MW | 4.4268 MW |
| Qg MAE | 2.3370 MVAr | 2.3153 MVAr |
| Branch P MAE | 2.0661 MW | 1.8489 MW |
| Branch Q MAE | 0.8467 MVAr | 0.7661 MVAr |
| KCL P MAE | 0.6888 MW | 0.6074 MW |
| KCL Q MAE | 0.3686 MVAr | 0.3303 MVAr |
| Cost MAPE | 4.9132% | 4.9088% |
| Cost MAPE standard deviation | 0.5138% | 0.3500% |

Full 100-epoch training particularly improved branch-flow and KCL metrics.

Several runs reached their best checkpoint late:

~~~text
seed 42: epoch 89
seed 44: epoch 97
seed 45: epoch 88
seed 46: epoch 94
~~~

Therefore, early-stopping patience 25 is too short for the final reported experiment.

### 10.7 Reproducibility artifacts

Training script:

- `scripts/run_stage13_multiseed_full100.sh`

Aggregation script:

- `scripts/summarize_seed_group.py`

Formal result report:

- `reports/stage13_full100_holdout_killgen_loads_5seed.md`

Preliminary early-stopping report:

- `reports/stage13_preliminary_earlystop25_holdout_killgen_loads_5seed.md`
<!-- STAGE13_END -->

<!-- STAGE13_UNSEEN_GENERATOR_START -->
## 11. Stage13.1 unseen-generator holdout

This experiment evaluates zero-shot generalization to a generator outage that is completely absent from the training set.

### 11.1 Split definition

- held-out generator node index: `0`;
- source graphs: `5806`;
- training graphs: `3920`;
- validation graphs: `1886`;
- training condition: the held-out generator is always online;
- validation condition: the held-out generator is always offline;
- leakage check: passed;
- seeds: 42, 43, 44, 45, 46;
- training length: 100 epochs;
- early stopping: disabled.

Regular generator nodes are placed before the ext-grid node in the graph representation, so held-out generator index `0` refers to a regular generator rather than the slack/ext-grid node.

### 11.2 Five-seed results

| Metric | Mean ± Std | Min | Max |
|---|---:|---:|---:|
| Theta MAE | 0.0338 ± 0.0048 | 0.0280 | 0.0402 |
| V MAE | 0.0140 ± 0.0059 p.u. | 0.0090 | 0.0232 |
| Validation loss | 0.0497 ± 0.0031 | 0.0445 | 0.0526 |
| Pg MAE | 10.5641 ± 0.8961 MW | 9.4911 | 11.4040 |
| Qg MAE | 8.6426 ± 0.6264 MVAr | 7.7371 | 9.3880 |
| Branch P MAE | 4.4950 ± 0.3377 MW | 4.2404 | 4.9725 |
| Branch Q MAE | 2.4207 ± 0.6909 MVAr | 2.0101 | 3.6245 |
| KCL P MAE | 2.1394 ± 0.7939 MW | 1.4957 | 3.0205 |
| KCL Q MAE | 1.7306 ± 0.6338 MVAr | 1.0435 | 2.4314 |
| Balance P MAE | 16.2955 ± 7.9580 MW | 5.5718 | 24.6558 |
| Balance Q MAE | 9.2943 ± 4.1040 MVAr | 4.8126 | 13.5694 |
| Cost MAPE | 21.4438 ± 1.6324% | 20.0712 | 23.4444 |

### 11.3 Comparison with the combination holdout

| Metric | Combination holdout | Unseen-generator holdout | Error multiplier |
|---|---:|---:|---:|
| Pg MAE | 4.4268 MW | 10.5641 MW | 2.39× |
| Qg MAE | 2.3153 MVAr | 8.6426 MVAr | 3.73× |
| Branch P MAE | 1.8489 MW | 4.4950 MW | 2.43× |
| Branch Q MAE | 0.7661 MVAr | 2.4207 MVAr | 3.16× |
| KCL P MAE | 0.6074 MW | 2.1394 MW | 3.52× |
| KCL Q MAE | 0.3303 MVAr | 1.7306 MVAr | 5.24× |
| Balance P MAE | 2.8444 MW | 16.2955 MW | 5.73× |
| Balance Q MAE | 2.0175 MVAr | 9.2943 MVAr | 4.61× |
| Cost MAPE | 4.9088% | 21.4438% | 4.37× |

### 11.4 Interpretation

The unseen-generator split is substantially harder than the earlier combination holdout.

The results indicate:

1. outage-aware features allow the network to represent a generator as online or offline;
2. the model performs reasonably when outage patterns are represented in the training distribution;
3. the model does not strongly generalize to a generator whose offline state is never observed during training;
4. dispatch, reactive-power prediction, system balance, and generation cost degrade sharply;
5. the very large balance-error standard deviations show strong sensitivity to random initialization under this out-of-distribution condition.

Therefore, outage-aware encoding is necessary but not sufficient for zero-shot generator-outage generalization.

The correct conclusion is not that the model has learned arbitrary outage compositionality. Instead, it has learned partial interpolation across outage combinations represented in training.

### 11.5 Recommended follow-up

Potential improvements include:

- balanced outage sampling across all regular generators;
- leave-one-generator-out training curricula;
- generator identity or electrical-position embeddings;
- bus-generator structural positional encodings;
- auxiliary prediction of total available generation;
- physics projection after neural inference;
- contingency-aware pretraining;
- training across multiple grid topologies.

Detailed report:

- `reports/stage13_unseen_generator_5seed.md`

Split manifest:

- `reports/stage13_unseen_generator_manifest.json`

Reproduction tools:

- `scripts/make_unseen_generator_split.py`
- `scripts/run_stage13_unseen_generator_5seed.sh`
<!-- STAGE13_UNSEEN_GENERATOR_END -->

<!-- STAGE13_UNSEEN_LOAD_START -->
## 12. Stage13.2 unseen high-load range holdout

This experiment evaluates extrapolation from lower-load operating points to a disjoint high-load validation range.

### 12.1 Split definition

- source graphs: `5806`;
- training graphs: `4898`;
- excluded gap graphs: `37`;
- validation graphs: `871`;
- training quantile cutoff: `0.70`;
- validation quantile cutoff: `0.85`;
- maximum training load: `189.2000 MW`;
- minimum validation load: `189.3581 MW`;
- separation gap: `0.1581 MW`;
- leakage check: passed;
- seeds: 42, 43, 44, 45, 46;
- training length: 100 epochs;
- early stopping: disabled.

The training split contains operating points at or below the lower-load cutoff. Validation contains only operating points at or above the high-load cutoff. Samples in the intermediate range are excluded.

### 12.2 Five-seed aggregate results

| Metric | Mean ± Std | Min | Max |
|---|---:|---:|---:|
| Theta MAE | 0.0161 ± 0.0017 | 0.0146 | 0.0187 |
| V MAE | 0.0069 ± 0.0006 p.u. | 0.0066 | 0.0079 |
| Validation loss | 0.0075 ± 0.0001 | 0.0074 | 0.0077 |
| Pg MAE | 4.5614 ± 0.0905 MW | 4.4094 | 4.6426 |
| Qg MAE | 2.7130 ± 0.1161 MVAr | 2.6102 | 2.9020 |
| Branch P MAE | 1.7551 ± 0.0651 MW | 1.6870 | 1.8540 |
| Branch Q MAE | 0.8534 ± 0.0196 MVAr | 0.8249 | 0.8741 |
| KCL P MAE | 0.5202 ± 0.0735 MW | 0.4415 | 0.6028 |
| KCL Q MAE | 0.3204 ± 0.0640 MVAr | 0.2549 | 0.4064 |
| Balance P MAE | 2.2812 ± 0.9940 MW | 1.3611 | 3.9012 |
| Balance Q MAE | 0.9845 ± 0.3006 MVAr | 0.6010 | 1.2429 |
| Cost MAPE | 7.7132 ± 0.2475% | 7.3589 | 7.9998 |

### 12.3 Comparison with the combination holdout

| Metric | Combination holdout | Unseen high-load | Relative change |
|---|---:|---:|---:|
| Pg MAE | 4.4268 MW | 4.5614 MW | +3.04% |
| Qg MAE | 2.3153 MVAr | 2.7130 MVAr | +17.18% |
| Branch P MAE | 1.8489 MW | 1.7551 MW | -5.07% |
| Branch Q MAE | 0.7661 MVAr | 0.8534 MVAr | +11.40% |
| KCL P MAE | 0.6074 MW | 0.5202 MW | -14.36% |
| KCL Q MAE | 0.3303 MVAr | 0.3204 MVAr | -3.00% |
| Balance P MAE | 2.8444 MW | 2.2812 MW | -19.80% |
| Balance Q MAE | 2.0175 MVAr | 0.9845 MVAr | -51.20% |
| Cost MAPE | 4.9088% | 7.7132% | +57.13% |

### 12.4 Interpretation

The unseen high-load split is considerably easier than the unseen-generator split.

The results indicate:

1. generator active-power prediction changes only slightly under the high-load holdout;
2. reactive-power prediction degrades more noticeably;
3. branch-flow and KCL errors remain stable;
4. global power-balance errors do not deteriorate and are lower than in the earlier combination holdout;
5. cost prediction degrades substantially, increasing from 4.9088% to 7.7132% MAPE;
6. random-seed variability remains relatively low for most local prediction metrics.

This suggests that the model can extrapolate many electrical state and flow quantities over the tested load range, while generation-cost estimation is more sensitive to high-load distribution shift.

### 12.5 Important limitation

The numerical separation between the maximum training load and minimum validation load is only `0.1581 MW`.

Although the quantile partitions are disjoint and the leakage check passes, the absolute load gap is small because many samples share similar aggregate load levels. Therefore, this experiment should be described as a quantile-based high-load holdout rather than strong long-range extrapolation.

A stronger follow-up should generate a dedicated validation set with load scaling beyond the maximum scaling used for training.

### 12.6 Reproducibility artifacts

Detailed report:

- `reports/stage13_unseen_load_5seed.md`

Aggregate CSV:

- `reports/stage13_unseen_load_5seed.csv`

Per-seed CSV:

- `reports/stage13_unseen_load_5seed_per_seed.csv`

Split manifest:

- `reports/stage13_unseen_load_manifest.json`

Reproduction tools:

- `scripts/make_unseen_load_split.py`
- `scripts/run_stage13_unseen_load_5seed.sh`
<!-- STAGE13_UNSEEN_LOAD_END -->

<!-- STAGE13_TOPOLOGY_START -->
## 13. Stage13.3 topology-changing held-out line experiment

This experiment evaluates whether the heterogeneous GraphSAGE model can generalize to a single-line outage topology that is completely absent from training.

### 13.1 Split construction

- case: IEEE 30-bus system;
- held-out line position: `40`;
- held-out pandapower line index: `40`;
- held-out connection: bus `5 -> 27`;
- training graphs: `2638`;
- validation graphs: `80`;
- intact-topology training graphs: `699`;
- held-out outage validation graphs: `80`;
- training topologies: intact topology and successful single-line outages other than line 40;
- validation topology: line 40 outage only;
- leakage check: passed;
- seeds: 42, 43, 44, 45, 46;
- training length: 100 epochs;
- early stopping: disabled.

Each outage operating point was independently solved using AC optimal power flow. The line outage changes the branch-node set, endpoint relations, cycle basis, optimal generator dispatch, voltage solution, and branch-flow targets.

### 13.2 Five-seed aggregate results

| Metric | Mean ± Std | Min | Max |
|---|---:|---:|---:|
| Theta MAE | 0.0083 ± 0.0043 | 0.0058 | 0.0159 |
| V MAE | 0.0056 ± 0.0012 p.u. | 0.0044 | 0.0074 |
| Validation loss | 0.0078 ± 0.0018 | 0.0061 | 0.0098 |
| Pg MAE | 3.2478 ± 0.1521 MW | 3.0748 | 3.4249 |
| Qg MAE | 3.3172 ± 0.0868 MVAr | 3.2087 | 3.4313 |
| Branch P MAE | 1.4060 ± 0.1802 MW | 1.2340 | 1.6953 |
| Branch Q MAE | 1.3759 ± 0.1557 MVAr | 1.1915 | 1.5474 |
| KCL P MAE | 1.3008 ± 0.4013 MW | 0.9589 | 1.7475 |
| KCL Q MAE | 1.3747 ± 0.3577 MVAr | 0.9938 | 1.8404 |
| Balance P MAE | 3.0652 ± 1.5245 MW | 1.5326 | 5.6028 |
| Balance Q MAE | 2.8411 ± 1.4299 MVAr | 1.4619 | 5.2456 |
| Cost MAPE | 5.6195 ± 0.7062% | 4.7710 | 6.3397 |

### 13.3 Per-seed behavior

| Seed | Best epoch | Validation loss | Pg MAE | Branch P MAE | KCL P MAE | Cost MAPE |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 76 | 0.0070 | 3.2905 MW | 1.3661 MW | 0.9589 MW | 4.7710% |
| 43 | 72 | 0.0063 | 3.1053 MW | 1.2340 MW | 1.0374 MW | 4.9586% |
| 44 | 44 | 0.0097 | 3.4249 MW | 1.4449 MW | 1.7475 MW | 6.0768% |
| 45 | 25 | 0.0098 | 3.0748 MW | 1.6953 MW | 1.7307 MW | 5.9513% |
| 46 | 34 | 0.0061 | 3.3435 MW | 1.2899 MW | 1.0296 MW | 6.3397% |

All five runs completed the full 100 epochs. The reported metrics use the checkpoint with the lowest validation loss.

### 13.4 Comparison with previous holdouts

| Metric | Combination holdout | High-load holdout | Topology holdout |
|---|---:|---:|---:|
| Pg MAE | 4.4268 MW | 4.5614 MW | 3.2478 MW |
| Qg MAE | 2.3153 MVAr | 2.7130 MVAr | 3.3172 MVAr |
| Branch P MAE | 1.8489 MW | 1.7551 MW | 1.4060 MW |
| Branch Q MAE | 0.7661 MVAr | 0.8534 MVAr | 1.3759 MVAr |
| KCL P MAE | 0.6074 MW | 0.5202 MW | 1.3008 MW |
| KCL Q MAE | 0.3303 MVAr | 0.3204 MVAr | 1.3747 MVAr |
| Balance P MAE | 2.8444 MW | 2.2812 MW | 3.0652 MW |
| Balance Q MAE | 2.0175 MVAr | 0.9845 MVAr | 2.8411 MVAr |
| Cost MAPE | 4.9088% | 7.7132% | 5.6195% |

The topology holdout has lower active-power and branch-active-flow errors than the earlier combination and high-load holdouts. However, reactive-power, KCL, and global balance errors are substantially higher.

### 13.5 Interpretation

The model successfully processes and predicts a graph topology that is absent from training. This supports the inductive capability of the heterogeneous GraphSAGE architecture.

The results show three different behaviors:

1. **Voltage and active-flow prediction remain effective.**
   Voltage magnitude, generator active power, and branch active power retain relatively low errors.

2. **Reactive-power prediction is more topology-sensitive.**
   Generator reactive power and branch reactive flow deteriorate more strongly under the unseen outage.

3. **Physical consistency is less stable than supervised prediction.**
   KCL and global balance errors increase, even when direct node and branch prediction errors remain moderate.

This indicates that accurate supervised predictions do not automatically guarantee strict physical consistency after a topology shift.

### 13.6 Seed sensitivity

The standard deviations of local supervised metrics are moderate, but KCL and balance metrics vary strongly across seeds.

For example, Balance P MAE ranges from `1.5326 MW` to `5.6028 MW`. The best checkpoint epoch ranges from 25 to 76.

This suggests that the topology-generalization solution has multiple optimization regimes. Model selection based only on validation loss may not consistently select the checkpoint with the best physical residuals.

### 13.7 Limitations

This experiment has several important limitations:

- validation contains only 80 graphs;
- only one line outage is held out;
- the held-out line was selected from outage candidates with high successful-sample availability;
- different line outages can have very different electrical difficulty;
- the current objective does not directly enforce KCL or global power balance because `lambda_feas=0`;
- validation loss is not identical to the final physical-consistency metrics.

Therefore, the result demonstrates single-topology inductive generalization, but does not establish uniform N−1 contingency generalization.

A stronger evaluation should repeat the holdout across several electrically diverse lines and report macro-averaged results.

### 13.8 Reproducibility artifacts

Detailed report:

- `reports/stage13_topology_5seed.md`

Aggregate results:

- `reports/stage13_topology_5seed.csv`

Per-seed results:

- `reports/stage13_topology_5seed_per_seed.csv`

Split manifest:

- `reports/stage13_topology_manifest.json`

Generation and validation tools:

- `scripts/generate_topology_outage_dataset.py`
- `scripts/check_topology_dataset.py`

Training launcher:

- `scripts/run_stage13_topology_5seed.sh`
<!-- STAGE13_TOPOLOGY_END -->
