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
