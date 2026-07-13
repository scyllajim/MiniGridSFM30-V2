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
