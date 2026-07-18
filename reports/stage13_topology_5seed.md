# Stage13.3 Topology-Changing Held-Out Line Experiment

- Held-out line position: `40`
- Held-out pandapower line index: `40`
- Held-out buses: `5 -> 27`
- Training graphs: `2638`
- Validation graphs: `80`
- Base-topology training graphs: `699`
- Held-out outage validation graphs: `80`
- Leakage check: `passed`
- Epochs: `100`
- Early stopping: `disabled`
- Seeds: `42, 43, 44, 45, 46`

Training contains the intact topology and successful single-line outages other than the held-out line. Validation contains only the held-out line outage.

Run prefix: `stage13_topology_full100_seed`
Seeds: 42, 43, 44, 45, 46

## Aggregate results

| Metric | Mean ± Std | Min | Max | Unit | N |
|---|---:|---:|---:|---|---:|
| Theta MAE | 0.0083 ± 0.0043 | 0.0058 | 0.0159 |  | 5 |
| V MAE | 0.0056 ± 0.0012 | 0.0044 | 0.0074 | p.u. | 5 |
| Validation loss | 0.0078 ± 0.0018 | 0.0061 | 0.0098 |  | 5 |
| Pg MAE | 3.2478 ± 0.1521 | 3.0748 | 3.4249 | MW | 5 |
| Qg MAE | 3.3172 ± 0.0868 | 3.2087 | 3.4313 | MVAr | 5 |
| Branch P MAE | 1.4060 ± 0.1802 | 1.2340 | 1.6953 | MW | 5 |
| Branch Q MAE | 1.3759 ± 0.1557 | 1.1915 | 1.5474 | MVAr | 5 |
| KCL P MAE | 1.3008 ± 0.4013 | 0.9589 | 1.7475 | MW | 5 |
| KCL Q MAE | 1.3747 ± 0.3577 | 0.9938 | 1.8404 | MVAr | 5 |
| Balance P MAE | 3.0652 ± 1.5245 | 1.5326 | 5.6028 | MW | 5 |
| Balance Q MAE | 2.8411 ± 1.4299 | 1.4619 | 5.2456 | MVAr | 5 |
| Cost MAPE | 5.6195 ± 0.7062 | 4.7710 | 6.3397 | % | 5 |

## Per-seed results

| Seed | Epoch | Pg MW | Qg MVAr | BrP MW | BrQ MVAr | KCL P MW | Balance P MW | Cost % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 76 | 3.2905 | 3.4313 | 1.3661 | 1.3585 | 0.9589 | 2.3930 | 4.7710 |
| 43 | 72 | 3.1053 | 3.2087 | 1.2340 | 1.2630 | 1.0374 | 5.6028 | 4.9586 |
| 44 | 44 | 3.4249 | 3.3563 | 1.4449 | 1.5191 | 1.7475 | 1.5326 | 6.0768 |
| 45 | 25 | 3.0748 | 3.2572 | 1.6953 | 1.5474 | 1.7307 | 2.9095 | 5.9513 |
| 46 | 34 | 3.3435 | 3.3327 | 1.2899 | 1.1915 | 1.0296 | 2.8882 | 6.3397 |
