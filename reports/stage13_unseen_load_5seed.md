# Stage13.2 Unseen High-Load Range Holdout

- Source graphs: `5806`
- Training graphs: `4898`
- Gap graphs excluded: `37`
- Validation graphs: `871`
- Training maximum load: `189.2000 MW`
- Validation minimum load: `189.3581 MW`
- Separation gap: `0.1581 MW`
- Train quantile cutoff: `0.70`
- Validation quantile cutoff: `0.85`
- Leakage check: `passed`
- Epochs: `100`
- Early stopping: `disabled`
- Seeds: `42, 43, 44, 45, 46`

Run prefix: `stage13_unseen_load_full100_seed`
Seeds: 42, 43, 44, 45, 46

## Aggregate results

| Metric | Mean ± Std | Min | Max | Unit | N |
|---|---:|---:|---:|---|---:|
| Theta MAE | 0.0161 ± 0.0017 | 0.0146 | 0.0187 |  | 5 |
| V MAE | 0.0069 ± 0.0006 | 0.0066 | 0.0079 | p.u. | 5 |
| Validation loss | 0.0075 ± 0.0001 | 0.0074 | 0.0077 |  | 5 |
| Pg MAE | 4.5614 ± 0.0905 | 4.4094 | 4.6426 | MW | 5 |
| Qg MAE | 2.7130 ± 0.1161 | 2.6102 | 2.9020 | MVAr | 5 |
| Branch P MAE | 1.7551 ± 0.0651 | 1.6870 | 1.8540 | MW | 5 |
| Branch Q MAE | 0.8534 ± 0.0196 | 0.8249 | 0.8741 | MVAr | 5 |
| KCL P MAE | 0.5202 ± 0.0735 | 0.4415 | 0.6028 | MW | 5 |
| KCL Q MAE | 0.3204 ± 0.0640 | 0.2549 | 0.4064 | MVAr | 5 |
| Balance P MAE | 2.2812 ± 0.9940 | 1.3611 | 3.9012 | MW | 5 |
| Balance Q MAE | 0.9845 ± 0.3006 | 0.6010 | 1.2429 | MVAr | 5 |
| Cost MAPE | 7.7132 ± 0.2475 | 7.3589 | 7.9998 | % | 5 |

## Per-seed results

| Seed | Epoch | Pg MW | Qg MVAr | BrP MW | BrQ MVAr | KCL P MW | Balance P MW | Cost % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 47 | 4.5937 | 2.6102 | 1.7656 | 0.8582 | 0.4521 | 1.5960 | 7.7103 |
| 43 | 48 | 4.6426 | 2.9020 | 1.8540 | 0.8741 | 0.5218 | 1.3611 | 7.8804 |
| 44 | 67 | 4.5557 | 2.7429 | 1.6870 | 0.8249 | 0.4415 | 2.1975 | 7.9998 |
| 45 | 54 | 4.6057 | 2.6572 | 1.7065 | 0.8433 | 0.5828 | 2.3503 | 7.3589 |
| 46 | 38 | 4.4094 | 2.6526 | 1.7623 | 0.8663 | 0.6028 | 3.9012 | 7.6168 |
