# Stage13.1 Unseen-Generator Holdout

- Held-out generator index: `0`
- Source graphs: `5806`
- Training graphs: `3920`
- Validation graphs: `1886`
- Leakage check: `passed`
- Training condition: held-out generator is always online
- Validation condition: held-out generator is always offline
- Epochs: `100`
- Seeds: `42, 43, 44, 45, 46`

Run prefix: `stage13_unseen_generator_full100_seed`
Seeds: 42, 43, 44, 45, 46

## Aggregate results

| Metric | Mean ± Std | Min | Max | Unit | N |
|---|---:|---:|---:|---|---:|
| Theta MAE | 0.0338 ± 0.0048 | 0.0280 | 0.0402 |  | 5 |
| V MAE | 0.0140 ± 0.0059 | 0.0090 | 0.0232 | p.u. | 5 |
| Validation loss | 0.0497 ± 0.0031 | 0.0445 | 0.0526 |  | 5 |
| Pg MAE | 10.5641 ± 0.8961 | 9.4911 | 11.4040 | MW | 5 |
| Qg MAE | 8.6426 ± 0.6264 | 7.7371 | 9.3880 | MVAr | 5 |
| Branch P MAE | 4.4950 ± 0.3377 | 4.2404 | 4.9725 | MW | 5 |
| Branch Q MAE | 2.4207 ± 0.6909 | 2.0101 | 3.6245 | MVAr | 5 |
| KCL P MAE | 2.1394 ± 0.7939 | 1.4957 | 3.0205 | MW | 5 |
| KCL Q MAE | 1.7306 ± 0.6338 | 1.0435 | 2.4314 | MVAr | 5 |
| Balance P MAE | 16.2955 ± 7.9580 | 5.5718 | 24.6558 | MW | 5 |
| Balance Q MAE | 9.2943 ± 4.1040 | 4.8126 | 13.5694 | MVAr | 5 |
| Cost MAPE | 21.4438 ± 1.6324 | 20.0712 | 23.4444 | % | 5 |

## Per-seed results

| Seed | Epoch | Pg MW | Qg MVAr | BrP MW | BrQ MVAr | KCL P MW | Balance P MW | Cost % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 35 | 11.4040 | 8.7985 | 4.2683 | 2.0226 | 1.5973 | 17.3336 | 20.0712 |
| 43 | 10 | 9.4911 | 7.7371 | 4.7333 | 2.3891 | 3.0205 | 5.5718 | 23.4444 |
| 44 | 5 | 11.3208 | 8.3535 | 4.9725 | 3.6245 | 2.9955 | 11.1930 | 20.5572 |
| 45 | 64 | 9.7344 | 9.3880 | 4.2404 | 2.0574 | 1.5881 | 24.6558 | 22.9789 |
| 46 | 25 | 10.8702 | 8.9362 | 4.2602 | 2.0101 | 1.4957 | 22.7231 | 20.1671 |
