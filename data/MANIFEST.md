# Dataset Manifest

Large raw datasets and processed graph files are not committed to GitHub.

## Main datasets

| Dataset | Requested | Feasible | Infeasible | Seed | Description |
|---|---:|---:|---:|---:|---|
| `case30_pure_modes_3000.pkl` | 3000 | 2751 | 249 | merged seed 9200 | Pure perturbation modes |
| `stage11_killgen_loads_3000.pkl` | 3000 | 2091 | 909 | 12001 | Generator outage plus load variation |
| `stage11_killgen_derate_1000.pkl` | 1000 | 964 | 36 | 12002 | Generator outage plus weak line derating |
| `stage11_case30_outage_augmented` | — | 5806 | — | merged | Main outage-augmented feasible dataset |

## Integrity checks

The stable pure-mode dataset was re-solved and checked for:

- feasible/infeasible label agreement;
- OPF re-solve consistency;
- objective-cost consistency;
- malformed records.

## Local checksums

```bash
sha256sum data/raw/*.pkl
sha256sum data/processed/*.pt
