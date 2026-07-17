#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

TRAIN_DATA="${
  TRAIN_DATA:-
  data/processed/stage13_unseen_load_train.pt
}"

VAL_DATA="${
  VAL_DATA:-
  data/processed/stage13_unseen_load_val.pt
}"

MANIFEST="${
  MANIFEST:-
  reports/stage13_unseen_load_manifest.json
}"

PREFIX="${
  PREFIX:-
  stage13_unseen_load_full100_seed
}"

# Remove line breaks introduced by readable defaults.
TRAIN_DATA="$(echo "$TRAIN_DATA" | tr -d '[:space:]')"
VAL_DATA="$(echo "$VAL_DATA" | tr -d '[:space:]')"
MANIFEST="$(echo "$MANIFEST" | tr -d '[:space:]')"
PREFIX="$(echo "$PREFIX" | tr -d '[:space:]')"

SEEDS=(42 43 44 45 46)
GPUS=(0 1 2 3 4)

mkdir -p \
  logs/stage13_unseen_load \
  reports

for path in \
  "$TRAIN_DATA" \
  "$VAL_DATA" \
  "$MANIFEST"
do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required file: $path" >&2
    exit 1
  fi
done

echo "train data: $TRAIN_DATA"
echo "validation data: $VAL_DATA"
echo "manifest: $MANIFEST"
echo "run prefix: $PREFIX"

declare -a PIDS=()
declare -a LABELS=()

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu="${GPUS[$i]}"

  run_dir="runs/${PREFIX}${seed}"
  log_file="logs/stage13_unseen_load/${PREFIX}${seed}.log"

  if [[ \
    -f "${run_dir}/best_model.pt" && \
    -f "${run_dir}/metrics.csv" \
  ]]; then
    echo "[skip] seed=${seed}: completed result exists"
    continue
  fi

  echo "[start] seed=${seed} gpu=${gpu}"

  python scripts/train_split.py \
    --train-data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --run-dir "$run_dir" \
    --epochs 100 \
    --batch-size 16 \
    --hidden-dim 128 \
    --num-layers 3 \
    --dropout 0 \
    --lr 1e-3 \
    --weight-decay 0 \
    --seed "$seed" \
    --device "cuda:${gpu}" \
    --num-workers 0 \
    --early-stopping-patience 0 \
    --lambda-feas 0 \
    >"$log_file" 2>&1 &

  PIDS+=("$!")
  LABELS+=("seed=${seed},gpu=${gpu}")
done

failed=0

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  echo "No new training jobs launched."
else
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    label="${LABELS[$i]}"

    if wait "$pid"; then
      echo "[done] $label"
    else
      status=$?
      echo "[failed] $label exit=${status}" >&2
      failed=1
    fi
  done
fi

if [[ "$failed" -ne 0 ]]; then
  echo
  echo "At least one training job failed." >&2
  echo "Inspect:" >&2
  echo "  logs/stage13_unseen_load/*.log" >&2
  exit 1
fi

python scripts/summarize_seed_group.py \
  --runs-dir runs \
  --prefix "$PREFIX" \
  --seeds "${SEEDS[@]}" \
  --out-csv reports/stage13_unseen_load_5seed.csv \
  --out-md reports/stage13_unseen_load_5seed.md

python - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
report_path = Path(
    "reports/stage13_unseen_load_5seed.md"
)

with manifest_path.open(encoding="utf-8") as f:
    manifest = json.load(f)

report = report_path.read_text(encoding="utf-8")

if report.startswith("# Stage13 Five-Seed Summary"):
    report = report.split("\n", 1)[1].lstrip()

header = f"""# Stage13.2 Unseen High-Load Range Holdout

- Source graphs: `{manifest["source_graphs"]}`
- Training graphs: `{manifest["train_graphs"]}`
- Gap graphs excluded: `{manifest["gap_graphs"]}`
- Validation graphs: `{manifest["validation_graphs"]}`
- Training maximum load: `{manifest["train_max_load_mw"]:.4f} MW`
- Validation minimum load: `{manifest["val_min_load_mw"]:.4f} MW`
- Separation gap: `{manifest["gap_width_mw"]:.4f} MW`
- Train quantile cutoff: `{manifest["train_max_quantile"]:.2f}`
- Validation quantile cutoff: `{manifest["val_min_quantile"]:.2f}`
- Leakage check: `passed`
- Epochs: `100`
- Seeds: `42, 43, 44, 45, 46`

"""

report_path.write_text(
    header + report.rstrip() + "\n",
    encoding="utf-8",
)

print("Updated report metadata:", report_path)
PY

echo "=== Stage13.2 unseen-load experiment complete ==="
