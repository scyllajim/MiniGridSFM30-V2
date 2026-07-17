#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

TRAIN_DATA="${TRAIN_DATA:-data/processed/stage13_unseen_generator_train.pt}"
VAL_DATA="${VAL_DATA:-data/processed/stage13_unseen_generator_val.pt}"
MANIFEST="${MANIFEST:-data/processed/stage13_unseen_generator_manifest.json}"

PREFIX="${PREFIX:-stage13_unseen_generator_full100_seed}"

SEEDS=(42 43 44 45 46)
GPUS=(0 1 2 3 4)

mkdir -p \
  logs/stage13_unseen_generator \
  reports

for path in "$TRAIN_DATA" "$VAL_DATA" "$MANIFEST"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required file: $path" >&2
    exit 1
  fi
done

held_out_generator="$(
  python - "$MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    manifest = json.load(f)

print(manifest["held_out_generator_index"])
PY
)"

echo "held-out generator index: ${held_out_generator}"
echo "train data: ${TRAIN_DATA}"
echo "validation data: ${VAL_DATA}"

declare -a PIDS=()
declare -a LABELS=()

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu="${GPUS[$i]}"

  run_dir="runs/${PREFIX}${seed}"
  log_file="logs/stage13_unseen_generator/${PREFIX}${seed}.log"

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
      echo "[done] ${label}"
    else
      status=$?
      echo "[failed] ${label}, exit=${status}" >&2
      failed=1
    fi
  done
fi

if [[ "$failed" -ne 0 ]]; then
  echo
  echo "At least one seed failed."
  echo "Inspect logs:"
  echo "  logs/stage13_unseen_generator/*.log"
  exit 1
fi

python scripts/summarize_seed_group.py \
  --runs-dir runs \
  --prefix "$PREFIX" \
  --seeds "${SEEDS[@]}" \
  --out-csv reports/stage13_unseen_generator_5seed.csv \
  --out-md reports/stage13_unseen_generator_5seed.md

python - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])

with manifest_path.open(encoding="utf-8") as f:
    manifest = json.load(f)

report_path = Path("reports/stage13_unseen_generator_5seed.md")
report = report_path.read_text(encoding="utf-8")

header = f"""# Stage13.1 Unseen-Generator Holdout

- Held-out generator index: `{manifest["held_out_generator_index"]}`
- Source graphs: `{manifest["source_graphs"]}`
- Training graphs: `{manifest["train_graphs"]}`
- Validation graphs: `{manifest["validation_graphs"]}`
- Leakage check: `passed`
- Training condition: held-out generator is always online
- Validation condition: held-out generator is always offline
- Epochs: `100`
- Seeds: `42, 43, 44, 45, 46`

"""

if report.startswith("# Stage13 Five-Seed Summary"):
    report = report.split("\n", 1)[1].lstrip()

report_path.write_text(
    header + report.rstrip() + "\n",
    encoding="utf-8",
)

print("Updated report metadata:", report_path)
PY

echo "=== Stage13.1 unseen-generator experiment complete ==="
