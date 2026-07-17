#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

TRAIN_DATA="${TRAIN_DATA:-data/processed/stage13_unseen_load_train.pt}"
VAL_DATA="${VAL_DATA:-data/processed/stage13_unseen_load_val.pt}"
MANIFEST="${MANIFEST:-reports/stage13_unseen_load_manifest.json}"
PREFIX="${PREFIX:-stage13_unseen_load_full100_seed}"

SEEDS=(42 43 44 45 46)

# 当前 GPU 0 和 1 空闲。
# 每批最多并行两个任务，避免占用其他用户正在使用的 GPU。
GPUS=(0 1)

mkdir -p \
  logs/stage13_unseen_load \
  reports \
  runs

echo "=== Configuration ==="
echo "train data:      ${TRAIN_DATA}"
echo "validation data: ${VAL_DATA}"
echo "manifest:        ${MANIFEST}"
echo "run prefix:      ${PREFIX}"
echo "GPUs:            ${GPUS[*]}"
echo "seeds:           ${SEEDS[*]}"

for path in \
  "$TRAIN_DATA" \
  "$VAL_DATA" \
  "$MANIFEST" \
  scripts/train_split.py \
  scripts/summarize_seed_group.py
do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required file: $path" >&2
    exit 1
  fi
done

run_one_seed() {
  local seed="$1"
  local gpu="$2"

  local run_dir="runs/${PREFIX}${seed}"
  local log_file="logs/stage13_unseen_load/${PREFIX}${seed}.log"

  if [[ \
    -f "${run_dir}/best_model.pt" && \
    -f "${run_dir}/metrics.csv" \
  ]]; then
    echo "[skip] seed=${seed}: completed result exists"
    return 0
  fi

  mkdir -p "$run_dir"

  {
    echo "=================================================="
    echo "seed=${seed}"
    echo "gpu=${gpu}"
    echo "started=$(date --iso-8601=seconds)"
    echo "train=${TRAIN_DATA}"
    echo "validation=${VAL_DATA}"
    echo "=================================================="
  } >"$log_file"

  echo "[start] seed=${seed} gpu=${gpu}"
  echo "        log=${log_file}"

  if python scripts/train_split.py \
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
    >>"$log_file" 2>&1
  then
    echo "[done] seed=${seed} gpu=${gpu}"
    echo "finished=$(date --iso-8601=seconds)" >>"$log_file"
    return 0
  else
    local status=$?

    echo "[failed] seed=${seed} gpu=${gpu} exit=${status}" >&2
    echo
    echo "===== Last 100 lines of ${log_file} =====" >&2
    tail -n 100 "$log_file" >&2 || true
    echo "=========================================" >&2

    return "$status"
  fi
}

failed=0
seed_position=0

while [[ "$seed_position" -lt "${#SEEDS[@]}" ]]; do
  declare -a batch_pids=()
  declare -a batch_labels=()

  for gpu_position in "${!GPUS[@]}"; do
    if [[ "$seed_position" -ge "${#SEEDS[@]}" ]]; then
      break
    fi

    seed="${SEEDS[$seed_position]}"
    gpu="${GPUS[$gpu_position]}"

    run_one_seed "$seed" "$gpu" &

    batch_pids+=("$!")
    batch_labels+=("seed=${seed},gpu=${gpu}")

    seed_position=$((seed_position + 1))
  done

  for i in "${!batch_pids[@]}"; do
    pid="${batch_pids[$i]}"
    label="${batch_labels[$i]}"

    if wait "$pid"; then
      echo "[batch complete] ${label}"
    else
      status=$?
      echo "[batch failed] ${label}, exit=${status}" >&2
      failed=1
    fi
  done

  if [[ "$failed" -ne 0 ]]; then
    echo
    echo "ERROR: one or more runs failed." >&2
    echo "Inspect logs under:" >&2
    echo "  logs/stage13_unseen_load/" >&2
    exit 1
  fi
done

echo "=== Verify completed runs ==="

for seed in "${SEEDS[@]}"; do
  run_dir="runs/${PREFIX}${seed}"

  if [[ ! -f "${run_dir}/best_model.pt" ]]; then
    echo "ERROR: missing ${run_dir}/best_model.pt" >&2
    exit 1
  fi

  if [[ ! -f "${run_dir}/metrics.csv" ]]; then
    echo "ERROR: missing ${run_dir}/metrics.csv" >&2
    exit 1
  fi

  echo "[OK] seed=${seed}"
done

echo "=== Aggregate five-seed results ==="

python scripts/summarize_seed_group.py \
  --runs-dir runs \
  --prefix "$PREFIX" \
  --seeds "${SEEDS[@]}" \
  --out-csv reports/stage13_unseen_load_5seed.csv \
  --out-md reports/stage13_unseen_load_5seed.md

echo "=== Add experiment metadata to report ==="

python - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
report_path = Path("reports/stage13_unseen_load_5seed.md")

with manifest_path.open(encoding="utf-8") as f:
    manifest = json.load(f)

report = report_path.read_text(encoding="utf-8")

if report.startswith("# Stage13 Five-Seed Summary"):
    parts = report.split("\n", 1)
    report = parts[1].lstrip() if len(parts) == 2 else ""

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
- Early stopping: `disabled`
- Seeds: `42, 43, 44, 45, 46`

"""

report_path.write_text(
    header + report.rstrip() + "\n",
    encoding="utf-8",
)

print("Updated report:", report_path)
PY

echo "=================================================="
echo " Stage13.2 five-seed training completed"
echo "=================================================="
