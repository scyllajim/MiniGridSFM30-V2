#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

TRAIN_DATA="${TRAIN_DATA:-data/processed/stage11_train_pure_plus_killgen_derate_feasible_graphs.pt}"
VAL_DATA="${VAL_DATA:-data/processed/stage11_val_killgen_loads_feasible_graphs.pt}"
PREFIX="${PREFIX:-stage13_full100_holdout_killgen_loads_seed}"

SEEDS=(42 43 44 45 46)
GPUS=(0 1 2 3 4)

mkdir -p logs/stage13
declare -a PIDS=()
declare -a LABELS=()

for path in "$TRAIN_DATA" "$VAL_DATA"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing dataset: $path" >&2
    exit 1
  fi
done

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu="${GPUS[$i]}"
  run_dir="runs/${PREFIX}${seed}"
  log_file="logs/stage13/${PREFIX}${seed}.log"

  if [[ -f "${run_dir}/best_model.pt" && -f "${run_dir}/metrics.csv" ]]; then
    echo "[skip] seed=${seed}: completed run already exists"
    continue
  fi

  echo "[start] seed=${seed} gpu=${gpu} log=${log_file}"

  python scripts/train_split.py \
    --train-data "$TRAIN_DATA" \
    --val-data "$VAL_DATA" \
    --run-dir "$run_dir" \
    --epochs 100 \
    --batch-size 16 \
    --hidden-dim 128 \
    --num-layers 3 \
    --lr 1e-3 \
    --seed "$seed" \
    --device "cuda:${gpu}" \
    --num-workers 0 \
    --early-stopping-patience 0 \
    --early-stopping-min-delta 0 \
    --lambda-feas 0 \
    >"$log_file" 2>&1 &

  PIDS+=("$!")
  LABELS+=("seed=${seed},gpu=${gpu}")
done

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  echo "No new jobs launched."
else
  failed=0

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

  if [[ "$failed" -ne 0 ]]; then
    echo "At least one training failed. Inspect logs/stage13/*.log" >&2
    exit 1
  fi
fi

python scripts/summarize_seed_group.py \
  --runs-dir runs \
  --prefix "$PREFIX" \
  --seeds "${SEEDS[@]}" \
  --out-csv reports/stage13_full100_holdout_killgen_loads_5seed.csv \
  --out-md reports/stage13_full100_holdout_killgen_loads_5seed.md

echo "=== Stage13 full-100 five-seed experiment complete ==="
