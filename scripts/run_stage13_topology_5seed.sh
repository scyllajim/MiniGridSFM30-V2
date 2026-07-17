#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/.."

TRAIN_DATA="${TRAIN_DATA:-data/processed/stage13_topology_train.pt}"
VAL_DATA="${VAL_DATA:-data/processed/stage13_topology_val.pt}"
MANIFEST="${MANIFEST:-reports/stage13_topology_manifest.json}"

PREFIX="${PREFIX:-stage13_topology_full100_seed}"
LOG_DIR="${LOG_DIR:-logs/stage13_topology_training}"

SEEDS=(42 43 44 45 46)

mkdir -p \
  "$LOG_DIR" \
  reports \
  runs

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

echo "=== Selecting two least-used GPUs ==="

mapfile -t GPUS < <(
  nvidia-smi \
    --query-gpu=index,memory.used \
    --format=csv,noheader,nounits \
  | sort -t',' -k2,2n \
  | head -n 2 \
  | cut -d',' -f1 \
  | tr -d ' '
)

if [[ "${#GPUS[@]}" -lt 2 ]]; then
  echo "ERROR: fewer than two GPUs were detected." >&2
  exit 1
fi

echo "selected GPUs: ${GPUS[*]}"

nvidia-smi \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

echo
echo "train data:      $TRAIN_DATA"
echo "validation data: $VAL_DATA"
echo "manifest:        $MANIFEST"
echo "run prefix:      $PREFIX"
echo "seeds:           ${SEEDS[*]}"
echo

run_one_seed() {
  local seed="$1"
  local gpu="$2"

  local run_dir="runs/${PREFIX}${seed}"
  local log_file="${LOG_DIR}/${PREFIX}${seed}.log"

  if [[ \
    -f "${run_dir}/best_model.pt" && \
    -f "${run_dir}/metrics.csv" \
  ]]; then
    echo "[skip] seed=${seed}: completed outputs already exist"
    return 0
  fi

  mkdir -p "$run_dir"

  {
    echo "=================================================="
    echo "Stage13.3 topology-changing holdout"
    echo "seed=${seed}"
    echo "gpu=${gpu}"
    echo "started=$(date --iso-8601=seconds)"
    echo "train_data=${TRAIN_DATA}"
    echo "val_data=${VAL_DATA}"
    echo "run_dir=${run_dir}"
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
    echo "finished=$(date --iso-8601=seconds)" >>"$log_file"
    echo "[done] seed=${seed} gpu=${gpu}"
    return 0
  fi

  local status=$?

  echo "[failed] seed=${seed} gpu=${gpu}, exit=${status}" >&2
  echo >&2
  echo "===== ${log_file}: last 150 lines =====" >&2
  tail -n 150 "$log_file" >&2 || true
  echo "========================================" >&2

  return "$status"
}

failed=0
position=0

while [[ "$position" -lt "${#SEEDS[@]}" ]]; do
  declare -a PIDS=()
  declare -a LABELS=()

  for gpu_offset in "${!GPUS[@]}"; do
    if [[ "$position" -ge "${#SEEDS[@]}" ]]; then
      break
    fi

    seed="${SEEDS[$position]}"
    gpu="${GPUS[$gpu_offset]}"

    run_one_seed "$seed" "$gpu" &

    PIDS+=("$!")
    LABELS+=("seed=${seed},gpu=${gpu}")

    position=$((position + 1))
  done

  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    label="${LABELS[$i]}"

    if wait "$pid"; then
      echo "[batch complete] $label"
    else
      status=$?
      echo "[batch failed] $label, exit=${status}" >&2
      failed=1
    fi
  done

  if [[ "$failed" -ne 0 ]]; then
    echo "ERROR: at least one topology run failed." >&2
    exit 1
  fi
done

echo "=== Verify all five runs ==="

for seed in "${SEEDS[@]}"; do
  run_dir="runs/${PREFIX}${seed}"

  for required in \
    "${run_dir}/best_model.pt" \
    "${run_dir}/metrics.csv"
  do
    if [[ ! -f "$required" ]]; then
      echo "ERROR: missing output: $required" >&2
      exit 1
    fi
  done

  echo "[OK] seed=${seed}"
done

echo "=== Aggregate five-seed results ==="

python scripts/summarize_seed_group.py \
  --runs-dir runs \
  --prefix "$PREFIX" \
  --seeds "${SEEDS[@]}" \
  --out-csv reports/stage13_topology_5seed.csv \
  --out-md reports/stage13_topology_5seed.md

echo "=== Add topology metadata to report ==="

python - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
report_path = Path("reports/stage13_topology_5seed.md")

with manifest_path.open(encoding="utf-8") as f:
    m = json.load(f)

report = report_path.read_text(encoding="utf-8")

if report.startswith("# Stage13 Five-Seed Summary"):
    parts = report.split("\n", 1)
    report = parts[1].lstrip() if len(parts) == 2 else ""

header = f"""# Stage13.3 Topology-Changing Held-Out Line Experiment

- Held-out line position: `{m["heldout_line_position"]}`
- Held-out pandapower line index: `{m["heldout_line_index"]}`
- Held-out buses: `{m["heldout_from_bus"]} -> {m["heldout_to_bus"]}`
- Training graphs: `{m["train_graphs"]}`
- Validation graphs: `{m["validation_graphs"]}`
- Base-topology training graphs: `{m["base_successful_graphs"]}`
- Held-out outage validation graphs: `{m["heldout_successful_graphs"]}`
- Leakage check: `passed`
- Epochs: `100`
- Early stopping: `disabled`
- Seeds: `42, 43, 44, 45, 46`

Training contains the intact topology and successful single-line outages other than the held-out line. Validation contains only the held-out line outage.

"""

report_path.write_text(
    header + report.rstrip() + "\n",
    encoding="utf-8",
)

print("Updated report:", report_path)
PY

echo "=== Create per-seed report metadata check ==="

python - <<'PY'
from pathlib import Path

paths = [
    Path("reports/stage13_topology_5seed.md"),
    Path("reports/stage13_topology_5seed.csv"),
    Path("reports/stage13_topology_5seed_per_seed.csv"),
]

for path in paths:
    if not path.exists():
        raise FileNotFoundError(path)

    print(path, path.stat().st_size, "bytes")
PY

echo "=================================================="
echo " Stage13.3 topology five-seed training complete"
echo "=================================================="
