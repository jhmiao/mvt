#!/bin/bash

#SBATCH --account=ssuen_1733
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --exclusive
#SBATCH --job-name=v3_single
#SBATCH --output=/project2/ssuen_1733/output/v3_single-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=miaojing@usc.edu

# -------------------------
# Load environment
# -------------------------
module purge
module load conda
module load gurobi
source "$(conda info --base)/etc/profile.d/conda.sh"

# -------------------------
# Paths
# -------------------------
REPO_ROOT=/project2/ssuen_1733/mvt
PROJECT_ROOT="$REPO_ROOT/v3"

# Optional: per-job working directory
# WORKDIR=/project2/ssuen_1733/mvt_runs_${SLURM_JOB_ID}
# mkdir -p "$WORKDIR"

# -------------------------
# Python environment
# -------------------------
ENV_PY="/home1/miaojing/.conda/envs/mvt/bin/python"

echo "Using Python:"
"$ENV_PY" -V

echo "Checking core dependencies:"
"$ENV_PY" - <<EOF
import numpy, gurobipy
print("numpy:", numpy.__version__)
print("gurobi:", gurobipy.gurobi.version())
EOF

# -------------------------
# Run
# -------------------------
cd "$REPO_ROOT"
echo "[$(date)] Starting 4-week experiments on $(hostname)"

WORK_LIMIT=1000
WARM_START_AFTER=100
WORKLOAD_PENALTY_WEIGHT=10
LEADERS_PENALTY_WEIGHT=250
OUTPUT_ROOT="$PROJECT_ROOT/outputs/weeks_${WORKLOAD_PENALTY_WEIGHT}_${LEADERS_PENALTY_WEIGHT}"
GUROBI_OUTPUT="${GUROBI_OUTPUT:-0}"

mkdir -p "$OUTPUT_ROOT"

LOCATIONS=(c101 c201 r101 rc101)
EVENT_TYPES=(Even)
SEEDS=(42 43 44 45)
NUM_EVENT_TYPES=${#EVENT_TYPES[@]}
NUM_COMBINATIONS=$((${#LOCATIONS[@]} * NUM_EVENT_TYPES))
ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if (( ARRAY_TASK_ID < 0 || ARRAY_TASK_ID >= NUM_COMBINATIONS )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${ARRAY_TASK_ID} is outside valid range 0-$((NUM_COMBINATIONS - 1))"
  exit 1
fi

LOCATION_INDEX=$((ARRAY_TASK_ID / NUM_EVENT_TYPES))
EVENT_TYPE_INDEX=$((ARRAY_TASK_ID % NUM_EVENT_TYPES))
LOCATION="${LOCATIONS[$LOCATION_INDEX]}"
EVENT_TYPE="${EVENT_TYPES[$EVENT_TYPE_INDEX]}"

INSTANCES=()
for SEED in "${SEEDS[@]}"; do
  INSTANCES+=("${LOCATION}_${EVENT_TYPE}_5p0std_seed${SEED}")
done

RUN_NAME="${LOCATION}_${EVENT_TYPE,,}_4week"
WARM_START_DIR="$OUTPUT_ROOT/${RUN_NAME}"

run_4week() {
  echo "[$(date)] Running: $*"
  "$ENV_PY" "$@"
}

echo "[$(date)] Array task ${ARRAY_TASK_ID}/${NUM_COMBINATIONS}: LOCATION=${LOCATION}, EVENT_TYPE=${EVENT_TYPE}"

# Baseline run. This saves the per-week variable files used as warm starts.
run_4week v3/src/experiments/run_4week_instance.py \
  --instances "${INSTANCES[@]}" \
  --run-name "$RUN_NAME" \
  --output-root "$OUTPUT_ROOT" \
  --work-limit "$WORK_LIMIT" \
  --warm-start-dir "$WARM_START_DIR" \
  --warm-start-after "$WARM_START_AFTER" \
  --gurobi-output "$GUROBI_OUTPUT"

for LEADERS_FAIRNESS_TYPE in day; do
  # Weekly workload fairness + weekly leader fairness.
  run_4week v3/src/experiments/run_4week_instance.py \
    --instances "${INSTANCES[@]}" \
    --run-name "$RUN_NAME" \
    --output-root "$OUTPUT_ROOT" \
    --work-limit "$WORK_LIMIT" \
    --warm-start-dir "$WARM_START_DIR" \
    --warm-start-after "$WARM_START_AFTER" \
    --gurobi-output "$GUROBI_OUTPUT" \
    --include-weekly-fairness-penalty-hours \
    --include-weekly-fairness-penalty-leaders \
    --leaders-fairness-type "$LEADERS_FAIRNESS_TYPE" \
    --workload-penalty-weight "$WORKLOAD_PENALTY_WEIGHT" \
    --leaders-penalty-weight "$LEADERS_PENALTY_WEIGHT"

  # Weekly workload fairness + running leader fairness.
  run_4week v3/src/experiments/run_4week_instance.py \
    --instances "${INSTANCES[@]}" \
    --run-name "$RUN_NAME" \
    --output-root "$OUTPUT_ROOT" \
    --work-limit "$WORK_LIMIT" \
    --warm-start-dir "$WARM_START_DIR" \
    --warm-start-after "$WARM_START_AFTER" \
    --gurobi-output "$GUROBI_OUTPUT" \
    --include-weekly-fairness-penalty-hours \
    --include-running-fairness-penalty \
    --leaders-fairness-type "$LEADERS_FAIRNESS_TYPE" \
    --workload-penalty-weight "$WORKLOAD_PENALTY_WEIGHT" \
    --leaders-penalty-weight "$LEADERS_PENALTY_WEIGHT"
done

# Weekly workload fairness only. Leader fairness type is irrelevant here.
run_4week v3/src/experiments/run_4week_instance.py \
  --instances "${INSTANCES[@]}" \
  --run-name "$RUN_NAME" \
  --output-root "$OUTPUT_ROOT" \
  --work-limit "$WORK_LIMIT" \
  --warm-start-dir "$WARM_START_DIR" \
  --warm-start-after "$WARM_START_AFTER" \
  --gurobi-output "$GUROBI_OUTPUT" \
  --include-weekly-fairness-penalty-hours \
  --workload-penalty-weight "$WORKLOAD_PENALTY_WEIGHT" \
  --leaders-penalty-weight "$LEADERS_PENALTY_WEIGHT"

echo "[$(date)] Finished combination: LOCATION=${LOCATION}, EVENT_TYPE=${EVENT_TYPE}"

echo "[$(date)] Finished 4-week experiments"
