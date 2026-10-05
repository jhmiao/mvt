#!/bin/bash

#SBATCH --account=ssuen_1733
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --exclusive
#SBATCH --job-name=v3_add_depot
#SBATCH --output=/project2/ssuen_1733/output/v3_add_depot-%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=miaojing@usc.eduå

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
OUTPUT_ROOT="$PROJECT_ROOT/outputs/weeks"

mkdir -p "$OUTPUT_ROOT"

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
echo "[$(date)] Starting add-depot 4-week experiments on $(hostname)"

TIME_LIMIT=600
WORKLOAD_PENALTY_WEIGHT=1
LEADERS_PENALTY_WEIGHT=15
GUROBI_OUTPUT="${GUROBI_OUTPUT:-0}"
DEPOT_GUROBI_OUTPUT="${DEPOT_GUROBI_OUTPUT:-0}"

LOCATIONS=(c101 c201 r101 rc101)
EVENT_TYPES=(Even Random2)
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

run_4week() {
  echo "[$(date)] Running: $*"
  "$ENV_PY" "$@"
}

echo "[$(date)] Array task ${ARRAY_TASK_ID}/${NUM_COMBINATIONS}: LOCATION=${LOCATION}, EVENT_TYPE=${EVENT_TYPE}"

# Matrix:
# - First no-depot solve optimizes either travel only, or travel + weekly workload fairness.
# - Depot solve optimizes either delta travel only, delta travel + weekly leader fairness,
#   or delta travel + running leader fairness.

# First: travel only. Depot: delta travel only.
run_4week v3/src/experiments/run_4week_add_depot.py \
  --instances "${INSTANCES[@]}" \
  --run-name "$RUN_NAME" \
  --time-limit "$TIME_LIMIT" \
  --gurobi-output "$GUROBI_OUTPUT" \
  --depot-gurobi-output "$DEPOT_GUROBI_OUTPUT"

# First: travel + weekly workload fairness. Depot: delta travel only.
run_4week v3/src/experiments/run_4week_add_depot.py \
  --instances "${INSTANCES[@]}" \
  --run-name "$RUN_NAME" \
  --time-limit "$TIME_LIMIT" \
  --gurobi-output "$GUROBI_OUTPUT" \
  --depot-gurobi-output "$DEPOT_GUROBI_OUTPUT" \
  --include-weekly-fairness-penalty-hours \
  --workload-penalty-weight "$WORKLOAD_PENALTY_WEIGHT"

# First: travel only. Depot: delta travel + weekly leader fairness.
run_4week v3/src/experiments/run_4week_add_depot.py \
  --instances "${INSTANCES[@]}" \
  --run-name "$RUN_NAME" \
  --time-limit "$TIME_LIMIT" \
  --gurobi-output "$GUROBI_OUTPUT" \
  --depot-gurobi-output "$DEPOT_GUROBI_OUTPUT" \
  --include-weekly-fairness-penalty-leaders \
  --leaders-penalty-weight "$LEADERS_PENALTY_WEIGHT"

# First: travel + weekly workload fairness. Depot: delta travel + weekly leader fairness.
run_4week v3/src/experiments/run_4week_add_depot.py \
  --instances "${INSTANCES[@]}" \
  --run-name "$RUN_NAME" \
  --time-limit "$TIME_LIMIT" \
  --gurobi-output "$GUROBI_OUTPUT" \
  --depot-gurobi-output "$DEPOT_GUROBI_OUTPUT" \
  --include-weekly-fairness-penalty-hours \
  --include-weekly-fairness-penalty-leaders \
  --workload-penalty-weight "$WORKLOAD_PENALTY_WEIGHT" \
  --leaders-penalty-weight "$LEADERS_PENALTY_WEIGHT"

# First: travel only. Depot: delta travel + running leader fairness.
run_4week v3/src/experiments/run_4week_add_depot.py \
  --instances "${INSTANCES[@]}" \
  --run-name "$RUN_NAME" \
  --time-limit "$TIME_LIMIT" \
  --gurobi-output "$GUROBI_OUTPUT" \
  --depot-gurobi-output "$DEPOT_GUROBI_OUTPUT" \
  --include-running-fairness-penalty \
  --leaders-penalty-weight "$LEADERS_PENALTY_WEIGHT"

# First: travel + weekly workload fairness. Depot: delta travel + running leader fairness.
run_4week v3/src/experiments/run_4week_add_depot.py \
  --instances "${INSTANCES[@]}" \
  --run-name "$RUN_NAME" \
  --time-limit "$TIME_LIMIT" \
  --gurobi-output "$GUROBI_OUTPUT" \
  --depot-gurobi-output "$DEPOT_GUROBI_OUTPUT" \
  --include-weekly-fairness-penalty-hours \
  --include-running-fairness-penalty \
  --workload-penalty-weight "$WORKLOAD_PENALTY_WEIGHT" \
  --leaders-penalty-weight "$LEADERS_PENALTY_WEIGHT"

echo "[$(date)] Finished add-depot 4-week experiments"
