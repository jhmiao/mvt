#!/bin/bash

#SBATCH --account=ssuen_1733
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=v3_add_depot
#SBATCH --output=/project2/ssuen_1733/output/v3_add_depot-%j.out
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

INSTANCES=(
  c101_Even_5p0std_seed42
  c101_Even_5p0std_seed43
  c101_Even_5p0std_seed44
  c101_Even_5p0std_seed45
)

RUN_NAME=c101_even_4week
TIME_LIMIT=300
WORKLOAD_PENALTY_WEIGHT=1
LEADERS_PENALTY_WEIGHT=10
GUROBI_OUTPUT="${GUROBI_OUTPUT:-0}"
DEPOT_GUROBI_OUTPUT="${DEPOT_GUROBI_OUTPUT:-0}"

run_4week() {
  echo "[$(date)] Running: $*"
  "$ENV_PY" "$@"
}

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
