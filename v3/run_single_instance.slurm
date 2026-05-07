#!/bin/bash

#SBATCH --account=ssuen_1733
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --job-name=v3_single
#SBATCH --output=/project2/ssuen_1733/output/v3_single-%j.out
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

INSTANCES=(
  c101_Even_5p0std_seed42
  c101_Even_5p0std_seed43
  c101_Even_5p0std_seed44
  c101_Even_5p0std_seed45
)

RUN_NAME=c101_even_4week
TIME_LIMIT=300
WARM_START_AFTER=30
WARM_START_DIR=v3/outputs/weeks/c101_even_4week
WORKLOAD_PENALTY_WEIGHT=1
LEADERS_PENALTY_WEIGHT=10
GUROBI_OUTPUT="${GUROBI_OUTPUT:-0}"

run_4week() {
  echo "[$(date)] Running: $*"
  "$ENV_PY" "$@"
}

# Baseline run. This saves the per-week variable files used as warm starts.
run_4week v3/src/experiments/run_4week_instance.py \
  --instances "${INSTANCES[@]}" \
  --run-name "$RUN_NAME" \
  --time-limit "$TIME_LIMIT" \
  --warm-start-dir "$WARM_START_DIR" \
  --warm-start-after "$WARM_START_AFTER" \
  --gurobi-output "$GUROBI_OUTPUT"

for LEADERS_FAIRNESS_TYPE in count day; do
  # Weekly workload fairness + weekly leader fairness.
  run_4week v3/src/experiments/run_4week_instance.py \
    --instances "${INSTANCES[@]}" \
    --run-name "$RUN_NAME" \
    --time-limit "$TIME_LIMIT" \
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
    --time-limit "$TIME_LIMIT" \
    --warm-start-dir "$WARM_START_DIR" \
    --warm-start-after "$WARM_START_AFTER" \
    --gurobi-output "$GUROBI_OUTPUT" \
    --include-weekly-fairness-penalty-hours \
    --include-running-fairness-penalty \
    --leaders-fairness-type "$LEADERS_FAIRNESS_TYPE" \
    --workload-penalty-weight "$WORKLOAD_PENALTY_WEIGHT" \
    --leaders-penalty-weight "$LEADERS_PENALTY_WEIGHT"

  # Weekly workload fairness only.
  run_4week v3/src/experiments/run_4week_instance.py \
    --instances "${INSTANCES[@]}" \
    --run-name "$RUN_NAME" \
    --time-limit "$TIME_LIMIT" \
    --warm-start-dir "$WARM_START_DIR" \
    --warm-start-after "$WARM_START_AFTER" \
    --gurobi-output "$GUROBI_OUTPUT" \
    --include-weekly-fairness-penalty-hours \
    --leaders-fairness-type "$LEADERS_FAIRNESS_TYPE" \
    --workload-penalty-weight "$WORKLOAD_PENALTY_WEIGHT" \
    --leaders-penalty-weight "$LEADERS_PENALTY_WEIGHT"
done

echo "[$(date)] Finished 4-week experiments"
