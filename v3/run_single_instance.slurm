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
PROJECT_ROOT=/project2/ssuen_1733/mvt/v3
OUTPUT_ROOT=/project2/ssuen_1733/output/v3

mkdir -p "$OUTPUT_ROOT"

# Optional: per-job working directory
WORKDIR=/project2/ssuen_1733/mvt_runs_${SLURM_JOB_ID}
mkdir -p "$WORKDIR"

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
cd "$PROJECT_ROOT"
echo "[$(date)] Starting run_single_instance on $(hostname)"

# Select instance/type; override via env vars if desired
INSTANCE="${INSTANCE:-c101}"
EVENT_TYPE="${EVENT_TYPE:-Even}"
WORK_LIMIT="${WORK_LIMIT:-150}"   # set empty to disable
TIME_LIMIT="${TIME_LIMIT:-}"      # seconds; leave empty to disable
GUROBI_OUTPUT="${GUROBI_OUTPUT:-0}"

EXTRA_ARGS=()
if [ -n "$TIME_LIMIT" ]; then
  EXTRA_ARGS+=(--time-limit "$TIME_LIMIT")
fi
if [ -n "$WORK_LIMIT" ]; then
  EXTRA_ARGS+=(--work-limit "$WORK_LIMIT")
fi

"$ENV_PY" -m src.experiments.run_single_instance \
  --project-root "$PROJECT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --instance "$INSTANCE" \
  --event-type "$EVENT_TYPE" \
  --gurobi-output "$GUROBI_OUTPUT" \
  "${EXTRA_ARGS[@]}"

echo "[$(date)] Finished run_single_instance"
