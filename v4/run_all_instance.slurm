#!/bin/bash

#SBATCH --account=ssuen_1733
#SBATCH --partition=main
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=04:10:00
#SBATCH --job-name=v4_exact_fairness
#SBATCH --output=/dev/null
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
PROJECT_ROOT=/project2/ssuen_1733/mvt/v4
OUTPUT_ROOT=/project2/ssuen_1733/output/v4_exact_weekly_fairness_w10_l100_mad
LOG_ROOT="$OUTPUT_ROOT/gurobi_logs"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
exec >"$LOG_ROOT/gurobi-${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.out" 2>&1

# Optional: per-job working directory
# WORKDIR=/project2/ssuen_1733/mvt_runs_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
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
# Select combination for this array task
# -------------------------
INSTANCES=(c101 c201 r101 rc101)
EVENT_TYPES=(Even Skewed1 Skewed2 Random1 Random2)

combo_count=$(( ${#INSTANCES[@]} * ${#EVENT_TYPES[@]} ))
task_id=${SLURM_ARRAY_TASK_ID:-0}

if [ "$task_id" -ge "$combo_count" ]; then
  echo "Task id $task_id exceeds combo count $combo_count; exiting."
  exit 1
fi

inst_idx=$(( task_id / ${#EVENT_TYPES[@]} ))
ev_idx=$(( task_id % ${#EVENT_TYPES[@]} ))

INSTANCE=${INSTANCES[$inst_idx]}
EVENT_TYPE=${EVENT_TYPES[$ev_idx]}

# Allow overrides via env if desired
INSTANCE=${INSTANCE_OVERRIDE:-$INSTANCE}
EVENT_TYPE=${EVENT_TYPE_OVERRIDE:-$EVENT_TYPE}

# -------------------------
# Run
# -------------------------
cd "$PROJECT_ROOT"
echo "[$(date)] Starting run_single_instance on $(hostname) for ${INSTANCE}_${EVENT_TYPE}"

WORK_LIMIT="${WORK_LIMIT:-}"   # set empty to disable
TIME_LIMIT="${TIME_LIMIT:-14400}"  # four hours; leave empty to disable

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
  --workload-penalty-weight 10 \
  --leaders-penalty-weight 100 \
  --leaders-fairness-type day \
  --workload-penalty-aggregation mean_absolute_deviation \
  --leaders-penalty-aggregation mean_absolute_deviation \
  --gurobi-output 1 \
  "${EXTRA_ARGS[@]}"

echo "[$(date)] Finished run_single_instance for ${INSTANCE}_${EVENT_TYPE}"
