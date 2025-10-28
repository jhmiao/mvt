#!/bin/bash

#SBATCH --account=ssuen_1733

#SBATCH --partition=main

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=64

#SBATCH --mem=0

#SBATCH --time=48:00:00

#SBATCH --job-name=gurobi_all_parallel

#SBATCH --output=/project2/ssuen_1733/output/gurobi_all_parallel-%j.out

#SBATCH --mail-type=END,FAIL

#SBATCH --mail-user=miaojing@usc.edu

#SBATCH --array=0-3


module purge
module load conda
module load gurobi
source "$(conda info --base)/etc/profile.d/conda.sh"

# paths
mkdir -p /project2/$USER/output
WORKDIR=${PROJECT:-/project2/$USER}/mvt_runs_${SLURM_JOB_ID}
mkdir -p "$WORKDIR"


ENV_PY="/home1/miaojing/.conda/envs/mvt/bin/python"
which conda
"$ENV_PY" -V
"$ENV_PY" -c "import pandas, numpy, gurobipy; print('pandas', pandas.__version__, 'numpy', numpy.__version__, 'gurobi', gurobipy.gurobi.version())"

INSTANCES=(c101 c201 r101 rc101)
INSTANCE=${INSTANCES[$SLURM_ARRAY_TASK_ID]}

cd "$SLURM_SUBMIT_DIR"
"$ENV_PY" run_daily_heur.py \
  --overall-seconds 2000 \
  --workdir "$WORKDIR" \
  --seed 20 \
  --grb-seed 20 \
  --threads ${SLURM_CPUS_PER_TASK:-1} \
  --cap 25 \
  --instances $INSTANCE

