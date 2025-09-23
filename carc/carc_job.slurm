#!/bin/bash

#SBATCH --account=ssuen_1733

#SBATCH --partition=main

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=64

#SBATCH --mem=0

#SBATCH --time=1:30:00

#SBATCH --job-name=gurobi_all_parallel

#SBATCH --output=/scratch1/miaojing/output/gurobi_all_parallel-%j.out

#SBATCH --mail-type=END,FAIL

#SBATCH --mail-user=miaojing@usc.edu


module purge
module load conda
module load gurobi
source "$(conda info --base)/etc/profile.d/conda.sh"

# paths
mkdir -p /scratch1/$USER/output
WORKDIR=${SCRATCH:-/scratch1/$USER}/mvt_runs_${SLURM_JOB_ID}
mkdir -p "$WORKDIR"


ENV_PY="/home1/miaojing/.conda/envs/mvt/bin/python"
which conda
"$ENV_PY" -V
"$ENV_PY" -c "import pandas, numpy, gurobipy; print('pandas', pandas.__version__, 'numpy', numpy.__version__, 'gurobi', gurobipy.gurobi.version())"

cd "$SLURM_SUBMIT_DIR"
"$ENV_PY" run_carc.py --time-limit 3600 --workdir "$WORKDIR" --seed 19 --cap 25

