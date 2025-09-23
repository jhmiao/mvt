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

#SBATCH --mail-type=START,END,FAIL

#SBATCH --mail-user=miaojing@usc.edu

# Load modules as needed, e.g. Python and Gurobi (adjust per CARC module names)
module purge
module load conda
module load gurobi
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mvt
conda install -c gurobi gurobipy

# Choose scratch working directory
WORKDIR=${SCRATCH:-/scratch1/$USER}/mvt_runs_${SLURM_JOB_ID}
mkdir -p "$WORKDIR"

# Move to your project directory where run_carc.py and your code live
cd "$SLURM_SUBMIT_DIR"

# Run with a time limit matching your Slurm wallclock, minus buffer (e.g., 95%)
# For a 12-hour Slurm time, 11h 20m = 40800 seconds.
python run_carc.py --time-limit 3600 --workdir "$WORKDIR" --seed 19 --cap 25
