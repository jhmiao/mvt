#!/bin/bash

#SBATCH --account=ssuen_1733

#SBATCH --partition=epyc-64

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=64

#SBATCH --mem=0

#SBATCH --time=12:00:00

#SBATCH --job-name=gurobi_all_parallel

#SBATCH --output=/scratch1/miaojing/gurobi_all_parallel-%j.out

#SBATCH --mail-type=START,END,FAIL

#SBATCH --mail-user=miaojing@usc.edu


# Load any required modules

module load gurobi/11.0.2
module load python/3.11.9
module load parallel


# Directory containing all instance subfolders

BASE_DIR=${1:-"/project/nannicin_1432/steiner_tree_generation/corrected_code/recomputed_instances"}



# Path to the child script that generates the .lp

CHILD_SCRIPT="/project/nannicin_1432/steiner_tree_generation/corrected_code/gurobi_patch_solution.sh"



# Determine concurrency: use SLURM_CPUS_PER_TASK if set, otherwise all available CPUs

MAX_JOBS=${SLURM_CPUS_PER_TASK:-$(getconf _NPROCESSORS_ONLN)}

echo "Will run up to $MAX_JOBS parallel jobs"



# Iterate over each instance directory

for INSTANCE_DIR in "$BASE_DIR"/*/; do

  # If we've reached the concurrency limit, wait for one to finish

  while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do

    wait -n

  done



  echo "Launching: $INSTANCE_DIR"

  "$CHILD_SCRIPT" "$INSTANCE_DIR" &

done



# Wait for any remaining background jobs

wait



echo "All instance submissions complete."
