#!/bin/bash

#SBATCH --account=ssuen_1733
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --job-name=daily_heur_perm
#SBATCH --output=/project2/ssuen_1733/output/daily_heur_perm-%A.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=miaojing@usc.edu
#SBATCH --array=0-3   # 4 instances → one node each

# ---- Load environment ----
module purge
module load conda
module load gurobi
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---- Paths ----
mkdir -p /project2/ssuen_1733/output
WORKDIR_BASE=/project2/ssuen_1733/mvt_runs_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$WORKDIR_BASE"

# ---- Python environment ----
ENV_PY="/home1/miaojing/.conda/envs/mvt/bin/python"
"$ENV_PY" -V

# ---- Instance list ----
INSTANCES=(c101-a c201-a r101-a rc101-a)
INSTANCE=${INSTANCES[$SLURM_ARRAY_TASK_ID]}

cd "$SLURM_SUBMIT_DIR"
echo "[$(date)] Starting all permutations for instance=$INSTANCE on node $(hostname)"
echo "Workdir base: $WORKDIR_BASE"
echo "---------------------------------------"

# ---- Generate all 120 permutations and loop ----
for PERM_IDX in $(seq 0 119); do
    PERM=$( "$ENV_PY" - <<'EOF'
import itertools, sys
perms = list(itertools.permutations(range(5)))
idx = int(sys.argv[1])
print(",".join(map(str, perms[idx])))
EOF
$PERM_IDX )

    echo "[$(date)] Running permutation $PERM_IDX with order=[$PERM]"

    # Create a unique workdir for this permutation
    WORKDIR="$WORKDIR_BASE/perm_${PERM_IDX}"
    mkdir -p "$WORKDIR"

    # Run the daily heuristic with custom order
    "$ENV_PY" run_daily_heur_custom_order.py \
        --instances "$INSTANCE" \
        --custom_day_order "$PERM" \
        --worklimit 1000 \
        --threads ${SLURM_CPUS_PER_TASK} \
        --seed 20 \
        --workdir "$WORKDIR" || echo "[$(date)] Permutation $PERM_IDX failed, continuing..."

    echo "[$(date)] Finished permutation $PERM_IDX for $INSTANCE"
    echo "---------------------------------------"

done

echo "[$(date)] All 120 permutations complete for instance=$INSTANCE"
