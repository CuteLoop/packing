#!/bin/bash
#SBATCH --job-name=poly200
#SBATCH --array=1-50
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=4G

# Diversified search: vary shrink epsilon and bisect trials per task.
MIN_SHRINKS=(5e-4 1e-3 2e-3 5e-3)
TRIALS_BISECT=(14 18 22 26)

IDX=$((SLURM_ARRAY_TASK_ID % 4))
MIN_SHRINK=${MIN_SHRINKS[$IDX]}
TRIALS=${TRIALS_BISECT[$IDX]}
SEED=$((12345 + SLURM_ARRAY_TASK_ID))

./HPC_parallel 200 "$SEED" \
  --run_id "$SLURM_ARRAY_TASK_ID" \
  --out_prefix "run_${SLURM_ARRAY_TASK_ID}" \
  --min_shrink "$MIN_SHRINK" \
  --max_shrink 2e-2 \
  --target_success 0.05 \
  --trials_bisect "$TRIALS"
