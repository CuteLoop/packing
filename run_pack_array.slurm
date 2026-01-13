#!/bin/bash
#SBATCH --job-name=treepack
#SBATCH --output=logs/treepack_%A_%a.out
#SBATCH --error=logs/treepack_%A_%a.err
#SBATCH --array=1-200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=8G

# If your site requires partition/account/QOS, uncomment and set:
# #SBATCH --partition=standard
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH --qos=normal

set -euo pipefail
ulimit -s unlimited

module purge
module load gcc

mkdir -p logs csv img

N="${SLURM_ARRAY_TASK_ID}"
OUTPREFIX="N${N}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"

echo "===== START ====="
echo "Time:    $(date)"
echo "Host:    $(hostname)"
echo "Job:     ${SLURM_JOB_ID}"
echo "Task:    ${SLURM_ARRAY_TASK_ID}"
echo "N:       ${N}"
echo "Binary:  $(readlink -f bin/HPC_parallel)"
echo "Prefix:  ${OUTPREFIX}"
echo "==============="

# 4 hours = 14400 seconds
bin/HPC_parallel "${N}" 12345 \
  --run_id "${SLURM_ARRAY_TASK_ID}" \
  --time_limit 14400 \
  --checkpoint_every 600 \
  --out_prefix "${OUTPREFIX}"

echo "===== DONE ====="
echo "Time: $(date)"
