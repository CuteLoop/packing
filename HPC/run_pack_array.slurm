#!/bin/bash
#SBATCH --job-name=treepack
#SBATCH --output=logs/treepack_%A_%a.out
#SBATCH --error=logs/treepack_%A_%a.err
#SBATCH --array=1-200%20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:25:00
#SBATCH --mem=8G

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
echo "Binary:  $(readlink -f bin/HPC_parallel || true)"
echo "Prefix:  ${OUTPREFIX}"
echo "==============="

bin/HPC_parallel "${N}" 12345 \
  --run_id "${N}" \
  --time_limit 1200 \
  --checkpoint_every 120 \
  --out_prefix "${OUTPREFIX}"

echo "===== DONE ====="
echo "Time: $(date)"
