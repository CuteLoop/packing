#!/bin/bash
#SBATCH --job-name=treepack_demo
#SBATCH --output=logs/demo_%A_%a.out
#SBATCH --error=logs/demo_%A_%a.err
#SBATCH --array=1-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=2G

set -euo pipefail
ulimit -s unlimited

module purge
module load gcc

# CRITICAL: ensure job runs inside the demo directory
cd ~/packing/HPC_DEMO

mkdir -p logs csv img

N="${SLURM_ARRAY_TASK_ID}"
OUTPREFIX="N${N}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"

echo "===== START ====="
echo "PWD:     $(pwd)"
echo "Time:    $(date)"
echo "Host:    $(hostname)"
echo "Job:     ${SLURM_JOB_ID}"
echo "Task:    ${SLURM_ARRAY_TASK_ID}"
echo "N:       ${N}"
echo "Binary:  $(readlink -f bin/HPC_parallel)"
echo "Prefix:  ${OUTPREFIX}"
echo "==============="

# Short run for debugging
./bin/HPC_parallel "${N}" 12345 \
  --run_id "${N}" \
  --time_limit 60 \
  --checkpoint_every 15 \
  --out_prefix "${OUTPREFIX}"

echo "===== DONE ====="
echo "Time: $(date)"

echo "Artifacts:"
ls -lh csv img | sed -n '1,200p'
