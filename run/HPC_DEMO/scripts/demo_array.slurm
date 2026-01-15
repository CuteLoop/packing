#!/bin/bash
#SBATCH --job-name=treepack_sweep_demo
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --array=1-10%10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem=2G

set -euo pipefail
ulimit -s unlimited

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs csv img bin

echo "===== START ====="
echo "PWD:     $(pwd)"
echo "Time:    $(date)"
echo "Host:    $(hostname)"
echo "Job:     ${SLURM_JOB_ID}"
echo "Task:    ${SLURM_ARRAY_TASK_ID}"
echo "==============="

# Make 'module' available if the site uses Lmod/Environment Modules
if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load gcc  >/dev/null 2>&1 || true
fi

echo "Building on compute node..."
gcc -O3 -std=c11 -Wall -Wextra -pedantic \
  -march=x86-64 -mtune=generic \
  src/HPC_parallel.c -o bin/HPC_parallel -lm

echo "Binary:  $(readlink -f bin/HPC_parallel)"
echo "ldd:"
ldd bin/HPC_parallel | sed -n '1,80p' || true

N="${SLURM_ARRAY_TASK_ID}"
OUTPREFIX="N${N}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"

echo "N:       ${N}"
echo "Prefix:  ${OUTPREFIX}"
echo "==============="

./bin/HPC_parallel "${N}" 12345 \
  --run_id "${N}" \
  --time_limit 60 \
  --checkpoint_every 15 \
  --out_prefix "${OUTPREFIX}"

echo "===== DONE ====="
echo "Time: $(date)"
echo "Artifacts (top of dirs):"
ls -lh csv img | sed -n '1,200p'
