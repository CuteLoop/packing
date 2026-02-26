#!/bin/bash
#SBATCH --job-name=treepack_demo_1_10
#SBATCH --output=logs/demo_%A_%a.out
#SBATCH --error=logs/demo_%A_%a.err
#SBATCH --array=1-10%10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
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

# Make 'module' available if the site uses Environment Modules/Lmod
if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

# Load gcc if modules exist; otherwise assume gcc is already available
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load gcc  >/dev/null 2>&1 || true
fi

echo "Building on compute node..."
make -s clean >/dev/null 2>&1 || true
make -s all

echo "Binary:  $(readlink -f bin/solver)"
echo "ldd:"
ldd bin/HPC_parallel | sed -n '1,120p' || true
echo "==============="

N="${SLURM_ARRAY_TASK_ID}"
OUTPREFIX="DEMO_N${N}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
TRIALS="${TRIALS:-3}"
SEED=12345

echo "N:       ${N}"
echo "Prefix:  ${OUTPREFIX}"
echo "Running: demo mode, low trials (quick best artifacts)"
echo "==============="

./bin/solver "${N}" "${TRIALS}" "${OUTPREFIX}" "${SEED}" "${N}"

echo "===== DONE ====="
echo "Time: $(date)"
echo "Artifacts (matching this task):"
ls -lh csv img | egrep "${OUTPREFIX}|^csv|^img" | sed -n '1,200p' || true

echo "Latest files (for quick sanity):"
ls -lt img | head -n 20 || true
ls -lt csv | head -n 20 || true
