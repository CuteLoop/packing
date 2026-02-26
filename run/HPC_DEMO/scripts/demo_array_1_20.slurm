#!/bin/bash
#SBATCH --job-name=treepack_demo
#SBATCH --output=logs/demo_%A_%a.out
#SBATCH --error=logs/demo_%A_%a.err
#SBATCH --array=1-20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=2G

set -euo pipefail
ulimit -s unlimited

# If your site uses modules:
if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load gcc  >/dev/null 2>&1 || true
fi

WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKDIR"

mkdir -p logs csv img

if [ ! -x bin/solver ]; then
  echo "Building on compute node..."
  make -s all
fi

N="${SLURM_ARRAY_TASK_ID}"
OUTPREFIX="N${N}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
TRIALS="${TRIALS:-5}"
SEED=12345

echo "===== START ====="
echo "PWD:     $(pwd)"
echo "Time:    $(date)"
echo "Host:    $(hostname)"
echo "Job:     ${SLURM_JOB_ID}"
echo "Task:    ${SLURM_ARRAY_TASK_ID}"
echo "N:       ${N}"
echo "Binary:  $(readlink -f bin/solver)"
echo "Prefix:  ${OUTPREFIX}"
echo "==============="

./bin/solver "${N}" "${TRIALS}" "${OUTPREFIX}" "${SEED}" "${N}"

echo "===== DONE ====="
echo "Time: $(date)"
echo "Artifacts:"
ls -lh csv img | sed -n '1,200p'