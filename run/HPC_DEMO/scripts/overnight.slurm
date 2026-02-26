#!/bin/bash
#SBATCH --job-name=treepack_overnight_demo
#SBATCH --output=logs/overnight_%A_%a.out
#SBATCH --error=logs/overnight_%A_%a.err
#SBATCH --array=1-200%20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=2G
#SBATCH --signal=B:TERM@90

set -euo pipefail
ulimit -s unlimited

# ---------- robust working directory ----------
WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKDIR"

mkdir -p logs csv img bin

echo "===== START ====="
echo "PWD:      $(pwd)"
echo "Time:     $(date)"
echo "Host:     $(hostname)"
echo "Job:      ${SLURM_JOB_ID:-NA}"
echo "Task:     ${SLURM_ARRAY_TASK_ID:-NA}"
echo "NodeList: ${SLURM_NODELIST:-NA}"
echo "==============="

# ---------- modules (site-dependent) ----------
if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  module load gcc  >/dev/null 2>&1 || true
fi

# ---------- build on compute node to avoid GLIBC mismatch ----------
echo "Building on compute node (avoids GLIBC mismatch)..."
make -s clean >/dev/null 2>&1 || true
make -s all

echo "Binary: $(readlink -f bin/solver)"
echo "ldd:"
ldd bin/HPC_parallel | sed -n '1,120p' || true
echo "==============="

# ---------- sweep parameters ----------
N="${SLURM_ARRAY_TASK_ID}"
SEED=12345

# Unique prefix prevents collisions across the array
OUTPREFIX="N${N}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"

TRIALS="${TRIALS:-8}"

echo "N: ${N}"
echo "Seed: ${SEED}"
echo "Prefix: ${OUTPREFIX}"
echo "Trials: ${TRIALS}"
echo "==============="

# ---------- run ----------
# Use srun to bind correctly on some systems; harmless otherwise.
srun ./bin/solver "${N}" "${TRIALS}" "${OUTPREFIX}" "${SEED}" "${N}"

echo "===== DONE ====="
echo "Time: $(date)"

# ---------- artifact summary ----------
NNN="$(printf "%03d" "$N")"
echo "Artifacts (if present):"
ls -lh "img/${OUTPREFIX}_best_N${NNN}.svg" 2>/dev/null || true
ls -lh "csv/${OUTPREFIX}_best_polys_N${NNN}.csv" 2>/dev/null || true
ls -lh "img/${OUTPREFIX}_checkpoint_N${NNN}.svg" 2>/dev/null || true
ls -lh "csv/${OUTPREFIX}_checkpoint_N${NNN}.csv" 2>/dev/null || true
