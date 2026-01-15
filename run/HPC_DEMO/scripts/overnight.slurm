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
gcc -O3 -std=c11 -Wall -Wextra -pedantic \
  -march=x86-64 -mtune=generic \
  src/HPC_parallel.c -o bin/HPC_parallel -lm

echo "Binary: $(readlink -f bin/HPC_parallel)"
echo "ldd:"
ldd bin/HPC_parallel | sed -n '1,120p' || true
echo "==============="

# ---------- sweep parameters ----------
N="${SLURM_ARRAY_TASK_ID}"
SEED=12345

# Unique prefix prevents collisions across the array
OUTPREFIX="N${N}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"

# Goal: bisection always finishes; polish is capped and will not dominate walltime.
# NOTE: In your code, --time_limit applies ONLY to polish (bracket + bisect ignore it).
# So we keep bracket/bisect trials small to bound total runtime.
#
# For a 30-min allocation:
# - Bracket trials: small
# - Bisect trials:  small
# - Polish: hard cap (e.g., 12 minutes) with periodic checkpoints
TRIALS_BRACKET=2
TRIALS_BISECT=1
TRIALS_POLISH=2

POLISH_TIME_LIMIT_SEC=720          # 12 minutes polish cap
CHECKPOINT_EVERY_SEC=180           # 3 minute checkpoints during polish

# DEMO mode: reduces A/B iterations and default trials internally; we still override trials explicitly.
DEMO_FLAG="--demo"

echo "N: ${N}"
echo "Seed: ${SEED}"
echo "Prefix: ${OUTPREFIX}"
echo "Trials: bracket=${TRIALS_BRACKET} bisect=${TRIALS_BISECT} polish=${TRIALS_POLISH}"
echo "Polish cap: time_limit=${POLISH_TIME_LIMIT_SEC}s checkpoint_every=${CHECKPOINT_EVERY_SEC}s"
echo "==============="

# ---------- run ----------
# Use srun to bind correctly on some systems; harmless otherwise.
srun ./bin/HPC_parallel "${N}" "${SEED}" \
  --run_id "${N}" \
  --out_prefix "${OUTPREFIX}" \
  ${DEMO_FLAG} \
  --trials_bracket "${TRIALS_BRACKET}" \
  --trials_bisect  "${TRIALS_BISECT}" \
  --trials_polish  "${TRIALS_POLISH}" \
  --time_limit "${POLISH_TIME_LIMIT_SEC}" \
  --checkpoint_every "${CHECKPOINT_EVERY_SEC}"

echo "===== DONE ====="
echo "Time: $(date)"

# ---------- artifact summary ----------
NNN="$(printf "%03d" "$N")"
echo "Artifacts (if present):"
ls -lh "img/${OUTPREFIX}_best_N${NNN}.svg" 2>/dev/null || true
ls -lh "csv/${OUTPREFIX}_best_polys_N${NNN}.csv" 2>/dev/null || true
ls -lh "img/${OUTPREFIX}_checkpoint_N${NNN}.svg" 2>/dev/null || true
ls -lh "csv/${OUTPREFIX}_checkpoint_N${NNN}.csv" 2>/dev/null || true
