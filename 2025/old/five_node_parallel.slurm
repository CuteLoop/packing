#!/bin/bash
#SBATCH --job-name=old_five_node
#SBATCH --account=ece569
#SBATCH --partition=standard
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/old_five_node_%j.out
#SBATCH --error=logs/old_five_node_%j.err

# Five-node embarrassingly-parallel launcher for hpc_parallel.c.
# One controller task runs on each node and launches many independent seeds locally.
# Nodes are requested exclusively so each controller can use all CPUs on its node minus 2.
# Outputs are separated under out/node_XX/ so runs can be grouped by node later.

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKDIR"

mkdir -p logs out

echo "===== PREFLIGHT ====="
echo "HOSTNAME=$(hostname)"
echo "WORKDIR=$WORKDIR"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-none}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-none}"
echo "SLURM_EXPORT_ENV=${SLURM_EXPORT_ENV:-unset}"
echo "PATH=$PATH"
which gcc || true
which /bin/bash || true
echo "====================="

if command -v module >/dev/null 2>&1; then
    module purge >/dev/null 2>&1 || true
    module load gcc >/dev/null 2>&1 || true
fi

echo "Compiling hpc_parallel.c ..."
gcc -O3 -march=native -std=c11 -Wall -Wextra -pedantic \
    hpc_parallel.c -o HPC_parallel_old -lm

echo "Compile OK: $(readlink -f HPC_parallel_old)"

# ---------- tunables (override via sbatch --export=...) ----------
N="${N:-100}"
RUNS_PER_NODE="${RUNS_PER_NODE:-64}"
BASE_SEED="${BASE_SEED:-12345}"
TIME_LIMIT="${TIME_LIMIT:-14100}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-600}"
RESERVE_CPUS="${RESERVE_CPUS:-2}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}}"
OUT_BASE="out/N$(printf '%03d' "$N")/${RUN_TAG}"
mkdir -p "$OUT_BASE"

echo "===== FIVE-NODE PARALLEL CONFIG ====="
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Nodes requested:  ${SLURM_JOB_NUM_NODES}"
echo "N:                ${N}"
echo "RUNS_PER_NODE:    ${RUNS_PER_NODE}"
echo "BASE_SEED:        ${BASE_SEED}"
echo "TIME_LIMIT:       ${TIME_LIMIT}"
echo "CHECKPOINT_EVERY: ${CHECKPOINT_EVERY}"
echo "RESERVE_CPUS:     ${RESERVE_CPUS}"
echo "RUN_TAG:          ${RUN_TAG}"
echo "OUT_BASE:         ${OUT_BASE}"
echo "====================================="

MANIFEST="logs/old_five_node_${SLURM_JOB_ID}_manifest.txt"
{
    echo "date=$(date -Is)"
    echo "hostname=$(hostname)"
    echo "workdir=$WORKDIR"
    echo "slurm_job_id=${SLURM_JOB_ID:-none}"
    echo "slurm_nodes=${SLURM_JOB_NUM_NODES:-none}"
    echo "slurm_ntasks=${SLURM_NTASKS:-none}"
    echo "slurm_export_env=${SLURM_EXPORT_ENV:-unset}"
    echo "N=$N"
    echo "RUNS_PER_NODE=$RUNS_PER_NODE"
    echo "BASE_SEED=$BASE_SEED"
    echo "TIME_LIMIT=$TIME_LIMIT"
    echo "CHECKPOINT_EVERY=$CHECKPOINT_EVERY"
    echo "RESERVE_CPUS=$RESERVE_CPUS"
    echo "RUN_TAG=$RUN_TAG"
    echo "OUT_BASE=$OUT_BASE"
    git rev-parse HEAD 2>/dev/null | sed 's/^/git_head=/' || true
    git diff --stat 2>/dev/null || true
    module list 2>&1 || true
    ulimit -a || true
} > "$MANIFEST"
echo "Wrote manifest: $MANIFEST"

srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 --cpu-bind=none /bin/bash "$WORKDIR/five_node_worker.sh" \
    "$WORKDIR" "$N" "$RUNS_PER_NODE" "$BASE_SEED" "$TIME_LIMIT" "$CHECKPOINT_EVERY" "$RESERVE_CPUS" "$OUT_BASE" "$RUN_TAG"

echo "===== OUTPUT SANITY ====="
echo "OUT_BASE: $OUT_BASE"
if [ -d "$OUT_BASE" ]; then
    FILE_COUNT=$(find "$OUT_BASE" -type f | wc -l | awk '{print $1}')
    NODE_DIRS=$(find "$OUT_BASE" -maxdepth 1 -type d -name 'node_*' | wc -l | awk '{print $1}')
    HIST_COUNT=$(find "$OUT_BASE" -type f -name '*_history_log.csv' | wc -l | awk '{print $1}')
    BEST_COUNT=$(find "$OUT_BASE" -type f -name '*_best_polys_N*.csv' | wc -l | awk '{print $1}')
    echo "Node dirs:      $NODE_DIRS"
    echo "File count:     $FILE_COUNT"
    echo "History logs:   $HIST_COUNT"
    echo "Best CSV files: $BEST_COUNT"
    if [ "$FILE_COUNT" -eq 0 ]; then
        echo "WARNING: OUT_BASE is empty. Check per-task .err logs and worker launch output above." >&2
    fi
else
    echo "WARNING: OUT_BASE directory was not created: $OUT_BASE" >&2
fi

echo "===== ALL NODES DONE ====="
echo "Time: $(date)"
