#!/bin/bash
#SBATCH --job-name=old_one_node
#SBATCH --account=ece569
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/old_one_node_%j.out
#SBATCH --error=logs/old_one_node_%j.err

# One-node embarrassingly-parallel launcher for hpc_parallel.c
# Runs many independent seeds concurrently on a single node.

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKDIR"

mkdir -p csv img logs csv/history img/history

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
RUNS="${RUNS:-64}"                  # number of independent seeds
BASE_SEED="${BASE_SEED:-12345}"
TIME_LIMIT="${TIME_LIMIT:-14100}"   # seconds per run
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-600}"
if [ -z "${WORKERS:-}" ]; then
    DETECTED_CPUS="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-}}"
    if [ -z "${DETECTED_CPUS}" ] || [ "${DETECTED_CPUS}" -lt 1 ]; then
        DETECTED_CPUS=$(nproc)
    fi
    WORKERS=$((DETECTED_CPUS - 2))
else
    DETECTED_CPUS="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-unknown}}"
fi

if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi
echo "Detected CPUs:   ${DETECTED_CPUS}"
echo "OS reserve:      2"
if [ "$RUNS" -lt 1 ]; then RUNS=1; fi

echo "===== ONE-NODE PARALLEL CONFIG ====="
echo "Node:            $(hostname)"
echo "N:               ${N}"
echo "RUNS:            ${RUNS}"
echo "WORKERS:         ${WORKERS}"
echo "BASE_SEED:       ${BASE_SEED}"
echo "TIME_LIMIT:      ${TIME_LIMIT}"
echo "CHECKPOINT_EVERY:${CHECKPOINT_EVERY}"
echo "===================================="

run_one() {
    local idx="$1"
    local seed=$((BASE_SEED + idx))
    local prefix="N${N}_job${SLURM_JOB_ID}_w${idx}"

    ./HPC_parallel_old "$N" "$seed" \
      --run_id "$idx" \
      --out_prefix "$prefix" \
      --time_limit "$TIME_LIMIT" \
      --checkpoint_every "$CHECKPOINT_EVERY"
}

export -f run_one
export N BASE_SEED SLURM_JOB_ID TIME_LIMIT CHECKPOINT_EVERY

# Launch RUNS independent seeds with a max of WORKERS concurrent processes.
seq 1 "$RUNS" | xargs -I{} -P "$WORKERS" bash -lc 'run_one "$@"' _ {}

echo "===== DONE ====="
echo "Time: $(date)"

echo "Recent history logs:"
ls -1t logs/N${N}_job${SLURM_JOB_ID}_w*_history_log.csv 2>/dev/null | head -n 10 || true
