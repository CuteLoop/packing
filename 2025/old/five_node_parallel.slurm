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

echo "===== FIVE-NODE PARALLEL CONFIG ====="
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Nodes requested:  ${SLURM_JOB_NUM_NODES}"
echo "N:                ${N}"
echo "RUNS_PER_NODE:    ${RUNS_PER_NODE}"
echo "BASE_SEED:        ${BASE_SEED}"
echo "TIME_LIMIT:       ${TIME_LIMIT}"
echo "CHECKPOINT_EVERY: ${CHECKPOINT_EVERY}"
echo "RESERVE_CPUS:     ${RESERVE_CPUS}"
echo "====================================="

export WORKDIR N RUNS_PER_NODE BASE_SEED TIME_LIMIT CHECKPOINT_EVERY RESERVE_CPUS

srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 bash -lc '
set -euo pipefail

cd "$WORKDIR"
NODE_ID="${SLURM_NODEID}"
HOST_SHORT="$(hostname -s 2>/dev/null || hostname)"
NODE_TAG="$(printf "node_%02d" "$NODE_ID")"
NODE_OUT_DIR="out/${NODE_TAG}"
mkdir -p "$NODE_OUT_DIR/csv" "$NODE_OUT_DIR/img" "$NODE_OUT_DIR/logs" \
         "$NODE_OUT_DIR/csv/history" "$NODE_OUT_DIR/img/history"

DETECTED_CPUS="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-}}"
if [ -z "$DETECTED_CPUS" ] || [ "$DETECTED_CPUS" -lt 1 ]; then
    DETECTED_CPUS=$(nproc)
fi
WORKERS=$((DETECTED_CPUS - RESERVE_CPUS))
if [ "$WORKERS" -lt 1 ]; then WORKERS=1; fi

echo "===== NODE LAUNCH ====="
echo "Node ID:       $NODE_ID"
echo "Hostname:      $HOST_SHORT"
echo "Detected CPUs: $DETECTED_CPUS"
echo "Reserve CPUs:  $RESERVE_CPUS"
echo "Workers:       $WORKERS"
echo "Output dir:    $NODE_OUT_DIR"
echo "======================="

run_one() {
    local local_idx="$1"
    local global_idx=$(( NODE_ID * RUNS_PER_NODE + local_idx ))
    local seed=$(( BASE_SEED + global_idx ))
    local prefix="N${N}_job${SLURM_JOB_ID}_${NODE_TAG}_${HOST_SHORT}_w$(printf "%03d" "$local_idx")"

    (
        cd "$NODE_OUT_DIR"
        "$WORKDIR/HPC_parallel_old" "$N" "$seed" \
            --run_id "$global_idx" \
            --out_prefix "$prefix" \
            --time_limit "$TIME_LIMIT" \
            --checkpoint_every "$CHECKPOINT_EVERY"
    )
}

export -f run_one
export WORKDIR N RUNS_PER_NODE BASE_SEED TIME_LIMIT CHECKPOINT_EVERY SLURM_JOB_ID NODE_ID NODE_TAG HOST_SHORT
seq 1 "$RUNS_PER_NODE" | xargs -I{} -P "$WORKERS" bash -lc "run_one \"\$@\"" _ {}

echo "===== NODE DONE ====="
echo "Node ID:    $NODE_ID"
echo "Hostname:   $HOST_SHORT"
echo "Recent history logs:"
ls -1t "$NODE_OUT_DIR/logs"/*_history_log.csv 2>/dev/null | head -n 10 || true
'

echo "===== ALL NODES DONE ====="
echo "Time: $(date)"
