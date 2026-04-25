#!/bin/bash
set -euo pipefail

WORKDIR="${1:-${WORKDIR:-}}"
N="${2:-${N:-}}"
RUNS_PER_NODE="${3:-${RUNS_PER_NODE:-}}"
BASE_SEED="${4:-${BASE_SEED:-}}"
TIME_LIMIT="${5:-${TIME_LIMIT:-}}"
CHECKPOINT_EVERY="${6:-${CHECKPOINT_EVERY:-}}"
RESERVE_CPUS="${7:-${RESERVE_CPUS:-2}}"

if [ -z "$WORKDIR" ] || [ -z "$N" ] || [ -z "$RUNS_PER_NODE" ] || [ -z "$BASE_SEED" ] || [ -z "$TIME_LIMIT" ] || [ -z "$CHECKPOINT_EVERY" ]; then
    echo "ERROR: missing required arguments to five_node_worker.sh" >&2
    echo "usage: five_node_worker.sh WORKDIR N RUNS_PER_NODE BASE_SEED TIME_LIMIT CHECKPOINT_EVERY [RESERVE_CPUS]" >&2
    exit 2
fi

cd "$WORKDIR"
NODE_ID="${SLURM_NODEID}"
HOST_SHORT="$(hostname -s 2>/dev/null || hostname)"
NODE_TAG="$(printf "node_%02d" "$NODE_ID")"
NODE_OUT_DIR="out/${NODE_TAG}"
mkdir -p "$NODE_OUT_DIR/csv" "$NODE_OUT_DIR/img" "$NODE_OUT_DIR/logs" \
         "$NODE_OUT_DIR/csv/history" "$NODE_OUT_DIR/img/history"

DETECTED_CPUS=$(nproc)
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
seq 1 "$RUNS_PER_NODE" | xargs -I{} -P "$WORKERS" /bin/bash -lc 'run_one "$@"' _ {}

echo "===== NODE DONE ====="
echo "Node ID:    $NODE_ID"
echo "Hostname:   $HOST_SHORT"
echo "Recent history logs:"
ls -1t "$NODE_OUT_DIR/logs"/*_history_log.csv 2>/dev/null | head -n 10 || true
