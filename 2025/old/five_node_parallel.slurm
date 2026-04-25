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
    git rev-parse HEAD 2>/dev/null | sed 's/^/git_head=/' || true
    git diff --stat 2>/dev/null || true
    module list 2>&1 || true
    ulimit -a || true
} > "$MANIFEST"
echo "Wrote manifest: $MANIFEST"

srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 --cpu-bind=none /bin/bash "$WORKDIR/five_node_worker.sh" \
    "$WORKDIR" "$N" "$RUNS_PER_NODE" "$BASE_SEED" "$TIME_LIMIT" "$CHECKPOINT_EVERY" "$RESERVE_CPUS"

echo "===== ALL NODES DONE ====="
echo "Time: $(date)"
