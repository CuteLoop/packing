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

srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 --cpu-bind=none /bin/bash "$WORKDIR/five_node_worker.sh" \
    "$WORKDIR" "$N" "$RUNS_PER_NODE" "$BASE_SEED" "$TIME_LIMIT" "$CHECKPOINT_EVERY" "$RESERVE_CPUS"

echo "===== ALL NODES DONE ====="
echo "Time: $(date)"
