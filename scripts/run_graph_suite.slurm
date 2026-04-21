#!/bin/bash
#SBATCH --job-name=pack_graph
#SBATCH --account=ece569
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/graph_%j.out
#SBATCH --error=logs/graph_%j.err

# Phase 6 — Graph suite: 2 seeds × 3 methods × N∈{5,10,20,50,100}
# Each seed pair runs concurrently (10 threads each).

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# Ensure we run from the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Create output directories
mkdir -p out logs

# Set OpenMP thread count
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SOLVER=./bin/solver
METHODS="ms erms pt"
N_VALUES="5 10 20 50 100"
BUDGET=3600    # 1 hour per run
R=10

SEED_A=1000
SEED_B=2000

for METHOD in $METHODS; do
    for N in $N_VALUES; do
        echo "=== ${METHOD} N=${N} seed=${SEED_A} & seed=${SEED_B} ==="

        OMP_NUM_THREADS=$R $SOLVER --study --method "$METHOD" --R "$R" --N "$N" \
            --time_budget_sec "$BUDGET" --seed "$SEED_A" --run_id 0 \
            --run_type graph --mode graph --save_best &
        PID_A=$!

        OMP_NUM_THREADS=$R $SOLVER --study --method "$METHOD" --R "$R" --N "$N" \
            --time_budget_sec "$BUDGET" --seed "$SEED_B" --run_id 1 \
            --run_type graph --mode graph --save_best &
        PID_B=$!

        wait $PID_A $PID_B
        echo "--- ${METHOD} N=${N} done ---"
    done
done

echo "=== Graph suite complete ==="
