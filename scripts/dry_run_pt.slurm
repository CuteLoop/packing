#!/bin/bash
#SBATCH --job-name=pack_dry_run
#SBATCH --account=ece569
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/dry_run_pt_%j.out
#SBATCH --error=logs/dry_run_pt_%j.err

# Phase 6.a — 5-minute dry run: PT method, N=20, R=20
# Designed for backfill scheduling under heavy cluster load.

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# Ensure we run from the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Create output directories
mkdir -p logs out

# Set OpenMP thread count
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

N=20
R=20
SEED=12345
BUDGET=280     # 280s SA budget, ~20s headroom within 5-min wall clock
METHOD=pt
RUN_ID=1

echo "=== Dry run: ${METHOD} N=${N} R=${R} seed=${SEED} budget=${BUDGET}s ==="
echo "  Threads: $OMP_NUM_THREADS  Host: $(hostname)  Date: $(date)"

./bin/solver --study --method "$METHOD" --mode graph --R "$R" --N "$N" \
    --time_budget_sec "$BUDGET" --seed "$SEED" --run_id "$RUN_ID" \
    --run_type smoke --save_best

echo "=== Dry run PT complete ==="
