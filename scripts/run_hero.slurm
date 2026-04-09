#!/bin/bash
#SBATCH --job-name=pack_hero
#SBATCH --account=ece569
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/hero_%j.out
#SBATCH --error=logs/hero_%j.err

# Phase 6 — Hero run: N=200, R=20, single job using polish mode.

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# Ensure we run from the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Create output directories
mkdir -p out logs

# Set OpenMP thread count
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SOLVER=./bin/solver
N=200
R=20
SEED=42
BUDGET=14400   # 4 hours

METHOD=erms    # best method from graph suite (update after Phase 7)

PREFIX="out/hero_${METHOD}_N${N}_s${SEED}"

echo "=== Hero run: ${METHOD} N=${N} R=${R} budget=${BUDGET}s ==="

$SOLVER --study --method "$METHOD" --R "$R" --N "$N" \
    --time_budget_sec "$BUDGET" --seed "$SEED" --run_id 0 \
    --out_prefix "$PREFIX" --mode hero --save_best

echo "=== Hero run complete ==="

# Generate analysis
python3 analysis/aggregate.py "${PREFIX}_${METHOD}_N$(printf '%03d' $N)_s${SEED}_bisection.csv"
