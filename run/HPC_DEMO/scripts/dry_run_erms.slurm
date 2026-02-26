#!/bin/bash
#SBATCH --job-name=pack_dry_run
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=4G
#SBATCH --partition=standard
#SBATCH --output=logs/dry_run_erms_%j.out
#SBATCH --error=logs/dry_run_erms_%j.err

# Phase 6.a — 5-minute dry run: ER-MS method, N=20, R=20
# Designed for backfill scheduling under heavy cluster load.

set -euo pipefail
mkdir -p logs out

export OMP_NUM_THREADS=20

N=20
R=20
SEED=12345
BUDGET=280     # 280s SA budget, ~20s headroom within 5-min wall clock
METHOD=erms
RUN_ID=2
PREFIX="out/dry_${METHOD}_N${N}_s${SEED}"

echo "=== Dry run: ${METHOD} N=${N} R=${R} seed=${SEED} budget=${BUDGET}s ==="
echo "  Threads: $OMP_NUM_THREADS  Host: $(hostname)  Date: $(date)"

./bin/solver --study --method "$METHOD" --mode graph --R "$R" --N "$N" \
    --time_budget_sec "$BUDGET" --seed "$SEED" --run_id "$RUN_ID" \
    --out_prefix "$PREFIX" --save_best

echo "=== Dry run ER-MS complete ==="
