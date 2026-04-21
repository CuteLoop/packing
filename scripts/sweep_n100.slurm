#!/bin/bash
#SBATCH --job-name=pack_n100
#SBATCH --account=ece569
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/n100_sweep_%j.out
#SBATCH --error=logs/n100_sweep_%j.err

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# Ensure we run from the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Create output directories
mkdir -p out logs

SOLVER=./bin/solver
METHODS="ms erms pt"
N=100
BUDGET=3600    # 1 hour maximum per run
R=10           # 10 threads per seed

SEED_A=1000
SEED_B=2000

        $SOLVER --study --method "$METHOD" --R "$R" --N "$N" \

for METHOD in $METHODS; do
    echo "Starting: ${METHOD} (Max 1 Hour) | Seed ${SEED_A} & Seed ${SEED_B} concurrently..."

    # Launch Seed A in background
    OMP_NUM_THREADS=$R $SOLVER --study --method "$METHOD" --R "$R" --N "$N" \
        --time_budget_sec "$BUDGET" --seed "$SEED_A" --run_id 0 \
        --run_type sweep_n100 --mode graph --save_best &
    PID_A=$!

    # Launch Seed B in background
    OMP_NUM_THREADS=$R $SOLVER --study --method "$METHOD" --R "$R" --N "$N" \
        --time_budget_sec "$BUDGET" --seed "$SEED_B" --run_id 1 \
        --run_type sweep_n100 --mode graph --save_best &
    PID_B=$!

    # Wait for BOTH seeds of the current method to finish (either by tolerance or timeout)
    wait $PID_A $PID_B
    echo "--- ${METHOD} completed (or timed out). Moving to next... ---"
            --time_budget_sec "$BUDGET" --seed "$SEED" --run_id 0 \

            --out_prefix "$PREFIX" --mode graph --save_best || {
            echo "  WARNING: solver exited with code $?"
        }

        echo "  End: $(date)"
    done
done

echo ""
echo "===== SWEEP N=100 COMPLETE ====="
echo "End: $(date)"
echo "Outputs:"
ls -lh out/sweep_*N100* 2>/dev/null
