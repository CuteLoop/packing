#!/bin/bash
#SBATCH --job-name=sweep_small
#SBATCH --account=ece569
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --output=logs/sweep_small_%j.out
#SBATCH --error=logs/sweep_small_%j.err

# Sweep: N=2,5,10,20,50 — all 3 methods, 2 seeds each.
# Runs sequentially within the job (10 threads per run).
# Budget per run follows Spec Section 12.1.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs out

# Load compiler
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
    module purge >/dev/null 2>&1 || true
    module load gcc  >/dev/null 2>&1 || true
fi

# Build on compute node
make -s all

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

SOLVER=./bin/solver
R=10
METHODS="ms erms pt"
SEEDS="42 99"

# Budget per N (seconds) — from Spec Section 12.1
budget_for_n() {
    case $1 in
        2|5|10) echo 120 ;;   # 2 min
        20)     echo 180 ;;   # 3 min
        50)     echo 360 ;;   # 6 min
        *)      echo 120 ;;
    esac
}

echo "===== SWEEP SMALL: N=2,5,10,20,50 ====="
echo "Start: $(date)"
echo "Host:  $(hostname)"
echo "Threads: $OMP_NUM_THREADS"
echo ""

for N in 2 5 10 20 50; do
    BUDGET=$(budget_for_n $N)
    for METHOD in $METHODS; do
        for SEED in $SEEDS; do
            PREFIX="out/sweep_${METHOD}_N${N}_s${SEED}"
            echo "--- N=$N method=$METHOD seed=$SEED budget=${BUDGET}s ---"
            echo "  Start: $(date)"

            $SOLVER --study --method "$METHOD" --R "$R" --N "$N" \
                --time_budget_sec "$BUDGET" --seed "$SEED" --run_id 0 \
                --out_prefix "$PREFIX" --mode graph --save_best || {
                echo "  WARNING: solver exited with code $?"
            }

            echo "  End: $(date)"
        done
    done
done

echo ""
echo "===== SWEEP SMALL COMPLETE ====="
echo "End: $(date)"
echo "Outputs:"
ls -lh out/sweep_* 2>/dev/null | head -60
