#!/bin/bash
#SBATCH --job-name=sweep_n100
#SBATCH --account=ece569
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=4G
#SBATCH --time=01:30:00
#SBATCH --output=logs/sweep_n100_%j.out
#SBATCH --error=logs/sweep_n100_%j.err

# Sweep: N=100 — all 3 methods, 2 seeds each.
# Budget: 10 min per run (Spec Section 12.1).
# Total: 3 methods × 2 seeds × 10 min = 60 min + overhead.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs out

if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
    module purge >/dev/null 2>&1 || true
    module load gcc  >/dev/null 2>&1 || true
fi

make -s all

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

SOLVER=./bin/solver
N=100
R=10
BUDGET=600    # 10 minutes
METHODS="ms erms pt"
SEEDS="42 99"

echo "===== SWEEP N=100 ====="
echo "Start: $(date)"
echo "Host:  $(hostname)"
echo "Threads: $OMP_NUM_THREADS"
echo ""

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

echo ""
echo "===== SWEEP N=100 COMPLETE ====="
echo "End: $(date)"
echo "Outputs:"
ls -lh out/sweep_*N100* 2>/dev/null
