#!/bin/bash
#SBATCH --job-name=sweep_n200
#SBATCH --account=ece569
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sweep_n200_%j.out
#SBATCH --error=logs/sweep_n200_%j.err

# Sweep: N=200 — all 3 methods, 1 seed each.
# Uses R=20 (full node) and 50 min per method.
# Total: 3 × 50 min = 150 min + overhead ≈ 3 hours.

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
N=200
R=20
BUDGET=3000   # 50 minutes per method
METHODS="ms erms pt"
SEED=42

echo "===== SWEEP N=200 ====="
echo "Start: $(date)"
echo "Host:  $(hostname)"
echo "Threads: $OMP_NUM_THREADS"
echo ""

for METHOD in $METHODS; do
    PREFIX="out/sweep_${METHOD}_N${N}_s${SEED}"
    echo "--- N=$N method=$METHOD seed=$SEED budget=${BUDGET}s ---"
    echo "  Start: $(date)"

    $SOLVER --study --method "$METHOD" --R "$R" --N "$N" \
        --time_budget_sec "$BUDGET" --seed "$SEED" --run_id 0 \
        --out_prefix "$PREFIX" --mode hero --save_best || {
        echo "  WARNING: solver exited with code $?"
    }

    echo "  End: $(date)"
done

echo ""
echo "===== SWEEP N=200 COMPLETE ====="
echo "End: $(date)"
echo "Outputs:"
ls -lh out/sweep_*N200* 2>/dev/null
