#!/bin/bash
#SBATCH --job-name=gate_a
#SBATCH --account=ece569
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/gate_a_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# ============================================================
# Gate A: N=20 end-to-end verification
# Each method must shrink the bracket by >= 3 probes.
# Runs are sequential (same 10 cores, one method at a time).
# Budget: 3 minutes per method per seed (Spec: 3min for N=20).
# ============================================================

N=20
R=10
BUDGET=180   # 3 minutes per Spec Section 12.1
SEEDS=(42 99)
METHODS=(ms erms pt)

PASS=0
FAIL=0

for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN_ID="${METHOD}_${SEED}"
        PREFIX="out/gateA_${RUN_ID}_N${N}"

        echo "=== Gate A: method=$METHOD seed=$SEED N=$N ==="
        echo "Start: $(date)"

        ./bin/solver --study \
            --method "$METHOD" \
            --N "$N" \
            --R "$R" \
            --time_budget_sec "$BUDGET" \
            --seed "$SEED" \
            --out_prefix "$PREFIX" \
            --mode graph

        # Find the bisection CSV
        BISECTION_CSV="${PREFIX}_${METHOD}_N$(printf '%03d' $N)_s${SEED}_bisection.csv"

        if [ ! -f "$BISECTION_CSV" ]; then
            echo "FAIL: CSV not found at $BISECTION_CSV"
            FAIL=$((FAIL + 1))
            continue
        fi

        # Count probes
        PROBES=$(tail -n +2 "$BISECTION_CSV" | wc -l)
        echo "  Probes completed: $PROBES"

        if [ "$PROBES" -ge 3 ]; then
            echo "  PASS: >= 3 probes"
            PASS=$((PASS + 1))
        else
            echo "  FAIL: < 3 probes (bracket didn't shrink enough)"
            FAIL=$((FAIL + 1))
        fi

        # Show bracket shrink
        FIRST_WIDTH=$(awk -F, 'NR==2 {print $NF}' "$BISECTION_CSV")
        LAST_WIDTH=$(awk -F, 'END {print $NF}' "$BISECTION_CSV")
        echo "  First bracket width: $FIRST_WIDTH"
        echo "  Final bracket width: $LAST_WIDTH"

        # Schema validation
        python3 analysis/validate_schema.py "$BISECTION_CSV" && echo "  Schema: OK" || echo "  Schema: FAIL"
        echo ""
    done
done

echo "========================================"
echo "Gate A Summary: $PASS passed, $FAIL failed (of $((${#METHODS[@]} * ${#SEEDS[@]})) runs)"
if [ "$FAIL" -eq 0 ]; then
    echo "GATE A: PASS"
else
    echo "GATE A: FAIL"
fi
echo "========================================"
