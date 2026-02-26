#!/bin/bash
#SBATCH --job-name=gate_c
#SBATCH --account=ece569
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --output=logs/gate_c_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# ============================================================
# Gate C: N=100 10-minute run finds feasibility at least once
#
# Each method gets 10 minutes (Spec Section 12.1: 10min for N=100).
# Success: at least one probe marked feasible in bisection CSV.
# ============================================================

N=100
R=10
BUDGET=600   # 10 minutes
SEEDS=(42 99)
METHODS=(ms erms pt)

PASS=0
FAIL=0
TOTAL=0

for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        RUN_ID="${METHOD}_${SEED}"
        PREFIX="out/gateC_${RUN_ID}_N${N}"
        BISECTION_CSV="${PREFIX}_${METHOD}_N$(printf '%03d' $N)_s${SEED}_bisection.csv"

        echo "=== Gate C: method=$METHOD seed=$SEED N=$N budget=${BUDGET}s ==="
        echo "Start: $(date)"

        TIMEOUT=$((BUDGET + 120))  # 2 min grace for N=100
        if timeout "$TIMEOUT" ./bin/solver --study \
            --method "$METHOD" \
            --N "$N" \
            --R "$R" \
            --time_budget_sec "$BUDGET" \
            --seed "$SEED" \
            --out_prefix "$PREFIX" \
            --mode graph; then
            echo "  Completed"
        else
            EXIT_CODE=$?
            if [ "$EXIT_CODE" -eq 124 ]; then
                echo "  FAIL: TIMEOUT"
                FAIL=$((FAIL + 1))
                continue
            fi
        fi

        if [ ! -f "$BISECTION_CSV" ]; then
            echo "  FAIL: CSV not found at $BISECTION_CSV"
            FAIL=$((FAIL + 1))
            continue
        fi

        # Schema check
        python3 analysis/validate_schema.py "$BISECTION_CSV" || true

        # Check for any feasible probe
        # Assume 'feasible' column exists — look for a '1' in it
        FEASIBLE_COUNT=$(awk -F, '
            NR==1 {
                for(i=1;i<=NF;i++) if($i ~ /feasible/ && $i !~ /feasible_ever/) fcol=i;
                next
            }
            {if($(fcol)+0 == 1) count++}
            END {print count+0}
        ' "$BISECTION_CSV")

        PROBES=$(tail -n +2 "$BISECTION_CSV" | wc -l)

        echo "  Probes: $PROBES"
        echo "  Feasible probes: $FEASIBLE_COUNT"

        # Show final L_best
        L_BEST=$(awk -F, '
            NR==1 {for(i=1;i<=NF;i++) if($i ~ /L_best/) col=i; next}
            END {print $(col)}
        ' "$BISECTION_CSV")
        echo "  Final L_best: $L_BEST"

        if [ "$FEASIBLE_COUNT" -gt 0 ]; then
            echo "  PASS: Found feasibility"
            PASS=$((PASS + 1))
        else
            echo "  FAIL: No feasible probe found in 10 minutes"
            FAIL=$((FAIL + 1))
        fi

        echo ""
    done
done

echo "========================================"
echo "Gate C Summary: $PASS passed, $FAIL failed (of $TOTAL runs)"
if [ "$FAIL" -eq 0 ]; then
    echo "GATE C: PASS"
else
    echo "GATE C: FAIL — check which methods/seeds failed"
fi
echo "========================================"
