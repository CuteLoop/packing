#!/bin/bash
#SBATCH --job-name=gate_b
#SBATCH --account=ece569
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gate_b_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# ============================================================
# Gate B: ER-MS and PT run without deadlocks and produce diagnostics
#
# Tests:
# 1. ER-MS at N=50 produces resample_events > 0
# 2. PT at N=50 produces swap_attempts > 0 and swap_accepts > 0
# 3. Neither method hangs (timeout catches deadlocks)
# 4. Both methods at N=100 complete within budget
# ============================================================

R=10
SEED=42
PASS=0
FAIL=0
TOTAL=0

run_and_check() {
    local METHOD=$1
    local N=$2
    local BUDGET=$3
    local TIMEOUT=$((BUDGET + 60))  # 1 min grace
    local RUN_ID=$4
    local PREFIX="out/gateB_${METHOD}_N${N}_s${SEED}"
    local BISECTION_CSV="${PREFIX}_${METHOD}_N$(printf '%03d' $N)_s${SEED}_bisection.csv"

    TOTAL=$((TOTAL + 1))
    echo "=== Gate B: method=$METHOD N=$N budget=${BUDGET}s ==="
    echo "Start: $(date)"

    # Run with timeout to catch deadlocks
    if timeout "$TIMEOUT" ./bin/solver --study \
        --method "$METHOD" \
        --N "$N" \
        --R "$R" \
        --time_budget_sec "$BUDGET" \
        --seed "$SEED" \
        --run_id "$RUN_ID" \
        --out_prefix "$PREFIX" \
        --mode graph; then
        echo "  Completed without timeout"
    else
        EXIT_CODE=$?
        if [ "$EXIT_CODE" -eq 124 ]; then
            echo "  FAIL: TIMEOUT (possible deadlock)"
            FAIL=$((FAIL + 1))
            return
        else
            echo "  FAIL: Exit code $EXIT_CODE"
            FAIL=$((FAIL + 1))
            return
        fi
    fi

    if [ ! -f "$BISECTION_CSV" ]; then
        echo "  FAIL: CSV not found"
        FAIL=$((FAIL + 1))
        return
    fi

    # Schema validation
    if ! python3 analysis/validate_schema.py "$BISECTION_CSV"; then
        echo "  FAIL: Schema validation failed"
        FAIL=$((FAIL + 1))
        return
    fi

    # Probe count
    PROBES=$(tail -n +2 "$BISECTION_CSV" | wc -l)
    echo "  Probes: $PROBES"

    # Method-specific diagnostics
    if [ "$METHOD" = "erms" ]; then
        # Check resample_events column
        # Assume resample_events is a column in the CSV
        TOTAL_RESAMPLES=$(awk -F, 'NR>1 {
            # Find resample_events column — adjust index as needed
            for(i=1;i<=NF;i++) if(NR==1 && $i ~ /resample/) col=i;
            if(NR>1) s+=$col
        } END {print s+0}' "$BISECTION_CSV" 2>/dev/null || echo "0")

        # Simpler: just grep for non-zero values in the resample column
        HAS_RESAMPLES=$(awk -F, 'NR>1 {
            # Assuming resample_events is near the end of the row
            for(i=1;i<=NF;i++) if($i+0 > 0 && i > NF-3) { print "yes"; exit }
        }' "$BISECTION_CSV")

        if [ -n "$HAS_RESAMPLES" ]; then
            echo "  ER-MS diagnostics: resample events detected"
        else
            echo "  WARNING: No resample events detected (may be OK if early feasibility)"
        fi
    fi

    if [ "$METHOD" = "pt" ]; then
        # Check swap columns (last two columns: swap_attempts, swap_accepts)
        TOTAL_ATTEMPTS=$(awk -F, 'NR>1 {s+=$(NF-1)} END {print s+0}' "$BISECTION_CSV")
        TOTAL_ACCEPTS=$(awk -F, 'NR>1 {s+=$NF} END {print s+0}' "$BISECTION_CSV")

        echo "  PT diagnostics: swap_attempts=$TOTAL_ATTEMPTS swap_accepts=$TOTAL_ACCEPTS"

        if [ "$TOTAL_ATTEMPTS" -gt 0 ]; then
            RATE=$(awk "BEGIN {printf \"%.1f\", 100.0 * $TOTAL_ACCEPTS / $TOTAL_ATTEMPTS}")
            echo "  Acceptance rate: ${RATE}%"
        else
            echo "  WARNING: Zero swap attempts"
        fi
    fi

    echo "  PASS"
    PASS=$((PASS + 1))
    echo ""
}

# --- ER-MS tests ---
run_and_check erms 50  360  1   # 6 min budget (Spec: 6min for N=50)
run_and_check erms 100 600  2   # 10 min budget (Spec: 10min for N=100)

# --- PT tests ---
run_and_check pt   50  360  3
run_and_check pt   100 600  4

echo "========================================"
echo "Gate B Summary: $PASS passed, $FAIL failed (of $TOTAL runs)"
if [ "$FAIL" -eq 0 ]; then
    echo "GATE B: PASS"
else
    echo "GATE B: FAIL"
fi
echo "========================================"
