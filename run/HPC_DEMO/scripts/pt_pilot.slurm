#!/bin/bash
#SBATCH --job-name=pt_pilot
#SBATCH --account=ece569
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/pt_pilot_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true

# ============================================================
# PT Pilot — Engineering Spec Section 10.3
# 10-minute run at N=100, R=10 to validate swap acceptance rate.
# Target: 15% — 60% acceptance.
# If outside range, adjust PT_DEFAULT_TMAX_RATIO and re-run.
# ============================================================

N=100
R=10
METHOD="pt"
BUDGET=600    # 10 minutes
SEED=42
PREFIX="out/pilot_N100_s42"

echo "=== PT Pilot ==="
echo "N=$N  R=$R  budget=${BUDGET}s  seed=$SEED"
echo "Start: $(date)"

./bin/solver --study \
    --method "$METHOD" \
    --N "$N" \
    --R "$R" \
    --time_budget_sec "$BUDGET" \
    --seed "$SEED" \
    --out_prefix "$PREFIX" \
    --mode graph

echo "End: $(date)"
echo ""

# --- Extract and display swap acceptance ---
BISECTION_CSV="${PREFIX}_${METHOD}_N$(printf '%03d' $N)_s${SEED}_bisection.csv"

if [ -f "$BISECTION_CSV" ]; then
    echo "=== Swap Acceptance Summary ==="
    # Header reference: ...,swap_attempts,swap_accepts
    # Sum across all probes
    TOTAL_ATTEMPTS=$(awk -F, 'NR>1 {s+=$(NF-1)} END {print s+0}' "$BISECTION_CSV")
    TOTAL_ACCEPTS=$(awk -F, 'NR>1 {s+=$NF} END {print s+0}' "$BISECTION_CSV")

    if [ "$TOTAL_ATTEMPTS" -gt 0 ]; then
        RATE=$(awk "BEGIN {printf \"%.1f\", 100.0 * $TOTAL_ACCEPTS / $TOTAL_ATTEMPTS}")
        echo "Total swap attempts: $TOTAL_ATTEMPTS"
        echo "Total swap accepts:  $TOTAL_ACCEPTS"
        echo "Acceptance rate:     ${RATE}%"
        echo ""
        # Evaluate against target range
        awk "BEGIN {
            rate = 100.0 * $TOTAL_ACCEPTS / $TOTAL_ATTEMPTS;
            if (rate < 15) print \"WARNING: Rate < 15% — reduce PT_DEFAULT_TMAX_RATIO (try 10)\";
            else if (rate > 60) print \"WARNING: Rate > 60% — increase PT_DEFAULT_TMAX_RATIO (try 50)\";
            else print \"OK: Acceptance rate in target range [15%, 60%]. Lock this ratio.\";
        }"
    else
        echo "WARNING: Zero swap attempts. Check that PT swap logic is running."
    fi

    echo ""
    echo "=== Per-probe breakdown ==="
    echo "probe_idx, swap_attempts, swap_accepts"
    awk -F, 'NR>1 {printf "  probe %s: %s attempts, %s accepts\n", $6, $(NF-1), $NF}' "$BISECTION_CSV"
else
    echo "ERROR: Bisection CSV not found at $BISECTION_CSV"
    echo "Check filename pattern against your CSV_SCHEMA.md"
fi

echo ""
echo "=== PT Pilot Complete ==="
