#!/bin/bash
# ============================================================
# check_gates.sh — Parse log files and summarize gate outcomes
#
# Run after all gate jobs complete:
#   bash scripts/check_gates.sh
# ============================================================

set -uo pipefail

echo "============================================"
echo "  Gate Experiment Results Summary"
echo "  $(date)"
echo "============================================"
echo ""

# --- PT Pilot ---
echo "=== PT Pilot ==="
PILOT_LOG=$(ls -t logs/pt_pilot_*.out 2>/dev/null | head -1)
if [ -n "$PILOT_LOG" ]; then
    echo "Log: $PILOT_LOG"
    # Extract acceptance rate
    grep -i "acceptance rate" "$PILOT_LOG" | tail -1 || echo "  No acceptance rate found"
    grep -E "^(OK|WARNING):" "$PILOT_LOG" | tail -1 || true
else
    echo "  No pilot log found"
fi
echo ""

# --- Gate A ---
echo "=== Gate A (N=20 bracket shrink) ==="
GATE_A_LOG=$(ls -t logs/gate_a_*.out 2>/dev/null | head -1)
if [ -n "$GATE_A_LOG" ]; then
    echo "Log: $GATE_A_LOG"
    grep "GATE A:" "$GATE_A_LOG" || echo "  No verdict found"
    grep "Summary:" "$GATE_A_LOG" || true
else
    echo "  No Gate A log found"
fi
echo ""

# --- Gate B ---
echo "=== Gate B (ER-MS + PT deadlock/diagnostics) ==="
GATE_B_LOG=$(ls -t logs/gate_b_*.out 2>/dev/null | head -1)
if [ -n "$GATE_B_LOG" ]; then
    echo "Log: $GATE_B_LOG"
    grep "GATE B:" "$GATE_B_LOG" || echo "  No verdict found"
    grep "Summary:" "$GATE_B_LOG" || true
    # Show PT swap rates
    grep "Acceptance rate:" "$GATE_B_LOG" || true
    grep "resample events" "$GATE_B_LOG" || true
else
    echo "  No Gate B log found"
fi
echo ""

# --- Gate C ---
echo "=== Gate C (N=100 feasibility) ==="
GATE_C_LOG=$(ls -t logs/gate_c_*.out 2>/dev/null | head -1)
if [ -n "$GATE_C_LOG" ]; then
    echo "Log: $GATE_C_LOG"
    grep "GATE C:" "$GATE_C_LOG" || echo "  No verdict found"
    grep "Summary:" "$GATE_C_LOG" || true
    # Show per-method results
    grep -E "(PASS|FAIL):.*feasib" "$GATE_C_LOG" || true
else
    echo "  No Gate C log found"
fi
echo ""

# --- Overall verdict ---
echo "============================================"
OVERALL_PASS=true
for LOG in "$PILOT_LOG" "$GATE_A_LOG" "$GATE_B_LOG" "$GATE_C_LOG"; do
    if [ -z "$LOG" ]; then
        echo "WARNING: Missing log file — not all gates completed"
        OVERALL_PASS=false
        break
    fi
done

if $OVERALL_PASS; then
    A_PASS=$(grep -c "GATE A: PASS" "$GATE_A_LOG" 2>/dev/null || echo "0")
    B_PASS=$(grep -c "GATE B: PASS" "$GATE_B_LOG" 2>/dev/null || echo "0")
    C_PASS=$(grep -c "GATE C: PASS" "$GATE_C_LOG" 2>/dev/null || echo "0")

    if [ "$A_PASS" -gt 0 ] && [ "$B_PASS" -gt 0 ] && [ "$C_PASS" -gt 0 ]; then
        echo "ALL GATES PASSED — ready for production graph suite + hero run"
    else
        echo "SOME GATES FAILED — investigate before proceeding"
    fi
fi
echo "============================================"
