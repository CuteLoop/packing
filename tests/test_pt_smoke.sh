#!/usr/bin/env bash
# tests/test_pt_smoke.sh
# Smoke test for PT correctness at N=5.
# Checks: builds, runs without crash, produces swap attempts,
#         swap_accepts <= swap_attempts, at least one probe completes.
#
# Usage: bash tests/test_pt_smoke.sh
# Exit: 0 = pass, 1 = fail

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PASS=0
FAIL=0
fail() { echo "FAIL: $*"; FAIL=$((FAIL+1)); }
pass() { echo "PASS: $*"; PASS=$((PASS+1)); }

echo "=== PT smoke test: N=5 ==="

# Build with PT_DEBUG so we can parse SUMMARY lines
echo "Building (PT_DEBUG)..."
make -B CFLAGS="-O3 -std=c11 -Wall -Wextra -Iinclude -fopenmp -DPT_DEBUG" \
    > /tmp/pt_smoke_build.log 2>&1 \
    && pass "build" \
    || { fail "build (see /tmp/pt_smoke_build.log)"; exit 1; }

echo "Running PT N=5 R=8 10s..."
OUT=$(./bin/solver --study --method pt --N 5 --R 8 \
    --time_budget_sec 10 --seed 99 --run_id 0 \
    --pt_Tmin 1.0 --pt_Tmax 25.0 --pt_K_epoch 500 \
    2>&1) || { fail "solver crashed"; echo "$OUT"; exit 1; }

pass "no crash"

# At least one SUMMARY line
SUMMARIES=$(echo "$OUT" | grep -c "PT DEBUG SUMMARY" || true)
if [ "$SUMMARIES" -ge 1 ]; then
    pass "found $SUMMARIES PT DEBUG SUMMARY lines"
else
    fail "no PT DEBUG SUMMARY lines found"
fi

# Extract cumulative swap_attempts and swap_accepts from all SUMMARY lines
TOTAL_ATT=$(echo "$OUT" | grep "PT DEBUG SUMMARY" | grep -oP 'attempts=\K[0-9]+' | paste -sd+ | bc 2>/dev/null || echo 0)
TOTAL_ACC=$(echo "$OUT" | grep "PT DEBUG SUMMARY" | grep -oP 'accepts=\K[0-9]+'  | paste -sd+ | bc 2>/dev/null || echo 0)

if [ "$TOTAL_ATT" -gt 0 ]; then
    pass "swap_attempts=$TOTAL_ATT > 0"
else
    fail "swap_attempts=0 (replicas never swapped)"
fi

if [ "$TOTAL_ACC" -le "$TOTAL_ATT" ]; then
    pass "swap_accepts=$TOTAL_ACC <= swap_attempts=$TOTAL_ATT"
else
    fail "swap_accepts=$TOTAL_ACC > swap_attempts=$TOTAL_ATT (impossible)"
fi

# distinctE should be > 1 in a healthy PT run
DISTINCT=$(echo "$OUT" | grep "PT DEBUG SUMMARY" | grep -oP 'distinctE=\K[0-9]+' | sort -n | head -1 || echo 0)
if [ "$DISTINCT" -gt 1 ]; then
    pass "distinctE=$DISTINCT > 1 (replicas are diverse)"
else
    fail "distinctE=$DISTINCT <= 1 (replicas collapsed — check step sizes or diversification)"
fi

# Bisection line exists
if echo "$OUT" | grep -q "Bisection complete"; then
    pass "Bisection completed normally"
else
    fail "No 'Bisection complete' line in output"
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
