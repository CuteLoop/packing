#!/bin/bash
# scripts/test_pt_study_smoke.sh
# Study-mode PT integration smoke test
set -e

BIN=bin/solver
OUTDIR=$(mktemp -d)
CSV="$OUTDIR/pt_bisect.csv"

$BIN --study --method pt --N 6 --R 3 --out_prefix "$OUTDIR/pt" --time_budget_sec 2

CSV_FILE=$(ls "$OUTDIR"/*bisection.csv | head -n1)
if [ ! -f "$CSV_FILE" ]; then
  echo "FAIL: No CSV produced" >&2
  exit 1
fi

grep -q swap_attempts "$CSV_FILE" || { echo "FAIL: swap_attempts column missing" >&2; exit 1; }
grep -q swap_accepts "$CSV_FILE" || { echo "FAIL: swap_accepts column missing" >&2; exit 1; }

awk -F, 'NR>1 { if ($18 < 0 || $19 < 0 || $19 > $18) { print "FAIL: swap_accepts out of bounds"; exit 1 } }' "$CSV_FILE"

echo "PT study-mode integration smoke test passed."
