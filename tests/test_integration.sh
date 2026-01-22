#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Building cmaes binary..."
make bin/cmaes_pack_poly

mkdir -p csv img

OUTLOG=tests/integration.log
echo "Running short integration: cmaes_pack_poly --N 2 --L 2.0 --evals 100 --seed 1" > "$OUTLOG"
./bin/cmaes_pack_poly --N 2 --L 2.0 --evals 100 --seed 1 >> "$OUTLOG" 2>&1 || true

# check for expected CSV output (pattern: csv/cma_best_N002.csv or csv/cma_best_N2.csv)
FOUND=$(ls csv | grep -E "cma_best_N0*2" | head -n1 || true)
if [ -z "$FOUND" ]; then
    echo "Integration failed: no cmaes CSV output found" >> "$OUTLOG"
    echo "See $OUTLOG"
    exit 1
fi

echo "Integration OK. Output file: csv/$FOUND"
echo "Log: $OUTLOG"
exit 0
