#!/bin/bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

N="${N:-3}"
MINUTES="${MINUTES:-5}"
TRIALS="${TRIALS:-9999}"
SEED="${SEED:-12345}"
OUTDIR="${OUTDIR:-callibration}"

NNN=$(printf "%03d" "$N")
ts=$(date +%Y%m%d_%H%M%S)
PREFIX_BASE="${PREFIX_BASE:-callibration_n${N}}"
PREFIX="${PREFIX_BASE}_${ts}"

mkdir -p "$OUTDIR" csv img

make -s all

timeout "${MINUTES}m" ./bin/solver "$N" "$TRIALS" "$PREFIX" "$SEED" "$N" || true

best_csv="csv/${PREFIX}_best_polys_N${NNN}.csv"
ckpt_csv="csv/${PREFIX}_checkpoint_N${NNN}.csv"
best_svg="img/${PREFIX}_best_N${NNN}.svg"
ckpt_svg="img/${PREFIX}_checkpoint_N${NNN}.svg"
log_csv="csv/${PREFIX}_log.csv"

[ -f "$best_csv" ] && mv "$best_csv" "$OUTDIR/"
[ -f "$ckpt_csv" ] && mv "$ckpt_csv" "$OUTDIR/"
[ -f "$best_svg" ] && mv "$best_svg" "$OUTDIR/"
[ -f "$ckpt_svg" ] && mv "$ckpt_svg" "$OUTDIR/"
[ -f "$log_csv" ] && mv "$log_csv" "$OUTDIR/"

echo "Saved outputs in: $OUTDIR"
