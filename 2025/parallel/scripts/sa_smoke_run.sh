#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# SA CirclePack — Simple Test Run (Smoke Run)
#
# Parameters used in this run:
#
#   N       = 20       (number of circles)
#   L       = 20.0     (container side length)
#   r       = 1.0      (circle radius)
#   alpha   = 0.0      (L-penalty; irrelevant if L is fixed)
#   T0      = 1.0      (initial temperature)
#   gamma   = 0.999    (geometric cooling factor per step)
#   n_steps = 300000   (number of Metropolis steps)
#   sigma   = 0.25     (proposal step std-dev; Gaussian N(0, sigma^2 I))
#   seed    = 123      (deterministic seed)
#
# Output:
#   All files are written to ./out/<script_name_without_.sh>/
#
#   - *_trace.csv  : energy / temperature / acceptance trace
#   - *_best.csv   : best configuration (circle centers)
#   - *_best.svg   : SVG visualization of best configuration
#
# View SVG with:
#   xdg-open out/<script_name>/<script_name>_best.svg
# -----------------------------------------------------------------------------

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Script name without extension
SCRIPT_NAME="$(basename "$0" .sh)"

# Output directory named after script
OUT_DIR="out/${SCRIPT_NAME}"
mkdir -p "$OUT_DIR"

# Build runner
make sa_run

# Output prefix (full path, used by C code)
OUT_PREFIX="${OUT_DIR}/${SCRIPT_NAME}"

# Explicit parameters
N=20
L=20.0
R=1.0
ALPHA=0.0
T0=1.0
GAMMA=0.999
N_STEPS=300000
SIGMA=0.25
SEED=123

./sa_run \
  "$N" \
  "$L" \
  "$R" \
  "$ALPHA" \
  "$T0" \
  "$GAMMA" \
  "$N_STEPS" \
  "$SIGMA" \
  "$SEED" \
  "$OUT_PREFIX"

echo
echo "Run complete."
echo "Output directory:"
echo "  $OUT_DIR"
echo
echo "Files written:"
echo "  ${OUT_PREFIX}_trace.csv"
echo "  ${OUT_PREFIX}_best.csv"
echo "  ${OUT_PREFIX}_best.svg"
