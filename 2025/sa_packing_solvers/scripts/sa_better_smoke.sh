#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# SA CirclePack — Better Smoke Run + Auto Plots (prettier)
#
# Parameters:
#   N       = 20
#   L       = 20.0
#   r       = 1.0
#   alpha   = 0.0
#   T0      = 2.0
#   gamma   = 0.99995
#   n_steps = 800000
#   sigma   = 0.30
#   seed    = 123
#
# Outputs go to: ./out/<script_name>/
#   <prefix>_trace.csv
#   <prefix>_best.csv
#   <prefix>_best.svg
#   <prefix>_plots.png
#   <prefix>_dE.png
#   <prefix>_moved.png
#
# Requires:
#   tools/plot_trace.py
#   Python deps: pandas, matplotlib (prefer via .venv)
# -----------------------------------------------------------------------------

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SCRIPT_NAME="$(basename "$0" .sh)"
OUT_DIR="out/${SCRIPT_NAME}"
mkdir -p "$OUT_DIR"

make sa_run

OUT_PREFIX="${OUT_DIR}/${SCRIPT_NAME}"

# Run parameters
N=20
L=20.0
R=1.0
ALPHA=0.0
T0=2.0
GAMMA=0.99995
N_STEPS=800000
SIGMA=0.30
SEED=123

echo "Running SA..."
echo "  OUT_PREFIX=${OUT_PREFIX}"
echo "  N=${N} L=${L} r=${R} alpha=${ALPHA} T0=${T0} gamma=${GAMMA} steps=${N_STEPS} sigma=${SIGMA} seed=${SEED}"
echo

./sa_run \
  "$N" "$L" "$R" "$ALPHA" "$T0" "$GAMMA" "$N_STEPS" "$SIGMA" "$SEED" "$OUT_PREFIX"

TRACE_CSV="${OUT_PREFIX}_trace.csv"
BEST_SVG="${OUT_PREFIX}_best.svg"

echo
echo "Run complete."
echo "  ${TRACE_CSV}"
echo "  ${OUT_PREFIX}_best.csv"
echo "  ${BEST_SVG}"

# Plotting
if [[ ! -f "tools/plot_trace.py" ]]; then
  echo "WARNING: tools/plot_trace.py not found; skipping plots."
  exit 0
fi

PYTHON="${ROOT}/.venv/bin/python"
if [[ -x "$PYTHON" ]]; then
  :
else
  PYTHON="python3"
fi

echo
echo "Generating plots..."
"$PYTHON" tools/plot_trace.py "$TRACE_CSV" --out-prefix "$OUT_PREFIX" --window 2000 --max-points 20000 --bins 160

echo
echo "Plots written:"
echo "  ${OUT_PREFIX}_plots.png"
echo "  ${OUT_PREFIX}_dE.png"
echo "  ${OUT_PREFIX}_moved.png"
echo
echo "View:"
echo "  xdg-open ${BEST_SVG}"
echo "  xdg-open ${OUT_PREFIX}_plots.png"
