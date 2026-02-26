#!/bin/bash
set -euo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
	# shellcheck disable=SC1091
	source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
	module purge >/dev/null 2>&1 || true
	module load gcc  >/dev/null 2>&1 || true
fi

WORKDIR="${WORKDIR:-$(pwd)}"
cd "$WORKDIR"

make -s all

N=100
TRIALS="${TRIALS:-8}"
SEED="${SEED:-12345}"
PREFIX="${PREFIX:-N100_run}"

./bin/solver "${N}" "${TRIALS}" "${PREFIX}" "${SEED}" "${N}"
