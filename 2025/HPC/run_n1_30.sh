#!/bin/bash
set -euo pipefail

THREADS=${1:-7}
LOGDIR=logs
BIN=./sa_pack_shrink_poly_hpc

mkdir -p "$LOGDIR"
: > "$LOGDIR/times_n1_30.txt"

for N in $(seq 1 30); do
  echo "Running N=$N..."
  /usr/bin/time -f "%e" "$BIN" "$N" --no_polish --threads "$THREADS" 2>> "$LOGDIR/times_n1_30.txt"
done

echo "Finished. Times appended to $LOGDIR/times_n1_30.txt"
