#!/bin/bash
set -euo pipefail

# modules may not exist in some shells
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load gcc  || true
fi

mkdir -p bin logs csv img

# Safer for heterogeneous clusters than -march=native
gcc -O3 -std=c11 -Wall -Wextra -pedantic \
  -march=x86-64 -mtune=generic \
  src/HPC_parallel.c -o bin/HPC_parallel -lm

echo "Built: $(readlink -f bin/HPC_parallel)"

# Quick sanity: show what runtime libs it will use
echo "ldd:"
ldd bin/HPC_parallel | sed -n '1,30p' || true
