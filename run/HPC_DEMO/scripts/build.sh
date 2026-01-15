#!/bin/bash
set -euo pipefail

module purge
module load gcc

mkdir -p bin

gcc -O3 -march=native -std=c11 -Wall -Wextra -pedantic \
  src/HPC_parallel.c -o bin/HPC_parallel -lm

echo "Built: $(readlink -f bin/HPC_parallel)"
