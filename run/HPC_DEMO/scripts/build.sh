#!/bin/bash
set -euo pipefail

# modules may not exist in some shells
if command -v module >/dev/null 2>&1; then
  module purge || true
  module load gcc  || true
fi

mkdir -p bin logs csv img

make -s all

echo "Built: $(readlink -f bin/solver)"

# Quick sanity: show what runtime libs it will use
echo "ldd:"
ldd bin/solver | sed -n '1,30p' || true
