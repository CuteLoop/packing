cat > build.sh <<'EOF'
#!/bin/bash
set -euo pipefail

module purge
module load gcc

# If your cluster needs a newer GCC, adjust the module name (e.g., gcc/12)
# module load gcc/12

mkdir -p bin

# IMPORTANT: ensure HPC_parallel.c begins with valid C, no stray English lines.
gcc -O3 -march=native -std=c11 -Wall -Wextra -pedantic \
  HPC_parallel.c -o bin/HPC_parallel -lm

echo "Built: bin/HPC_parallel"
EOF

chmod +x build.sh
