#!/bin/bash
set -euo pipefail

# Compile with specific optimizations for compute nodes
# Adjust the compiler module or CC if your cluster requires it
cc -O3 -march=native -mavx2 -mfma hpc_parallel_avx2.c -lm -o hpc_prod

mkdir -p logs results

sbatch <<'EOT'
#!/bin/bash
#SBATCH --job-name=pack200_2h
#SBATCH --array=1-100
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%a.out

./hpc_prod 200 $RANDOM --run_id $SLURM_ARRAY_TASK_ID --out_prefix results/run_$SLURM_ARRAY_TASK_ID --time_limit 7000
EOT
