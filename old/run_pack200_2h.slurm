#!/bin/bash
# --------------------
### Directives Section
# --------------------
#SBATCH --job-name=gato_pack200
#SBATCH --account=ece569
#SBATCH --partition=standard
#SBATCH --array=1-100                # Running 100 concurrent seeds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00              # 2-hour limit per job
#SBATCH --output=logs/%a.out         # Real-time output log per seed

# --------------------
### Code Section
# --------------------
if [ ! -x ./hpc_prod ]; then
	cc -O3 -march=native -mavx2 -mfma hpc_parallel_avx2.c -lm -o hpc_prod
fi
mkdir -p logs results
./hpc_prod 200 $RANDOM --run_id $SLURM_ARRAY_TASK_ID --out_prefix results/run_$SLURM_ARRAY_TASK_ID --time_limit 7000
