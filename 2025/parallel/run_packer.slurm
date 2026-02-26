#!/bin/bash
#SBATCH --job-name=packer_demo
#SBATCH --output=packer_%j.log
#SBATCH --error=packer_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32   # Adjust based on your HPC node (e.g., 16, 32, 64)
#SBATCH --time=00:10:00      # 10 minutes limit
#SBATCH --partition=standard # Adjust to your cluster's partition name

# Load modules if needed
# module load gcc

# Set OpenMP environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=true
export OMP_PLACES=cores

echo "Starting Packing Job on $(hostname)"
echo "Threads: $OMP_NUM_THREADS"

# Run the Packer (parallel mode)
./packer parallel $SLURM_CPUS_PER_TASK

echo "Job Complete."
