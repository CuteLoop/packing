#!/bin/bash
#SBATCH --job-name=old_smoke_n005_5n_3m
#SBATCH --account=ece569
#SBATCH --partition=standard
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=16G
#SBATCH --time=00:04:00
#SBATCH --output=logs/old_smoke_n005_5n_3m_%j.out
#SBATCH --error=logs/old_smoke_n005_5n_3m_%j.err

set -euo pipefail

export N=5
export RUNS_PER_NODE="${RUNS_PER_NODE:-8}"
export BASE_SEED="${BASE_SEED:-12345}"
export TIME_LIMIT=180
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-60}"
export RESERVE_CPUS="${RESERVE_CPUS:-2}"

/bin/bash five_node_parallel.slurm
