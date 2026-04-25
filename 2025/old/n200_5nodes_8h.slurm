#!/bin/bash
#SBATCH --job-name=old_n200_5n_8h
#SBATCH --account=ece569
#SBATCH --partition=standard
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/old_n200_5n_8h_%j.out
#SBATCH --error=logs/old_n200_5n_8h_%j.err

set -euo pipefail

export N=200
export RUNS_PER_NODE="${RUNS_PER_NODE:-64}"
export BASE_SEED="${BASE_SEED:-12345}"
export TIME_LIMIT=28500
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-600}"
export RESERVE_CPUS="${RESERVE_CPUS:-2}"

/bin/bash five_node_parallel.slurm
