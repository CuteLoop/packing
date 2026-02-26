#!/bin/bash
#SBATCH --job-name=chimera_n200
#SBATCH --output=chimera_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:p100:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G

echo "Starting N=200 Run on $(hostname)"
./chimera_run
