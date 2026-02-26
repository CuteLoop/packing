#!/bin/bash
#SBATCH --job-name=chimera_sweep
#SBATCH --output=logs/sweep_%a.out
#SBATCH --error=logs/sweep_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:p100:1
#SBATCH --time=04:00:00
#SBATCH --array=0-19

CHUNK_SIZE=10
OUTPUT_DIR="results"

mkdir -p "$OUTPUT_DIR"

START_N=$(( ($SLURM_ARRAY_TASK_ID * $CHUNK_SIZE) + 1 ))
END_N=$(( $START_N + $CHUNK_SIZE - 1 ))

OUTPUT_FILE="${OUTPUT_DIR}/data_${START_N}_${END_N}.csv"

echo ">> SLURM_JOB_ID: $SLURM_JOB_ID"
echo ">> ARRAY_TASK: $SLURM_ARRAY_TASK_ID"
echo ">> RUNNING: N=$START_N to $END_N"
echo ">> OUTPUT: $OUTPUT_FILE"

./chimera_run $START_N $END_N $OUTPUT_FILE
