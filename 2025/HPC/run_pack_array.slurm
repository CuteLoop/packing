#!/bin/bash
#SBATCH --job-name=treepack
#SBATCH --output=logs/treepack_%A_%a.out
#SBATCH --error=logs/treepack_%A_%a.err
#SBATCH --array=1-200%20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:25:00
#SBATCH --mem=8G

set -euo pipefail
ulimit -s unlimited

module purge
module load gcc

# Always run from the directory where the script lives (packing/HPC)
cd "$(dirname "$0")"

mkdir -p logs csv img summary

# Pick the binary robustly
BIN=""
if [[ -x "bin/HPC_parallel" ]]; then
  BIN="bin/HPC_parallel"
elif [[ -x "./HPC_parallel" ]]; then
  BIN="./HPC_parallel"
elif [[ -x "./hpc_parallel" ]]; then
  BIN="./hpc_parallel"
else
  echo "ERROR: cannot find executable (bin/HPC_parallel or ./HPC_parallel or ./hpc_parallel)"
  echo "PWD=$(pwd)"
  ls -la
  exit 1
fi

N="${SLURM_ARRAY_TASK_ID}"
OUTPREFIX="N${N}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"

echo "===== START ====="
echo "Time:    $(date)"
echo "Host:    $(hostname)"
echo "Job:     ${SLURM_JOB_ID}"
echo "Task:    ${SLURM_ARRAY_TASK_ID}"
echo "N:       ${N}"
echo "Binary:  $(readlink -f "${BIN}" || echo "${BIN}")"
echo "Prefix:  ${OUTPREFIX}"
echo "==============="

# Cap solver to 20 minutes; Slurm walltime gives cushion (25 min)
"${BIN}" "${N}" 12345 \
  --run_id "${N}" \
  --time_limit 1200 \
  --checkpoint_every 120 \
  --out_prefix "${OUTPREFIX}"

echo "===== DONE ====="
echo "Time: $(date)"
