#!/bin/bash
#SBATCH --job-name=old_n25_4h
#SBATCH --account=ece569
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/old_n25_%A_%a.out
#SBATCH --error=logs/old_n25_%A_%a.err
#SBATCH --array=1-8%2
#SBATCH --signal=B:TERM@120

# Old monolithic solver (hpc_parallel.c) — N=25, 4-hour wall time
# Array tasks provide seed diversity (8 independent runs)

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$WORKDIR"

mkdir -p csv img logs

echo "===== START ====="
echo "PWD:    $(pwd)"
echo "Time:   $(date)"
echo "Host:   $(hostname)"
echo "Job:    ${SLURM_JOB_ID:-NA}"
echo "Task:   ${SLURM_ARRAY_TASK_ID:-NA}"
echo "================="

# ---------- modules ----------
if command -v module >/dev/null 2>&1; then
    module purge >/dev/null 2>&1 || true
    module load gcc >/dev/null 2>&1 || true
fi

# ---------- compile on compute node ----------
echo "Compiling hpc_parallel.c ..."
gcc -O3 -march=native -std=c11 -Wall -Wextra -pedantic \
    hpc_parallel.c -o HPC_parallel_old -lm
echo "Compile OK: $(readlink -f HPC_parallel_old)"

# ---------- parameters ----------
N=25
SEED=$((12345 + SLURM_ARRAY_TASK_ID))
OUTPREFIX="N${N}_job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}"
WALLTIME_SEC=14400
POSTPROC_CUSHION_SEC="${POSTPROC_CUSHION_SEC:-300}"
TIME_LIMIT="${TIME_LIMIT:-14100}"   # default: 4h minus 5 min
MAX_RUN_SEC=$((WALLTIME_SEC - POSTPROC_CUSHION_SEC))
if [ "$MAX_RUN_SEC" -lt 60 ]; then MAX_RUN_SEC=60; fi
if [ "$TIME_LIMIT" -gt "$MAX_RUN_SEC" ]; then TIME_LIMIT="$MAX_RUN_SEC"; fi
CHECKPOINT_EVERY=600    # save every 10 min

echo "N=${N}  SEED=${SEED}  PREFIX=${OUTPREFIX}"
echo "WALLTIME_SEC=${WALLTIME_SEC}  CUSHION=${POSTPROC_CUSHION_SEC}  TIME_LIMIT=${TIME_LIMIT}"

# ---------- run ----------
T_START=$(date +%s)
./HPC_parallel_old "${N}" "${SEED}" \
    --run_id "${SLURM_ARRAY_TASK_ID}" \
    --out_prefix "${OUTPREFIX}" \
    --time_limit "${TIME_LIMIT}" \
    --checkpoint_every "${CHECKPOINT_EVERY}" \
    --trials_bracket 12 \
    --trials_bisect  10 \
    --trials_polish   8
T_END=$(date +%s)

echo "===== DONE ====="
echo "Time:     $(date)"
ELAPSED=$(( T_END - T_START ))
NNN="$(printf "%03d" "${N}")"
BEST_CSV="csv/${OUTPREFIX}_best_polys_N${NNN}.csv"

echo ""
echo "======= BENCHMARK SUMMARY ======="
echo "N:              ${N}"
echo "Seed:           ${SEED}"
echo "Task:           ${SLURM_ARRAY_TASK_ID}"
echo "Wall time (s):  ${ELAPSED}"
printf "Wall time:      %dh %02dm %02ds\n" $((ELAPSED/3600)) $(( (ELAPSED%3600)/60 )) $((ELAPSED%60))
if [ -f "${BEST_CSV}" ]; then
    BEST_L=$(grep '^#' "${BEST_CSV}" | grep -oE 'L=[0-9.eE+\-]+' | head -1 | cut -d= -f2)
    echo "Best L:         ${BEST_L}"
    python3 -c "N=${N}; area=0.245625; L=float('${BEST_L}'); print(f'Area eff:       {N*area/L**2:.6f}  (1.0 = theoretical min)')" 2>/dev/null || true
else
    echo "Best L:         (solver produced no output CSV)"
fi
TELEM="logs/${OUTPREFIX}_telemetry.csv"
[ -f "${TELEM}" ] && echo "Telemetry log:  ${TELEM}" || true
echo "================================="
echo ""
echo "--- Artifacts ---"
ls -lh "csv/${OUTPREFIX}_best_polys_N${NNN}.csv"    2>/dev/null || echo "  (no best csv)"
ls -lh "img/${OUTPREFIX}_best_N${NNN}.svg"           2>/dev/null || echo "  (no best svg)"
ls -lh "csv/${OUTPREFIX}_checkpoint_N${NNN}.csv"     2>/dev/null || echo "  (no checkpoint csv)"
