#!/bin/bash
#SBATCH --job-name=treepack_sum
#SBATCH --output=logs/treepack_summary_%j.out
#SBATCH --error=logs/treepack_summary_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem=1G

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p summary

# You can pass the array job id as $1; if omitted, it will summarize all matching outputs.
JOBID="${1:-}"

OUTCSV="summary/summary_${JOBID:-ALL}.csv"
OUTTXT="summary/summary_${JOBID:-ALL}.txt"

# Header:
echo "N,jobid,L,best_feas,prefix,file" > "${OUTCSV}"

# Collect best outputs. Your C writes a header like:
# # prefix=... run_id=... seed=... L=... best_feas=... N=...
if [[ -n "${JOBID}" ]]; then
  FILES=(csv/*"${JOBID}"*_best_polys_N*.csv)
else
  FILES=(csv/*_best_polys_N*.csv)
fi

# If glob doesn't match, bash keeps literal; guard:
if [[ "${FILES[0]}" == "csv/*"* ]]; then
  echo "No matching best_polys files found." | tee "${OUTTXT}"
  exit 0
fi

for f in "${FILES[@]}"; do
  # Parse the first header line for N, L, best_feas, prefix, run_id (jobid)
  # Example line:
  # # prefix=... run_id=7 seed=12345 L=12.34 best_feas=0 N=100
  hdr="$(head -n 1 "$f" || true)"

  # Extract fields with sed (robust enough for your header)
  prefix="$(echo "$hdr" | sed -n 's/.*prefix=\([^ ]*\).*/\1/p')"
  L="$(echo "$hdr" | sed -n 's/.* L=\([^ ]*\).*/\1/p')"
  feas="$(echo "$hdr" | sed -n 's/.* best_feas=\([^ ]*\).*/\1/p')"
  N="$(echo "$hdr" | sed -n 's/.* N=\([0-9]\+\).*/\1/p')"

  # Best-effort job id extraction from filename if present: N##_job<jobid>_task##
  jobid_from_name="$(basename "$f" | sed -n 's/.*_job\([0-9]\+\)_task.*/\1/p')"
  jobid_final="${jobid_from_name:-${JOBID:-NA}}"

  echo "${N:-NA},${jobid_final},${L:-NA},${feas:-NA},${prefix:-NA},$(basename "$f")" >> "${OUTCSV}"
done

# Sort by N ascending (numeric), keep header
{
  head -n 1 "${OUTCSV}"
  tail -n +2 "${OUTCSV}" | sort -t, -k1,1n
} > "${OUTCSV}.tmp" && mv "${OUTCSV}.tmp" "${OUTCSV}"

# Pretty text table
{
  echo "Summary file: ${OUTCSV}"
  echo
  column -s, -t "${OUTCSV}" | sed -e 's/^/  /'
  echo
  echo "Top-10 smallest L:"
  tail -n +2 "${OUTCSV}" | sort -t, -k3,3g | head -n 10 | column -s, -t | sed -e 's/^/  /'
} | tee "${OUTTXT}"

echo "Wrote:"
echo "  ${OUTCSV}"
echo "  ${OUTTXT}"
