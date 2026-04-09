#!/bin/bash
# ============================================================
# submit_gates.sh — Submit all gate experiments with dependencies
#
# Execution order:
#   1. PT Pilot (standalone, 15min)
#   2. Gate A  (after pilot, 20min)
#   3. Gate B  (after pilot, 30min) — can run parallel with Gate A
#   4. Gate C  (after Gate A + B, 45min)
#
# Usage:
#   cd /path/to/HPC_DEMO
#   bash scripts/submit_gates.sh
#
# After all jobs complete:
#   bash scripts/check_gates.sh    (summary of all results)
# ============================================================

set -euo pipefail
mkdir -p logs out

echo "=== Submitting Gate Experiments ==="
echo ""

# 1. PT Pilot — must run first to validate acceptance rate
PILOT_JOB=$(sbatch --parsable scripts/pt_pilot.slurm)
echo "PT Pilot:  Job $PILOT_JOB submitted (15min, R=10, N=100)"

# 2. Gate A — N=20 bracket shrink (depends on pilot to confirm PT works)
GATE_A_JOB=$(sbatch --parsable --dependency=afterok:$PILOT_JOB scripts/gate_a.slurm)
echo "Gate A:    Job $GATE_A_JOB submitted (depends on $PILOT_JOB)"

# 3. Gate B — ER-MS + PT deadlock check (depends on pilot)
GATE_B_JOB=$(sbatch --parsable --dependency=afterok:$PILOT_JOB scripts/gate_b.slurm)
echo "Gate B:    Job $GATE_B_JOB submitted (depends on $PILOT_JOB)"

# 4. Gate C — N=100 feasibility (depends on A and B passing)
GATE_C_JOB=$(sbatch --parsable --dependency=afterok:$GATE_A_JOB:$GATE_B_JOB scripts/gate_c.slurm)
echo "Gate C:    Job $GATE_C_JOB submitted (depends on $GATE_A_JOB + $GATE_B_JOB)"

echo ""
echo "=== All jobs submitted ==="
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "Cancel all:    scancel $PILOT_JOB $GATE_A_JOB $GATE_B_JOB $GATE_C_JOB"
echo ""
echo "Expected timeline:"
echo "  PT Pilot:  ~15 min"
echo "  Gate A+B:  ~30 min (parallel after pilot)"
echo "  Gate C:    ~45 min (after A+B)"
echo "  Total:     ~90 min wall clock"
echo ""
echo "After completion, check results:"
echo "  bash scripts/check_gates.sh"
