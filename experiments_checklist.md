# Experiments Checklist

## Phase 1: Setup and Calibration (The Gates)
- [x] Solver compiled on HPC (`bash scripts/build.sh`)
- [x] PT Pilot calibration, Gate A, Gate B, Gate C submitted (`bash scripts/submit_gates.sh`)
- [x] All gates passed (`bash scripts/check_gates.sh`)
- [x] PT swap acceptance rate verified in [15%, 60%]

## Phase 2: Graph Suite (Method Comparison)
- [x] Full graph suite submitted (`sbatch scripts/run_graph_suite.slurm`)
- [x] All runs completed (check `out/N*/graph_*` for outputs)
- [x] Results analyzed (`python3 scripts/analyze_comparison.py`)
- [ ] Plots and summary tables generated in `analysis/comparison/`

## Phase 3: Hero Run ($N=200$)
- [ ] Winning method selected and set in `scripts/run_hero.slurm`
- [ ] Hero run submitted (`sbatch scripts/run_hero.slurm`)
- [ ] Final metrics and packing efficiency calculated
- [ ] Submission CSV built (`python3 scripts/build_submission.py`)

---

**Notes:**
- Phase 2 (graph suite) is complete as of [date].
- Outputs for each run are in `out/N{NNN}/graph_{method}_{budget}/`.
- See `experiment_sweep.md` for detailed run/analysis instructions.
