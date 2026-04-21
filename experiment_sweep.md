# Experiment Sweep Guide

This document details how to run the main graph suite and the isolated N=100 sweep experiments, what outputs to expect, and how to locate logs and results for further analysis.

---

## 1. Overview

There are two main batch scripts for the sweep experiments:

- **Full Graph Suite:** `scripts/run_graph_suite.slurm` (N = 5, 10, 20, 50, 100)
- **N=100 Isolated Sweep:** `scripts/sweep_n100.slurm` (N = 100 only)

Both scripts run all three methods (`ms`, `erms`, `pt`) with two seeds per method, using 10 threads per seed (20 CPUs total per node).

---

## 2. Running the Experiments

### A. On the HPC Cluster

1. **Build the solver on the cluster:**
   ```bash
   bash scripts/build.sh
   # or
   make all
   ```

2. **Submit the full graph suite:**
   ```bash
   sbatch scripts/run_graph_suite.slurm
   ```

3. **Submit the N=100 sweep (optional, for focused analysis):**
   ```bash
   sbatch scripts/sweep_n100.slurm
   ```

---

## 3. Output Structure

All outputs are written to the `out/` directory, organized by N, method, and run type. Example structure:

```
out/
  N005/graph_ms_1h/N005_ms_s1000_r0_bisection.csv
  N100/graph_pt_1h/N100_pt_s2000_r1_bisection.csv
  N100/graph_erms_1h/...
  ...
```

Each run directory contains:
- `*_bisection.csv` — Bisection probe log (L_lo, L_hi, L_mid, feasible, energies, timing)
- `*_log.csv` — Periodic best energy/feasibility snapshots
- `*_best_state.csv` — Final best configuration (poses)
- `*_best_state.svg` — SVG visualization

---

## 4. Logs

- **Slurm job logs:**
  - `logs/graph_<jobid>.out` and `logs/graph_<jobid>.err` (full suite)
  - `logs/n100_sweep_<jobid>.out` and `logs/n100_sweep_<jobid>.err` (N=100 sweep)
- **Solver stdout/stderr** is captured in these files for each job.

---

## 5. Expected Results

For each method and seed:
- The solver will run up to 1 hour (3600s) per seed, but may finish early if the bracket tolerance or feasibility is reached.
- Each run produces a set of output files in its respective directory.
- The number of probes, final best L, and convergence time can be found in the CSVs.

**Metrics to extract:**
- Final best feasible L
- Time to first feasible solution
- Final bracket width (L_hi - L_lo)
- Number of probes completed

---

## 6. Further Analysis

- Use `scripts/analyze_comparison.py` to generate plots and summary statistics from the bisection CSVs:
  ```bash
  python3 scripts/analyze_comparison.py --glob "out/N*/graph_*/*_bisection.csv" --outdir analysis/comparison
  ```
- Results and plots will be saved in `analysis/comparison/`.
- For legacy data, see `scripts/analyze_sweep.py` and `scripts/build_submission.py`.

---

## 7. Troubleshooting

- If a job fails, check the corresponding `.err` log in `logs/` for error messages.
- Ensure the solver is built on the cluster (not copied from Windows).
- If output directories are missing, verify that `mkdir -p out logs` is present in the script.

---

## 8. References

- Engineering spec: `docs/DOCS.md`
- Experiment plan: `docs/Experiments.md`
- Output schema: `docs/CSV_SCHEMA.md`
- Analysis scripts: `scripts/analyze_comparison.py`, `scripts/analyze_sweep.py`
