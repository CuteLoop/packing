# Experiments.md

## Script Inventory and Experimental Purpose

This document describes every script in `scripts/` — what it runs, why it exists,
what it produces, and where it fits in the overall study pipeline defined in
`docs/DOCS.md` (Engineering Spec v1.3).

---

## Experiment Pipeline Overview

```
build.sh
   └─ compile solver

callibration_run.sh          ← legacy local dev smoke test
run_100.sh                   ← legacy local dev single run

submit_gates.sh              ← orchestrate the gate sequence
   ├─ pt_pilot.slurm         [1] PT acceptance-rate calibration
   ├─ gate_a.slurm           [2] N=20 bracket shrink (all methods)
   ├─ gate_b.slurm           [3] ER-MS + PT deadlock / diagnostics
   └─ gate_c.slurm           [4] N=100 feasibility (all methods)

check_gates.sh               ← parse logs and emit go/no-go verdict

dry_run_erms.slurm           ← 5-min smoke test: ER-MS N=20
dry_run_pt.slurm             ← 5-min smoke test: PT N=20

sweep_small.slurm            ← graph suite: N=2,5,10,20,50 (all methods, 2 seeds)
sweep_n100.slurm             ← graph suite: N=100 (all methods, 2 seeds)
sweep_n200.slurm             ← graph suite: N=200 (all methods, 1 seed)
run_graph_suite.slurm        ← full graph suite: N=5,10,20,50,100 (concurrent seeds)

run_hero.slurm               ← hero run: N=200, best method, bisection + polish

analyze_sweep.py             ← analysis: old HPC sweep CSVs → summary.csv + plots
analyze_comparison.py        ← analysis: bisection CSVs → method comparison plots
build_submission.py          ← post-processing: assemble Kaggle submission.csv
```

---

## Output Path Convention

All study-mode runs use the structured output path auto-built by the solver:

```
out/N{NNN}/{run_type}_{method}_{budget}/{NNN}_{method}_s{seed}_r{run_id}
```

Each run produces four files in that directory:

| File | Contents |
|---|---|
| `*_bisection.csv` | One row per probe: L_lo, L_hi, L_mid, feasible, energies, timing |
| `*_log.csv` | Periodic 2-second snapshots of best energy and feasibility |
| `*_best_state.csv` | Final best configuration (poses: cx, cy, theta) |
| `*_best_state.svg` | SVG visualization of the best configuration |

---

## Scripts

---

### `build.sh`

**Purpose:** Compile the solver binary on any machine (login node, compute node, local).

**What it does:**
- Loads GCC module if the HPC module system is present
- Runs `make all`
- Prints the resolved binary path
- Runs `ldd` to verify shared library linkage (catches GLIBC mismatch before submitting jobs)

**When to use:** Run once before any experiment sequence, or at the top of a new login session.

**Produces:** `bin/solver`

---

### `callibration_run.sh`

**Purpose:** Legacy local calibration smoke test for the old (non-study) solver interface.

**What it does:**
- Uses the old `./bin/solver N TRIALS PREFIX SEED RUN_ID` calling convention
- Runs for a configurable number of minutes (`MINUTES`, default 5) with `timeout`
- Moves outputs (CSV, SVG) into a `callibration/` output directory

**Parameters (env vars):**

| Variable | Default | Meaning |
|---|---|---|
| `N` | 3 | Number of polygons |
| `MINUTES` | 5 | Wall-clock limit |
| `TRIALS` | 9999 | Max bisection probes |
| `SEED` | 12345 | RNG seed |
| `OUTDIR` | `callibration` | Where to collect outputs |

**Note:** This script targets the legacy solver interface and does not use `--study` mode
or the structured `out/` hierarchy. It is retained for reference and local debugging only.
New experiments should use the Slurm scripts below.

**Produces:** `callibration/` directory with CSV and SVG files from the run.

---

### `run_100.sh`

**Purpose:** Legacy local one-shot run at N=100 using the old solver interface.

**What it does:**
- Builds the solver
- Invokes `./bin/solver 100 <TRIALS> <PREFIX> <SEED> 100`

**Parameters (env vars):**

| Variable | Default |
|---|---|
| `TRIALS` | 8 |
| `SEED` | 12345 |
| `PREFIX` | `N100_run` |

**Note:** Legacy script. Does not produce structured `out/` output. Useful only for quick
local sanity checks on the old bisection loop before the study-mode refactor.

---

### `submit_gates.sh`

**Purpose:** Submit all gate experiments to Slurm in dependency order and print a
monitoring summary.

**What it does:**
Submits four jobs with `sbatch --dependency=afterok:...` to enforce sequencing:

1. **PT Pilot** — must complete and validate swap acceptance before gates run
2. **Gate A** — depends on Pilot
3. **Gate B** — depends on Pilot (runs concurrently with Gate A)
4. **Gate C** — depends on Gate A **and** Gate B both passing

Prints job IDs, a `squeue` monitor command, a `scancel` all-jobs command, and an
expected timeline.

**Expected wall-clock timeline:**
```
PT Pilot:   ~15 min
Gate A+B:   ~30 min (parallel, after Pilot)
Gate C:     ~45 min (after A and B)
Total:      ~90 min
```

**When to use:** Run once from the repo root after `build.sh`. Do not submit gates
individually — the dependency chain prevents running Gate C before A and B pass.

```bash
bash scripts/submit_gates.sh
```

---

### `check_gates.sh`

**Purpose:** Parse the Slurm log files for all gate jobs and emit a consolidated
pass/fail summary.

**What it does:**
- Finds the most recent log for each of: PT Pilot, Gate A, Gate B, Gate C
- Extracts acceptance rate (PT Pilot), summary lines, and per-method verdicts
- Prints a final overall PASS/FAIL

**When to use:** After all gate jobs have completed (`squeue -u $USER` shows empty).

```bash
bash scripts/check_gates.sh
```

**Does not re-run anything. Read-only.**

---

### `pt_pilot.slurm`

**Purpose:** Calibrate PT temperature ladder before any gate or graph-suite run.
Corresponds to Spec Section 10.3.

**Experiment:**
- Method: PT
- N = 100, R = 10, budget = 10 minutes (600s), seed = 42
- Run type: `pilot`

**Success criterion:** PT swap acceptance rate falls in **[15%, 60%]**.
If outside this range, `PT_DEFAULT_TMAX_RATIO` must be adjusted (×2 up or down)
and this script must be re-run before proceeding.

**Output directory:** `out/N100/pilot_pt_10m/`

**What it produces:**
- Bisection CSV, log CSV
- Printed swap acceptance rate extracted from the bisection CSV

**Gate dependency:** All gate scripts (`gate_a`, `gate_b`, `gate_c`) depend on this
job completing successfully via `submit_gates.sh`.

---

### `gate_a.slurm`

**Purpose:** Verify that all three methods can shrink the bisection bracket by at least
3 probes at N=20. Corresponds to Phase 7, Gate A (Spec Section 7).

**Experiment:**
- N = 20, R = 10, budget = 180s (3 min), seeds = {42, 99}
- Methods: ms, erms, pt
- Run type: `gate_a` (6 runs total)

**Success criterion:** Every method × seed combination completes ≥ 3 bisection probes.

**Output directories:** `out/N020/gate_a_{method}_3m/` (one per method × seed)

**What it checks per run:**
1. Bisection CSV exists
2. Row count ≥ 3 (bracket shrunk by at least 3 probes)
3. Schema validation via `analysis/validate_schema.py`
4. Prints first and final bracket width

**Verdict:** Printed as `GATE A: PASS` or `GATE A: FAIL` at the end of the log.

---

### `gate_b.slurm`

**Purpose:** Verify that ER-MS and PT run without deadlocks and emit correct diagnostic
columns. Corresponds to Phase 7, Gate B.

**Experiment:**

| Method | N | Budget |
|---|---|---|
| erms | 50 | 360s (6 min) |
| erms | 100 | 600s (10 min) |
| pt | 50 | 360s (6 min) |
| pt | 100 | 600s (10 min) |

- R = 10, seed = 42, run type: `gate_b` (4 runs total)
- Each run is wrapped in `timeout (budget + 60s)` to catch deadlocks

**Success criteria:**
- Solver exits cleanly (no timeout, no non-zero exit code)
- ER-MS: `resample_events > 0` detected in bisection CSV
- PT: `swap_attempts > 0` and `swap_accepts > 0`; acceptance rate printed

**Output directories:** `out/N050/gate_b_{method}_6m/`, `out/N100/gate_b_{method}_10m/`

**Verdict:** `GATE B: PASS` or `GATE B: FAIL`.

---

### `gate_c.slurm`

**Purpose:** Verify that at least one method finds a feasible configuration at N=100
within a 10-minute budget. Corresponds to Phase 7, Gate C.

**Experiment:**
- N = 100, R = 10, budget = 600s (10 min), seeds = {42, 99}
- Methods: ms, erms, pt
- Run type: `gate_c` (6 runs total)
- Each run uses `timeout (600 + 120s)` grace period

**Success criterion:** At least one probe is marked `feasible=1` in the bisection CSV
for each method × seed combination.

**Output directories:** `out/N100/gate_c_{method}_10m/`

**What it checks per run:**
1. CSV exists and passes schema validation
2. Counts rows where `feasible == 1`
3. Extracts and prints final `L_best`

**Verdict:** `GATE C: PASS` or `GATE C: FAIL — check which methods/seeds failed`.

**Blocker:** If Gate C fails for all methods, γ(100) or the slice schedule must be
adjusted before submitting the graph suite (see Spec Section 8, 9.1).

---

### `dry_run_erms.slurm`

**Purpose:** 5-minute smoke test for ER-MS to verify the solver compiles and runs on
the compute node before submitting longer jobs.

**Experiment:**
- Method: erms, N = 20, R = 20, budget = 280s, seed = 12345, run_id = 2
- Run type: `smoke`
- Wall-clock limit: 5 minutes (`#SBATCH --time=00:05:00`)

**Output directory:** `out/N020/smoke_erms_4m/`

**When to use:** After a code change or a new cluster session, before submitting gate
jobs. Designed for backfill scheduling (short enough to start quickly under heavy load).

---

### `dry_run_pt.slurm`

**Purpose:** 5-minute smoke test for PT, identical in structure to `dry_run_erms.slurm`.

**Experiment:**
- Method: pt, N = 20, R = 20, budget = 280s, seed = 12345, run_id = 1
- Run type: `smoke`

**Output directory:** `out/N020/smoke_pt_4m/`

---

### `sweep_small.slurm`

**Purpose:** Graph-suite sweep for small N. Produces the convergence data used in
method comparison plots for N ≤ 50.

**Experiment:**

| N | Budget | Methods | Seeds |
|---|---|---|---|
| 2 | 120s (2 min) | ms, erms, pt | 42, 99 |
| 5 | 120s | ms, erms, pt | 42, 99 |
| 10 | 120s | ms, erms, pt | 42, 99 |
| 20 | 180s (3 min) | ms, erms, pt | 42, 99 |
| 50 | 360s (6 min) | ms, erms, pt | 42, 99 |

- R = 10 threads, runs sequentially within the job
- Run type: `smoke` (short graph suite; use `run_graph_suite.slurm` for quality runs)
- Wall-clock limit: 90 minutes

**Output directories:**
```
out/N002/smoke_ms_2m/
out/N005/smoke_erms_2m/
out/N050/smoke_pt_6m/
...  (30 run directories total)
```

**Produces:** 30 run directories, each with bisection CSV + log CSV + optional best-state files.

**Analysis:** Feed bisection CSVs into `scripts/analyze_comparison.py`.

---

### `sweep_n100.slurm`

**Purpose:** Graph-suite sweep for N=100. Produces the convergence data at the largest
comparison-suite problem size.

**Experiment:**
- N = 100, R = 10, budget = 600s (10 min), seeds = {42, 99}
- Methods: ms, erms, pt (6 runs total, sequential)
- Wall-clock limit: 90 minutes

**Note:** This script still uses the old `--out_prefix` convention and has not yet been
updated to `--run_type`. Update `PREFIX=` lines before use if structured output is needed.

**Output directories (old convention):** `out/sweep_{method}_N100_s{seed}/`

---

### `sweep_n200.slurm`

**Purpose:** Graph-suite sweep for N=200 comparing all three methods. Provides the
pre-hero comparative data at the heroic problem size.

**Experiment:**
- N = 200, R = 20, budget = 3000s (50 min per method), seed = 42
- Methods: ms, erms, pt (3 runs total, sequential)
- Mode: `hero` (bisection + polish)
- Wall-clock limit: 4 hours

**Note:** Like `sweep_n100.slurm`, still uses the old `--out_prefix` convention.

**Output directories (old convention):** `out/sweep_{method}_N200_s42/`

---

### `run_graph_suite.slurm`

**Purpose:** Full quality graph suite — the main algorithm comparison experiment.
Runs two seeds concurrently for each N × method combination to use all 20 CPUs.

**Experiment:**

| N | Methods | Seeds | Budget each | Threads each |
|---|---|---|---|---|
| 5 | ms, erms, pt | 1000, 2000 | 3600s (1 hour) | 10 |
| 10 | ms, erms, pt | 1000, 2000 | 3600s | 10 |
| 20 | ms, erms, pt | 1000, 2000 | 3600s | 10 |
| 50 | ms, erms, pt | 1000, 2000 | 3600s | 10 |
| 100 | ms, erms, pt | 1000, 2000 | 3600s | 10 |

- The two seeds for each N × method run in parallel (`&` + `wait`), using 10 + 10 = 20 CPUs
- Run type: `graph`
- Wall-clock limit: 4 hours

**Output directories:**
```
out/N005/graph_ms_1h/N005_ms_s1000_r0_bisection.csv
out/N005/graph_ms_1h/N005_ms_s2000_r1_bisection.csv
out/N100/graph_pt_1h/N100_pt_s1000_r0_bisection.csv
...  (30 run directories total)
```

**Analysis:** Feed all `*_bisection.csv` files into `scripts/analyze_comparison.py`.

**Prerequisite:** All gates must pass before submitting this job.

---

### `run_hero.slurm`

**Purpose:** The N=200 hero run — the primary demonstration result of the study.
Uses bisection followed by MS-polish to find the densest feasible packing at N=200.

**Experiment:**
- Method: `erms` (update to winner after graph suite)
- N = 200, R = 20, budget = 14400s (4 hours), seed = 42
- Mode: `hero` (bisection phase → polish phase)
- Run type: `hero`

**Output directory:** `out/N200/hero_erms_4h/`

**What it produces:**
- `*_bisection.csv` — bracket trace (bisection phase)
- `*_log.csv` — periodic energy snapshots
- `*_best_state.csv` — final best configuration (cx, cy, theta for all N=200 polygons)
- `*_best_state.svg` — SVG visualization of the packing

After the run, `analysis/aggregate.py` is invoked automatically to compute packing
efficiency η = N·A_poly / L².

**Spec reference:** Section 12.2 — 150 min bisection + 50 min polish.

**Prerequisite:** Graph suite complete; best method selected and set in `METHOD=erms` line.

---

### `analyze_comparison.py`

**Purpose:** Parse bisection CSVs from the graph suite and produce method-comparison plots.

**Inputs:** `out/*/graph_*/*_bisection.csv` (or any glob passed via `--glob`)

**Outputs (to `analysis/comparison/`):**
- Best L vs N per method
- Wall-clock runtime vs N
- Packing density η vs N
- Bracket width vs N
- Probes completed vs N

**Usage:**
```bash
python3 scripts/analyze_comparison.py
python3 scripts/analyze_comparison.py --glob "out/N*/graph_*/*_bisection.csv" \
    --outdir analysis/comparison
```

---

### `analyze_sweep.py`

**Purpose:** Parse the old HPC sweep `csv/*_best_polys_N*.csv` files (legacy `csv/`
directory) and produce per-N quality metrics.

**Inputs:** `csv/*_best_polys_N*.csv`

**Outputs (to `analysis/`):**
- `summary.csv` — per-N best L, η, feasibility
- `plots/*.png` — packing quality vs N

**Usage:**
```bash
python3 scripts/analyze_sweep.py
python3 scripts/analyze_sweep.py --csv_glob "csv/*_checkpoint_N*.csv"
```

**Note:** This script targets the old `csv/` output layout from before the structured
`out/` hierarchy was introduced. It remains useful for analyzing the archived HPC sweep
data in `csv/`.

---

### `build_submission.py`

**Purpose:** Assemble a Kaggle-style `submission.csv` from best-configuration CSVs.

**Inputs:** `csv/*_best_polys_N*.csv` (or any directory passed via `--input-dir`)

**Output:** `submission.csv`

**What it does:**
- Reads each `*_best_polys_N*.csv` in the input directory
- Parses the header line for L, N, seed, run_id
- Selects the best (lowest L) configuration per N
- Writes a combined submission file

**Usage:**
```bash
python3 scripts/build_submission.py --input-dir csv --output submission.csv
python3 scripts/build_submission.py --input-dir out/N200/hero_erms_4h --output submission.csv
```

---

## Experiment Execution Checklist

```
[ ] 1.  bash scripts/build.sh                   # compile solver
[ ] 2.  bash scripts/submit_gates.sh            # submit pilot + 3 gates
[ ] 3.  # wait ~90 min
[ ] 4.  bash scripts/check_gates.sh             # confirm all gates PASS
[ ] 5.  # if Gate C fails: adjust gamma/schedule, rebuild, repeat from step 2
[ ] 6.  sbatch scripts/run_graph_suite.slurm    # full graph suite (~4h)
[ ] 7.  # wait ~4h
[ ] 8.  python3 scripts/analyze_comparison.py   # generate comparison plots
[ ] 9.  # choose best method from plots, set METHOD= in run_hero.slurm
[ ] 10. sbatch scripts/run_hero.slurm           # hero run (~4h)
[ ] 11. # wait ~4h
[ ] 12. python3 scripts/build_submission.py     # assemble submission.csv
```

---

## Run-Type Vocabulary

| `--run_type` | Used by | Meaning |
|---|---|---|
| `smoke` | `dry_run_*.slurm`, `sweep_small.slurm` | Short correctness check, ≤ 5 min |
| `gate_a` | `gate_a.slurm` | Formal Gate A verification |
| `gate_b` | `gate_b.slurm` | Formal Gate B verification |
| `gate_c` | `gate_c.slurm` | Formal Gate C verification |
| `pilot` | `pt_pilot.slurm` | PT calibration run |
| `graph` | `run_graph_suite.slurm` | Quality comparison sweep |
| `hero` | `run_hero.slurm` | N=200 demonstration run |
| `dev` | local ad-hoc | Local development, not committed |
