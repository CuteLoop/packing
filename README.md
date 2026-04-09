# Non-Convex Polygon Packing

Parallel optimization study for minimizing the side length $L$ of a square containing $N$ identical non-convex polygons. Compares three strategies — **Multi-Start (MS)**, **Elite-Resample Multi-Start (ER-MS)**, and **Parallel Tempering (PT)** — under HPC constraints (≤20 CPU cores, ≤4 hours wall-clock).

## Repository Layout

```
run/HPC_DEMO/           ← Main study solver (the one you want)
  src/                  Source files (main.c, method_ms.c, method_erms.c, method_pt.c, ...)
  include/              Headers (replica.h, methods.h, bisection.h, physics.h, ...)
  tests/                Unit tests (8 test binaries)
  scripts/              Slurm job scripts + build/gate helpers
  analysis/             aggregate.py, validate_schema.py
  out/                  Solver outputs (bisection CSV, log CSV, SVGs)
  csv/                  Legacy single-run results (N=1–200)
  docs/                 CSV_SCHEMA.md

2025/HPC/               Legacy standalone HPC solver (hpc_parallel.c)
2025/parallel/          Experimental GPU/CUDA track (chimera solver)
2025/sa_circlepack/     Minimal circle-packing SA baseline
2025/sa_packing_solvers/ Enhanced SA solvers with bounds/feasibility
2025/sa_pack_shrink/    Circle packing with shrinking heuristic
2025/sa_pack_poly_shrink/ Polygon packing with tree initialization
2025/cmaes/             CMA-ES polygon packing

report/                 LaTeX report (main.tex + chapters/)
docs/                   Engineering spec, roadmap, tracking, inventory
tests/                  Top-level unit tests (AABB, geometry, spatial_hash, IO, utils)
```

## Quick Start

### Build the study solver

```bash
cd run/HPC_DEMO
make            # builds bin/solver
make test       # runs all 8 unit tests
```

### Run locally (demo mode)

```bash
# Pack N=10 polygons, 5 trials
./bin/solver 10 5 my_prefix

# Pack with specific seed
./bin/solver 10 5 my_prefix 42 0
```

### Run in study mode (method comparison)

```bash
# MS method, 4 replicas, N=20, 60-second budget
OMP_NUM_THREADS=4 ./bin/solver --study --method ms --R 4 --N 20 \
    --time_budget_sec 60 --seed 42 --run_id 0 \
    --out_prefix out/test_ms --mode graph --save_best

# ER-MS method
OMP_NUM_THREADS=4 ./bin/solver --study --method erms --R 4 --N 20 \
    --time_budget_sec 60 --seed 42 --run_id 0 \
    --out_prefix out/test_erms --mode graph --save_best

# PT method
OMP_NUM_THREADS=4 ./bin/solver --study --method pt --R 4 --N 20 \
    --time_budget_sec 60 --seed 42 --run_id 0 \
    --out_prefix out/test_pt --mode graph --save_best
```

### Submit on HPC (Slurm)

```bash
cd run/HPC_DEMO

# 1. Run decision gates first
bash scripts/submit_gates.sh
# Wait for completion, then check:
bash scripts/check_gates.sh

# 2. Full graph suite (3 methods × 5 N-values × 2 seeds)
sbatch scripts/run_graph_suite.slurm

# 3. Hero run (N=200, R=20, best method)
sbatch scripts/run_hero.slurm
```

### Analyze results

```bash
cd run/HPC_DEMO
python3 analysis/aggregate.py out/some_prefix_bisection.csv
python3 analysis/validate_schema.py out/some_prefix_bisection.csv
```

## Build legacy solvers (top-level)

```bash
# From repo root — builds all legacy binaries into bin/
make
make test       # unit tests against run/HPC_DEMO/include
make coverage   # requires lcov
make clean
```

Targets: `bin/sa_pack_shrink`, `bin/sa_pack_shrink_poly`, `bin/hpc_parallel`, `bin/cmaes_pack_poly`, `bin/sa_pack`

## Compile the report

```bash
cd report
make            # produces main.pdf (uses latexmk or pdflatex)
make view       # opens the PDF
make distclean  # remove PDF + intermediates
```

Requires `pdflatex` (or `latexmk` for automatic rebuild).

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0A–0A.5 | Inventory + struct dump | ✅ Done |
| 0B | ReplicaState + Workspace | ✅ Done |
| 0C | Refactor SA/physics | ✅ Done |
| 1 | Bisection + warm-start + logging | ✅ Done |
| 2 | Method A — MS | ✅ Done |
| 3 | Method B — ER-MS | ✅ Done |
| 4 | Method C — PT | ✅ Done |
| 5 | Polish (Stochastic Shave) | ✅ Done |
| 6 | Orchestration + analysis | ✅ Done |
| 7 | Decision gates + runs | 🔶 In progress |

**Next:** Submit gate jobs on HPC → graph suite → hero run → report.

See [docs/ROADMAP.md](docs/ROADMAP.md) for full details and [docs/tracking.md](docs/tracking.md) for phase-by-phase completion log.

## Key Files

| File | Purpose |
|------|---------|
| `run/HPC_DEMO/src/main.c` | Entry point (demo + study modes) |
| `run/HPC_DEMO/src/method_ms.c` | Multi-Start slice runner |
| `run/HPC_DEMO/src/method_erms.c` | Elite-Resample Multi-Start |
| `run/HPC_DEMO/src/method_pt.c` | Parallel Tempering |
| `run/HPC_DEMO/src/bisection.c` | Bisection driver on L |
| `run/HPC_DEMO/src/polish.c` | MS-polish + Stochastic Shave |
| `run/HPC_DEMO/src/replica.c` | ReplicaState swap/clone/rebuild |
| `run/HPC_DEMO/src/physics.c` | Penalty evaluation |
| `run/HPC_DEMO/src/spatial_hash.c` | Broadphase collision grid |
| `run/HPC_DEMO/scripts/run_graph_suite.slurm` | Graph suite job |
| `run/HPC_DEMO/scripts/run_hero.slurm` | Hero run job |
| `run/HPC_DEMO/analysis/aggregate.py` | Plot traces + compute η |
| `docs/DOCS.md` | Engineering specification |
