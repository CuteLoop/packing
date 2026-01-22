# Packing Codebase — Documentation

Purpose
-------
This repository contains C implementations and helpers for packing shapes (circles and a non-convex polygon "ChristmasTree") into a square container using optimization methods (simulated annealing, CMA-ES). It includes HPC-ready variants, visualization, and logging/export to CSV and SVG.

Top-level programs
------------------
- `sa_pack_shrink/sa_pack_shrink.c` — Simulated annealing for circle packing with an outer shrink loop to approximate minimal container side `L`.
- `vanilla_sa/sa_pack.c` — Simulated annealing (simpler) for circle packing; adaptive steps and incremental energy.
- `sa_pack_poly_shrink/sa_pack_shrink_poly.c` — SA for non-convex polygon packing using triangulation + SAT, spatial hashing, and L bracketing/bisection.
- `HPC/hpc_parallel.c` — HPC-oriented polygon-packing variant: deterministic seeding, job-array friendly outputs, robust bracketing.
- `cmaes/cmaes_pack_poly.c` — CMA-ES variant (fixed `L`) optimizing positions and rotations.
- `run/HPC_DEMO/src/main.c` — Refactored demo entry point using modular headers (`include/`) and components (`annealing`, `physics`, `spatial_hash`, `logger`).

Key algorithms & features
-------------------------
- Simulated annealing: adaptive step-size, two-phase schedule (explore → enforce feasibility), multi-start, teleport/reinsert moves.
- Incremental energy updates: neighbor-only recalculation for O(N) per move.
- Spatial hashing / uniform grid and AABB broad-phase to reduce collision checks.
- Triangle-triangle SAT for non-convex polygon collisions (triangulation-based).
- Outer-loop shrink: bracket + bisection on container side `L` to find minimal feasible packing.
- Deterministic RNG (SplitMix64 + xorshift64*) for reproducible HPC runs.

Build examples
--------------
Compile individual programs (examples):

```bash
gcc -O2 -std=c11 -Wall -Wextra -pedantic sa_pack_shrink/sa_pack_shrink.c -o bin/sa_pack_shrink -lm
gcc -O2 -std=c11 -Wall -Wextra -pedantic sa_pack_poly_shrink/sa_pack_shrink_poly.c -o bin/sa_pack_shrink_poly -lm
gcc -O3 -march=native -std=c11 -Wall -Wextra -pedantic HPC/hpc_parallel.c -o bin/hpc_parallel -lm
gcc -O3 -march=native -std=c11 -Wall -Wextra -pedantic cmaes/cmaes_pack_poly.c -o bin/cmaes_pack_poly -lm
``` 

Run examples
------------
Simple runs (defaults produce `csv/` and `img/` outputs):

```bash
./bin/sa_pack_shrink
./bin/sa_pack_shrink_poly
./bin/cmaes_pack_poly --N 7 --L 1.5 --evals 100000 --seed 1
``` 

HPC / SLURM
-----------
- Scripts and job array examples are under `HPC/` (e.g. `run_pack_array.slurm`, `run_n1_30.sh`). The HPC-ready executable (`HPC/hpc_parallel.c`) is designed for deterministic seeds and unique output prefixes.

Outputs
-------
- `csv/` — CSVs listing packed shapes and checkpoints (e.g. `*_best_polys_N###.csv`).
- `img/` — SVG visualizations of packings.
- `logs/` — timing and run logs.

Where to look first
-------------------
- For circle packing: `sa_pack_shrink/sa_pack_shrink.c` and `vanilla_sa/sa_pack.c`.
- For polygon packing: `sa_pack_poly_shrink/sa_pack_shrink_poly.c`, `HPC/hpc_parallel.c`, and `cmaes/cmaes_pack_poly.c`.
- For the refactored demo and modular components: `run/HPC_DEMO/src/` and `run/HPC_DEMO/include/`.

Next steps / notes
------------------
- If you want, I can add a short `Makefile` to `bin/` and wire simple `make` targets for common builds.
- I can also expand this file with examples showing typical command-line flags and sample outputs.

Function-level summaries
------------------------
Below are concise descriptions of the main functions (and logical groups) in each primary source file to help navigate the codebase.

- `sa_pack_shrink/sa_pack_shrink.c`:
	- RNG helpers: `splitmix64`, `rng_seed`, `xorshift64star`, `rng_u01`, `rng_uniform` — deterministic PRNG utilities.
	- State management: `state_alloc`, `state_free` — allocate/free circle state arrays (`x,y,r`).
	- Geometry/energy helpers: `overlap_pair`, `overlap_sum_for_k`, `outside_for_k` — compute pairwise overlap and outside penalties.
	- IO/visualization: `write_circles_csv`, `write_best_svg` — export CSV and SVG results.
	- Utilities and SA core: assorted helpers for proposal moves, acceptance, adaptive step controllers, reinsert/teleport moves, incremental energy updates, and the outer shrink loop — with `main` implementing multi-start + bracket/binary search on `L`.

- `vanilla_sa/sa_pack.c`:
	- Same RNG and basic utilities as `sa_pack_shrink`.
	- `state_alloc`/`state_free`, `overlap_pair`, `overlap_sum_for_k`, `outside_for_k` for circle geometry.
	- CSV/SVG exporters similar to `sa_pack_shrink`.
	- Simulated annealing core and `main` that runs the SA optimization (fixed `L` in this demo).

- `sa_pack_poly_shrink/sa_pack_shrink_poly.c`:
	- Filesystem helper: `ensure_dir`.
	- RNG and angle helpers: `splitmix64`, `rng_seed`, `wrap_angle_0_2pi`, etc.
	- Polygon base geometry: `BASE_V` data, `base_bounding_radius`, `base_polygon_area`.
	- Transform & AABB: `rot_trans`, `build_world_verts`, `aabb_of_verts`, `aabb_of_tri_pts`, `aabb_overlap`.
	- Triangulation & SAT helpers: functions to build triangle AABBs, perform triangle-triangle SAT tests, and higher-level polygon collision checks.
	- Spatial hash / broad-phase: uniform-grid initialization and neighbor iteration to accelerate pair checks.
	- SA core: move proposal (translation + rotation), adaptive step controllers, penalty/ramp schedule, reinsert moves, incremental feasibility/energy updates, and the outer shrink/bracketing + bisection controller; `main` drives runs and file outputs (CSV/SVG).

- `HPC/hpc_parallel.c`:
	- CLI and small utils: `usage`, `streq`, `file_exists`, `ensure_dir`, `now_seconds`.
	- RNG & deterministic seed creation: `splitmix64`, `rng_seed`, `make_trial_seed` — deterministic per-run seeding for job arrays.
	- Geometry and triangulation helpers analogous to `sa_pack_shrink_poly` (base polygon, transforms, AABB, SAT).
	- HPC-focused control: robust bracketing logic (separates provable area-infeasible bound from SA-discovered `L_low`), trial orchestration, logging, checkpointing, and `main` that supports many CLI flags (time limits, checkpoints, polish options, output prefix for SLURM arrays).

- `cmaes/cmaes_pack_poly.c`:
	- Small utils: `streq`, `ensure_dir`, `die`, `clamp`, angle wrapping.
	- RNG including `rng_normal` for Gaussian samples (Box–Muller) used by CMA-ES.
	- Geometry helpers: base polygon data, transforms, AABB, and triangle helpers.
	- Objective/evaluation: routines that map a CMA-ES candidate (cx,cy,theta for each poly) into an objective value combining `alpha*L + lambda*overlap + mu*outside`.
	- CMA-ES loop and `main` which parses CLI options and writes `csv/`/`img/` outputs for the best samples.

- `run/HPC_DEMO/src/main.c`:
	- Signal handling: `handle_sigterm` sets a global `g_stop_requested` flag.
	- Output helpers: `write_svg`, `write_csv` — format and write solver state to disk.
	- `usage` and `main` — parse CLI args, initialize `State`, set PhaseParams and Weights`, run the high-level bisection loop, call the modular annealing solver (`try_pack_at_current_L`-style functions in other modules), and persist snapshots/logs.

Test coverage and recommendations
--------------------------------
Current state:
- There are no automated tests in the repository (no `tests/` directory or test harness detected). Adding tests will improve correctness, regression safety, and refactoring confidence.

Suggested test strategy:
- Unit tests (fast, run locally):
	- RNG determinism: verify `rng_seed` + `xorshift64star` reproducibility.
	- Geometry primitives: `rot_trans`, `build_world_verts`, `aabb_of_verts`, `base_polygon_area`, `base_bounding_radius`.
	- Collision checks: triangle-triangle SAT, polygon AABB rejection, `overlap_pair` behaviour for circles.
	- Spatial hash / grid: insert/query semantics and neighbor enumeration.
	- IO helpers: tiny tests that write CSV/SVG to a temp dir and validate simple format.

- Integration / smoke tests (slower):
	- Small N pack attempts: run SA and CMA-ES for tiny `N` (e.g., 1–3) with deterministic seeds and assert feasibility or expected invariants (no overlaps beyond tolerance, output files produced).
	- Bracketing logic: run the outer shrink on a tiny problem and assert that `L_low < L_high` and that a feasible `L` is found within expected bounds.

- Test harness & tooling suggestions:
	- Use a lightweight C test framework such as Criterion (https://github.com/Snaipe/Criterion) or Unity for unit tests.
	- Add a `tests/` directory with individual test programs and a `Makefile` target `make test` that builds tests and runs them.
	- For coverage, compile with `-fprofile-arcs -ftest-coverage` (GCC) and run tests to generate `.gcda`/`.gcno`, then use `gcov`/`lcov` to produce HTML reports.

Example test build & coverage commands (suggested):

```bash
# Install Criterion or pick another test runner (system package or add as submodule)
# Build with coverage flags
gcc -O0 -g -fprofile-arcs -ftest-coverage -std=c11 -Wall -Irun/HPC_DEMO/include -c sa_pack_shrink/sa_pack_shrink.c -o sa_pack_shrink.o
gcc -O0 -g -fprofile-arcs -ftest-coverage sa_pack_shrink.o tests/test_geometry.c -o tests/test_geometry -lm
./tests/test_geometry
gcov sa_pack_shrink.c
lcov --capture --directory . --output-file coverage.info
genhtml coverage.info --output-directory out-coverage
``` 

Next steps I can take for you
----------------------------
- Add a `tests/` scaffold with 3–5 unit tests (RNG, geometry, simple overlap) and a `make test` target.
- Add coverage build targets and a small CI config (GitHub Actions) to run tests and upload coverage reports.


---
Generated on: 2026-01-21
