Yes — and I’ll do the full update now. The key is: **Roadmap v2.1 must be replaced** because Phase 0A revealed the interface is misaligned with the actual codebase (`common.h` already defines `State`, `Grid` exists, no `Polygon/Contact`, etc.). What you want is a **codebase-faithful, minimal-diff, agent-executable roadmap**.

Below is **ROADMAP v2.4 (Definitive)**. It fully incorporates:

* your Phase 0A inventory/graph,
* the critique about struct dumps + monolith pseudocode,
* “minimal diff” workspace design (reuse `Grid`),
* removal of speculative param structs and types,
* explicit control-flow replacement in `main.c`,
* explicit barrier mechanics,
* determinism policy,
* PT quench verify-after-rebuild.

---

# ROADMAP v2.4 — Definitive Agent-Executable Plan (HPC_DEMO → MS / ER-MS / PT + Hero)

## Locked global constraints

* Single-node CPU only, up to **20 cores**.
* OpenMP only (no MPI).
* Compare **3 methods**:
  **A = MS** (multistart), **B = ER-MS** (elite resample), **C = PT** (parallel tempering).
* Bisection on (L) for **N ∈ {5,10,20,50,100}** (graph suite) and **N=200** (hero).
* Determinism: **run-level determinism** under fixed seeds and `OMP_SCHEDULE=static`, not bitwise identical across different core counts.

---

## Phase 0A — Inventory + Dependency Graph (DONE ✅)

Your deliverables satisfy this phase.

### Locked notes carried forward

* Only real globals impacting solver behavior:

  * `BASE_V`, `TRIS` are read-only constants.
  * `logger.c` uses `g_csv` and `g_prefix` → **thread-0 logging only**.
  * `g_stop_requested` only in main → safe.

---

## Phase 0A.5 — Dump real structs + monolith pseudocode (DONE ✅)

**Goal:** prevent agent guessing and “clean-room redesign.”

### Completed tasks

- Struct definitions copied verbatim from:
  - `common.h`: `State`, `Grid`, `Totals`, `Weights`, `Vec2`, `Tri`, `AABB`
  - `utils.h`: `RNG`
  - `config.h`: `PhaseParams`
  - `annealing.c`: file-local `Move`
- Pseudocode outline documented for:
  - `try_pack_at_current_L(...)`
  - `run_phase(...)`
  - with explicit notes on eval calls, grid ops, geometry updates, and logging

### Deliverables

- `docs/STRUCT_DUMP.md`
- `docs/TRY_PACK_PSEUDOCODE.md`

### Gate 0A.5

- Agent can answer: “where does the proposal loop live?” and “what state is mutated per move?”

---

## Phase 0B — Lock **ReplicaState + Workspace** using EXISTING types (minimal diff) (DONE ✅)

**Goal:** enable swaps/clones without pointer bugs, without rewriting `spatial_hash.c`.

### Locked design decisions

1. **Do not create a new `State` type.** `common.h: State` remains untouched initially.
2. Introduce **new types**:

   * `ReplicaState` (flat POD; swappable)
   * `Workspace` (per-thread cache; contains one `Grid`)
3. **Reuse existing types** everywhere:

   * `Vec2`, `Tri`, `AABB`, `Grid`, `Totals`, `Weights`, `RNG`, `PhaseParams`
4. **No speculative types**: no `Polygon`, `Contact`, `EvalParams`, `SAParams`.
5. **No raw grid arrays**: Workspace uses existing `Grid` and `grid_*` API.

### Files (locked)

* `run/HPC_DEMO/include/replica.h`
* `run/HPC_DEMO/src/replica.c`
* `run/HPC_DEMO/tests/test_replica_swap.c`
* `run/HPC_DEMO/tests/test_rebuild_derived.c`
* Update `run/HPC_DEMO/Makefile` to add `make test`

### Locked `ReplicaState` contents (high-level, final fields chosen after Phase 0A.5 dump)

Must contain only:

* pose arrays: `cx[N]`, `cy[N]`, `th[N]`
* annealing scalars: `temp`, `step_xy`, `step_th`
* RNG: `RNG rng` (must be POD; if `RNG` is not POD, flatten it here)
* cached eval: `Totals totals`, `double energy`, `double feas`, `int is_feasible`
* counters: `proposals_done`, `epoch_proposals`, `replica_id`

### Locked `Workspace` contents

* `Grid grid` (one per thread)
* any scratch currently implicit in physics/annealing (only if Phase 0A.5 reveals it)
* If world-verts/AABB are currently stored in legacy `State`, they move here **only if needed**.

### Required functions (locked signatures)

* `replica_swap(ReplicaState*, ReplicaState*)` — POD swap
* `replica_clone(dst, src, new_seed)` — copy state + **rng reseed** (ER-MS)
* `rebuild_derived(const ReplicaState*, Workspace*, int N, double L, double cell)`
  performs:

  1. per-instance derived updates needed by physics (minimally: AABB if used)
  2. `grid_rebuild(&ws->grid, N, L, cell, rs->cx, rs->cy)`
* `evaluate_replica(ReplicaState*, Workspace*, const Weights*, int N, double L)`
  updates totals/energy/feas/is_feasible

### Gate 0B

* tests pass:

  * swap twice restores identical bytes for ReplicaState
  * clone produces distinct RNG streams
  * rebuild_derived creates a valid grid occupancy

---

## Phase 0C — Refactor annealing/physics to accept ReplicaState + Workspace (single replica) (DONE ✅)

**Goal:** remove the monolith dependency gradually, without breaking the solver.

### Locked approach: “Adapter-first, then migrate”

Because your existing physics functions accept `const State*`, we do this in two steps:

#### 0C.1 Adapter layer (fast path)

Implement a function:

* `void legacy_state_from_replica(State* tmp, const ReplicaState* rs, Workspace* ws, int N, double L)`
  that populates the **minimum required fields** in a temporary `State tmp` so you can call:
* `compute_totals_full_grid(&tmp)`
* `energy_from_totals(&tmp, &weights, &totals)`

This avoids rewriting physics immediately.

#### 0C.2 Migrate physics to `_ws` (correct path)

Once the system is stable, implement:

* `Totals compute_totals_full_grid_ws(const ReplicaState*, Workspace*)`
  and use it directly.

### Deliverables

* Updated `annealing.c` to run a single ReplicaState (R=1) per trial
* `tests/test_single_replica_regress.c`

### Gate 0C

* N=20 run completes
* no NaNs
* feasibility metric behaves consistently

---

## Phase 1 — New bisection driver + warm-start + logging (method-agnostic) (DONE ✅)

**Goal:** shared scaffolding for all methods.

### Locked control-flow change in `main.c`

* Keep old CLI:

  * `./bin/solver N trials out_prefix [seed] [run_id]` continues to work as “demo mode”
* Add **study mode** flags:

  * `--study --method {ms,erms,pt} --R <int> --time_budget_sec <int> --seed <u64> --run_id <u64>`

In study mode:

* `main.c` calls: `bisection_run(method_runner_fn, ...)`
* It does **not** call `try_pack_at_current_L` except in demo mode.

### Warm-start (locked)

Between probes:

* if last probe feasible: keep best ReplicaState `S*`
* initialize next probe from `S*` scaled into smaller L
* run repair micro-pass counted inside slice

### Logging (locked)

* logger is called **only by thread 0 / serial regions**
* produce:

  * `*_bisection.csv` per probe
  * `*_log.csv` every 2 seconds per run
  * log infeasible probes with `min_energy_in_slice` / `min_feas_in_slice`

### Deliverables

* `src/bisection.c`, `include/bisection.h`
* schema doc `docs/CSV_SCHEMA.md`
* `analysis/validate_schema.py` to fail fast

### Gate 1

* N=20, method=MS, R=1 produces valid bisection + logs.

---

## Phase 2 — Method A (MS): OpenMP multi-replica slice runner (DONE ✅)

**Goal:** baseline.

### Locked MS slice mechanics

Inside a slice:

* each replica runs SA for the whole slice budget (or until early stop)
* shared best updated with deterministic tie-break by replica_id
* early stop if any replica feasible (in bisection probe)

### Deliverables

* `src/method_ms.c`, `include/methods.h`
* `tests/test_ms_parallel.c`

### Gate 2

* N=50, R=10 runs without deadlocks and logs are correct. (DONE ✅)

---

## Phase 3 — Method B (ER-MS): barrier-resampling (DONE ✅)

**Goal:** elite injection without asynchronous thresholds.

### Locked ER-MS epoch mechanics

Let `K_resample = 200*N` proposals.
Loop until slice time expires:

1. `#pragma omp parallel for` each replica runs exactly K_resample proposals
2. barrier
3. thread 0:

   * sort by energy (or feas metric priority)
   * protect top 25%
   * clone+perturb rest
   * reseed RNG for clones
4. barrier

### Deliverables

* `src/method_erms.c`
* `tests/test_erms_resample.c`

### Gate 3

* resample events appear in logs (DONE ✅)
* protected replicas unchanged across resample (DONE ✅)

---

## Phase 4 — Method C (PT): barrier-swaps + quench-to-cold with verification (DONE ✅)

**Goal:** PT without slow bisection.

### Locked PT epoch mechanics

Let `K_swap = 200*N`.
Loop until slice time expires:

1. each replica runs exactly K_swap proposals at its temperature
2. barrier
3. thread 0 attempts adjacent swaps in deterministic parity schedule
4. barrier

### Locked early-stop (“quench-to-cold”) with **verify**

If any replica j becomes feasible:

1. copy ReplicaState[j] → ReplicaState[0] (copy RNG as well)
2. `rebuild_derived` on replica 0
3. run repair micro-pass on replica 0
4. **re-evaluate** replica 0
5. if still feasible → probe feasible and early-stop slice
6. else continue slice

### Deliverables

* `src/method_pt.c`
* `tests/test_pt_swap_math.c`
* `tests/test_pt_quench.c`

### Gate 4

* PT pilot at N=100 produces swap attempts and nonzero acceptance. (DONE ✅)

---

## Phase 5 — Standardized polish (MS-polish) + Stochastic Shave (DONE ✅)

**Goal:** make hero output pretty and strong.

### Locked polish

* always MS-polish R=20 at fixed L_best
* “Stochastic Shave”: if no improvement for 10 minutes:

  * set outside weight to 0 for 1 second (overlap weight unchanged)
  * restore

### Deliverables

* `src/polish.c`
* `tests/test_polish_shave.c`

### Gate 5

* N=50 feasibility not destroyed by polish. (DONE ✅)

---

## Phase 6 — Orchestration + analysis-first

**Goal:** do not burn 4 hours blindly.

### Locked run plan (20 cores)

* Graph suite:

  * run 2 seeds concurrently, each with R=10 (10 threads each)
* Hero:

  * one run, R=20

### Analysis deliverables

* `analysis/aggregate.py`:

  * validate schema
  * plot L_best(t)
  * compute packing efficiency η using `base_polygon_area()`

### Gate 6

* 10-minute dry run generates plots and η.

---

## Phase 7 — Decision gates before hero (locked)

* A: N=20 shrinks bracket by ≥3 probes
* B: ER-MS/PT no deadlocks, diagnostics present
* C: N=100 10-min run finds feasibility at least once

---

# Updated “Flat State + Workspace” contract (v2.4 version)

This replaces the old v2.1 `state.h` contract.

## The contract is:

* **ReplicaState** (POD, swappable)
* **Workspace** (contains existing `Grid` + minimal scratch)
* **Existing code types remain primary** (`Grid`, `RNG`, `Weights`, `Totals`)

No `Polygon`, no `Contact`, no raw grid arrays, no duplicate parameter structs.

---

## What still needs to happen next (immediate)

1. Begin Phase 6 (orchestration + analysis scripts).
2. Run Phase 7 decision gates.
