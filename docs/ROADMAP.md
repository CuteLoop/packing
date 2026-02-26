Absolutely — I’m incorporating the feedback that reduces ambiguity for an agent and lowers risk. Concretely, I’m updating the roadmap to:

* **Lock Option A (Flat State)** and remove alternatives.
* Add a **Phase 0A “Codebase Inventory + Dependency Graph”** (agent needs it).
* Provide the **exact Flat `State` + `Workspace` interface** now (Phase 0B deliverable).
* Remove conditional language (“if/or/pick one”).
* Specify **exact barrier mechanics** for ER-MS and PT.
* Clarify determinism: we guarantee **run-level determinism** (same seeds ⇒ same behavior given fixed OpenMP scheduling), but **not bitwise** floating-point equivalence across thread counts; logs are reproducible up to tolerances.
* Add **PT quench validity check**: after quench+repair, re-evaluate feasibility; only then early-stop.

I’m *not* incorporating MPI notes because we’re locked to single-node OpenMP.

Below is the **Roadmap v2.1** plus the **Flat State + Workspace interface** (the critical artifact).

---

# ROADMAP v2.1 — Agent-Executable Phases, Deliverables, Gates

## Phase 0A — Codebase Inventory and Dependency Graph (mandatory)

**Goal:** eliminate guessing.

### Tasks

1. Read all files in `run/HPC_DEMO/src/*.c` and `run/HPC_DEMO/include/*.h`.
2. Produce:

   * list of all structs/types
   * list of all global variables (file + name + type)
   * list of all functions (signature + file)
   * for each function: which globals it reads/writes (best-effort)
3. Produce a module dependency graph:

   * who calls who (at file granularity)
   * which modules depend on geometry vs hash vs physics

### Deliverables

* `docs/INVENTORY.md`
* `docs/DEPENDENCY_GRAPH.md` (simple text callgraph is fine)

### Gate 0A

* Inventory docs exist and are complete enough to locate all mutable state in `annealing.c`, `physics.c`, `spatial_hash.c`.

---

## Phase 0B — Lock ReplicaState + Workspace using existing types (minimal diff)

**Goal:** enable safe swaps/clones without pointer bugs, without rewriting `spatial_hash.c`.

### Locked design decisions

1. Do not create a new `State` type. `common.h: State` remains untouched initially.
2. Introduce new types:
   - `ReplicaState` (flat POD; swappable)
   - `Workspace` (per-thread cache; contains one `Grid`)
3. Reuse existing types everywhere:
   - `Vec2`, `Tri`, `AABB`, `Grid`, `Totals`, `Weights`, `RNG`, `PhaseParams`
4. No speculative types: no `Polygon`, `Contact`, `EvalParams`, `SAParams`.
5. No raw grid arrays: Workspace uses existing `Grid` and `grid_*` API.

### Deliverables (code)

- `run/HPC_DEMO/include/replica.h`
- `run/HPC_DEMO/src/replica.c`
- `run/HPC_DEMO/tests/test_replica_swap.c`
- `run/HPC_DEMO/tests/test_rebuild_derived.c`
- `run/HPC_DEMO/Makefile` target: `make test`

### Gate 0B

- POD swap works
- clone reseeds RNG properly
- rebuild uses existing `grid_rebuild` and `update_instance` logic (via wrapper)

---

## Phase 0C — Refactor SA/physics to be State-driven (single replica)

**Goal:** remove implicit mutable globals; make everything take `(State*, Workspace*)`.

### Tasks

1. Refactor penalty evaluation:

   * `evaluate_state(State* s, Workspace* w, double L, const EvalParams* p)`
2. Refactor proposal step:

   * `propose_step(State* s, Workspace* w, double L, const SAParams* sa)`
3. Ensure any scratch buffers used by collision checks move into `Workspace` (thread-safe).

### Deliverables

* updated `src/physics.c`, `src/annealing.c`, any touched modules
* `tests/test_single_replica_regress.c` (runs fixed seed and asserts basic invariants: no NaN, energies finite, feasibility stable once achieved)

### Gate 0C

* Single-replica run still produces valid outputs and no races/leaks.
* No module uses mutable global scratch buffers (or they are made const).

---

## Phase 1 — Shared scaffolding: bisection + warm-start + logging (method-agnostic)

**Goal:** shared driver comes before method logic.

### Tasks

1. Implement bisection engine per spec:

   * γ(N) table
   * slice schedule buckets
   * N=200 emergency expansion rule
2. Implement warm-start between probes:

   * store best feasible
   * downscale translations
   * repair pass counted inside slice
3. Implement CSV logging schema:

   * `*_bisection.csv`
   * `*_log.csv` (every 2s)
   * method diagnostic columns included

### Deliverables

* `src/bisection.c`, `include/bisection.h`
* `src/logger.c` updates
* `docs/CSV_SCHEMA.md`
* `analysis/validate_schema.py` (parses CSV, checks required columns)

### Gate 1

* Run N=20 MS (R=1) end-to-end produces parseable CSVs and shrinks bracket.

---

## Phase 2 — Method A: MS multi-replica slice runner (OpenMP)

**Goal:** parallel backbone.

### Tasks

1. Implement `run_slice_ms(...)` with OpenMP:

   * per-thread: `Workspace w[tid]`
   * per-replica: `State s[r]`
2. Early stop if any worker feasible:

   * shared atomic flag
3. Deterministic reductions:

   * best state update uses critical section (fixed tie-break by replica id)

### Deliverables

* `src/method_ms.c`, `include/methods.h`
* `tests/test_ms_parallel.c`

### Gate 2

* N=50 small run completes without races.
* `make test` passes.

---

## Phase 3 — Method B: ER-MS with barrier-based resampling

**Goal:** implement exact resampling mechanics (no “workers hit threshold asynchronously”).

### Barrier mechanics (locked)

Each ER-MS epoch does:

1. Each worker performs exactly `K_resample = 200*N` proposals (a for-loop).
2. `#pragma omp barrier`
3. Thread 0:

   * ranks states by energy
   * protects top 25%
   * clones/perturbs others (with RNG reseed-for-clone)
4. `#pragma omp barrier`
5. Next epoch begins.

### Deliverables

* `src/method_erms.c`
* `tests/test_erms_resample.c` (assert resample count increments correctly; protected states unchanged)

### Gate 3

* ER-MS runs and logs resample events; no deadlocks.

---

## Phase 4 — Method C: PT with barrier-based swaps + quench-to-cold

**Goal:** safe swaps and fast bisection.

### Barrier mechanics (locked)

Each PT epoch does:

1. Each replica runs exactly `K_swap = 200*N` proposals.
2. `#pragma omp barrier`
3. Thread 0 performs adjacent swap attempts in deterministic order:

   * (0,1), (2,3), … then next epoch flips parity (1,2),(3,4),…
4. `#pragma omp barrier`

### Early-stop rule (locked, improved)

If **any replica** becomes feasible at any time:

1. Copy that replica’s `State` into replica 0 (“quench-to-cold”).
2. Run `rebuild_derived` and a short repair pass on replica 0.
3. Re-evaluate feasibility on replica 0.
4. If replica 0 is feasible → mark probe feasible and end slice early.
5. Else continue slice (do not claim feasibility).

### Deliverables

* `src/method_pt.c`
* `tests/test_pt_swap_math.c` (checks swap acceptance formula on synthetic energies)
* `tests/test_pt_quench.c`

### Gate 4

* PT pilot at N=100 produces nonzero swap attempts and logs acceptance.

---

## Phase 5 — Standardized polish procedure (MS-polish + Stochastic Shave)

**Goal:** produce hero-quality packing without blowing it up.

### Polish mechanics (locked)

* Always polish with **MS-polish** at `R=20` on fixed `L_best`.
* Stall detector: if best penalty unchanged for 10 minutes:

  * apply **Stochastic Shave** for 1 second:

    * set outside penalty weight = 0
    * keep overlap penalty weight unchanged/high
  * restore outside penalty weight
* Continue until polish budget ends.

### Deliverables

* `src/polish.c`
* `tests/test_polish_shave.c` (shave toggles weights and restores them)

### Gate 5

* On N=50 feasible case, polish maintains feasibility and logs at least one improvement or stable penalties.

---

## Phase 6 — Orchestration + analysis-first

**Goal:** avoid wasting 4-hour runs.

### Tasks

1. Extend CLI: `--method`, `--R`, `--seed`, `--run_id`, `--mode graph|hero`
2. Slurm scripts:

   * graph suite: 2 seeds concurrently (R=10)
   * hero: 1 job (R=20)
3. Analysis scripts:

   * `analysis/aggregate.py` generates trace plots and computes η.

### Deliverables

* `scripts/run_graph_suite.slurm`
* `scripts/run_hero.slurm`
* `analysis/aggregate.py`

### Gate 6

* 10-minute dry run generates valid plots and η for a toy case.

---

## Phase 7 — Decision gates before hero

* Gate A: N=20 end-to-end shrinks bracket by ≥3 probes.
* Gate B: ER-MS and PT run without deadlocks and produce diagnostics.
* Gate C: N=100 10-minute run finds feasibility at least once (any method). If none do, fix γ or schedules before hero.

---

# LOCKED Phase 0B Interface — Flat `State` + `Workspace`

This is the “no ambiguity” contract the agent must implement.

## `include/state.h`

```c
#pragma once
#include <stdint.h>

#ifndef MAX_N
#define MAX_N 200
#endif

#ifndef MAX_CONTACTS
#define MAX_CONTACTS 200000
#endif

// Forward declarations for geometry structs already in your codebase.
typedef struct Polygon Polygon;
typedef struct AABB AABB;
typedef struct Contact Contact;

// Evaluation parameters (weights, thresholds)
typedef struct {
    double eps_feas;
    double w_overlap;
    double w_outside;
} EvalParams;

// SA parameters (step sizes, schedule knobs)
typedef struct {
    double step_pos;
    double step_theta;
    double temp;
} SAParams;

// POD-only swappable configuration.
typedef struct {
    // Pose variables
    double x[MAX_N];
    double y[MAX_N];
    double theta[MAX_N];

    // Annealing state
    double temp;            // current temperature (replica-specific)
    double step_pos;        // adaptive step size (if used)
    double step_theta;

    // RNG (POD)
    uint64_t rng_state;     // xorshift64* state (or your chosen 64-bit state)

    // Cached evaluation outputs from last eval
    double energy;
    double overlap_penalty;
    double outside_penalty;
    int is_feasible;

    // Counters
    uint64_t proposals_done;     // total proposals in this probe
    uint64_t epoch_proposals;    // proposals since last resample/swap
    int replica_id;              // 0..R-1 (for deterministic tie-breaks)
} State;

// Per-thread cache / scratch. Never swapped.
typedef struct {
    // Spatial hash buffers (preallocated, reused)
    int *grid_heads;
    int *grid_next;
    int *grid_keys;
    int grid_capacity;

    // Derived geometry (recomputed from State)
    AABB bounds[MAX_N];
    Polygon *transformed_polys;   // if heavy, allocate once in workspace
    int transformed_capacity;

    // Collision / physics scratch
    Contact *contacts;
    int contacts_capacity;

    // Any other per-thread scratch arrays used in physics/spatial_hash
    double *tmp_doubles;
    int tmp_capacity;
} Workspace;

// Lifecycle
void workspace_init(Workspace *w, int N);
void workspace_free(Workspace *w);

void state_init_random(State *s, int N, uint64_t seed, double L, double temp_init);
void state_clone(State *dst, const State *src, uint64_t new_seed);
void state_swap(State *a, State *b);

// Must be called after swap/clone and after any pose update batch.
void rebuild_derived(const State *s, Workspace *w, int N, double L);

// Evaluate penalties and feasibility; updates cached fields in State.
void evaluate_state(State *s, Workspace *w, int N, double L, const EvalParams *p);
```

## `src/state.c` (behavioral requirements)

* `state_swap` is a simple POD swap (safe `memcpy`).
* `state_clone` copies pose + annealing state, then **reseeds** RNG with `new_seed`.
* `rebuild_derived` does **exactly**:

  1. apply transforms to polygons (into workspace cache)
  2. recompute AABBs
  3. rebuild spatial hash grid from AABBs/polys
* `evaluate_state` calls into `physics.c` but **must not** use globals.

---

# Incorporating η into analysis (locked)

`analysis/aggregate.py` must compute:

* `A_total = N * A_poly` (since polygons identical)
* `eta = A_total / (L_best^2)` from final hero run

Benchmarks are descriptive only:

* η > 0.55 “solid”
* η > 0.65 “hero-tier”

---

# One last clarification: determinism

We guarantee:

* same seeds + same R + same scheduling policy ⇒ same outputs within numeric tolerance
  We do **not** require bitwise identical floating-point results across different R.

---

If you want, I can now also output:

* `docs/DEPENDENCY_GRAPH.md` *template* (the exact structure the agent should fill), and/or
* the first draft of `analysis/aggregate.py` that validates CSV schema and plots traces + η.
