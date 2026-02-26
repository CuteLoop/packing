Here is the revised engineering plan for the Chimera Nonconvex Packing Solver. This plan prioritizes correctness, incremental performance scaling, and deterministic reproducibility on HPC clusters.

# Chimera Nonconvex Packing Solver — Engineering Plan

**Target Architecture:** HPC / Slurm / MPI+OpenMP (Optional P100 GPU)

## Goal

Build a parallel-native solver to pack  distinct non-convex polygons into a square container, minimizing the side length .
**Core Strategy:** Islands + Relax/Squeeze/Kick + Elite Migration.

## Non-negotiables

> * **Correctness Harness:** Cached SAT (Separating Axis Theorem) must match the Triangle Oracle in `DEBUG` builds exactly (within ).
> * **Incremental Updates:** Accept/reject steps must only update touched polygons and allow cheap reversion.
> * **Scalable Broadphase:** Avoid  complexity; use grid hashing or spatial bins.
> * **Strict Determinism:** Fixed seeds + fixed ranks/threads must reproduce key trace signals (e.g., `best_L` over time) exactly.
> 
> 

---

## Phase 0: Benchmark Stabilization (1–2 Commits)

We begin by establishing a ground truth for collision detection speed and correctness.

### Deliverables

* **`tools/collision_benchmark.c`:** A CLI driver accepting:
* `--instances N --pairs M --seed S --mode all|tri|sat_base|sat_cache`
* `--check_agreement K` (Compares Triangle vs. SAT cache on  AABB-passing pairs)


* **Timing Instrumentation:** Separate clocks for cache build, broadphase, and narrowphase.
* **DEBUG Mode:** `-DDEBUG_ORACLE` triggers a non-zero exit on the first SAT/Triangle mismatch, printing pair IDs and poses.

### Acceptance Criteria

* `./bench --instances 2000 --pairs 200000 --mode all --seed 1` runs without crashing.
* Agreement test passes for  on random poses in DEBUG mode.

---

## Phase 1: Refactor into Solver-Ready Modules

Clean separation of concerns to support CPU-first development with future GPU hooks.

### Target Layout (CPU-First)

| Module | Description |
| --- | --- |
| `src/geom_vec2.h` | Vector operations (add, sub, dot, rot). |
| `src/aabb.h` | Axis-Aligned Bounding Box operations. |
| `src/shape_tree.c` | Static definition of polygon shapes. |
| `src/convex_decomp.c` | Triangle merging and local axis definition. |
| `src/instance_cache.c` | Dynamic world vertices/axes and AABBs. |
| `src/collide_sat.c` | Cached Separating Axis Theorem narrowphase. |
| `src/collide_tri_oracle.c` | **DEBUG ONLY**: Ground truth intersection checks. |

### Acceptance Criteria

* Memory ownership rules are explicit and leak-free (checked via Valgrind/Sanitizers).
* Benchmarks produce consistent results pre- and post-refactor.

---

## Phase 2: Mandatory Incremental Cache + Per-Part AABBs

To handle non-convex shapes efficiently, we decompose them into convex parts and maintain a hierarchy of bounding boxes.

### Deliverables

* **`InstanceCache` Structure:**
* `AABB *instAabb` (Per instance)
* `AABB *partAabb` (Per instance  part)
* `Vec2 *worldVerts`, `Vec2 *worldAxis`


* **`cache_update_one(D, C, i, newPose)`:**
* Updates world geometry and AABBs only for instance .


* **Pruning:** `collide_cached_convex()` skips sub-part checks if `partAabb`s do not overlap.

### Acceptance Criteria

* **Test:** `test_cache_update_one_matches_rebuild()` passes.
* **Performance:** Cached SAT speed improves significantly due to part-AABB pruning.

---

## Phase 3: Broadphase Grid Hash

Eliminating  checks using a spatial hashing approach.

### Deliverables

* **`src/grid_hash.c`:**
* `grid_init(cell_size, L_hint)`
* `grid_update_one(i, oldAabb, newAabb)`
* `grid_query_candidates(i, aabb, out_ids[])`


* **Determinism:** The output candidate list `out_ids[]` must be sorted to ensure consistent downstream processing order.

### Acceptance Criteria

* **Test:** `test_grid_no_false_negatives_smallN()` passes (brute force comparison).
* Candidate counts are significantly lower than  for typical distributions.

---

## Phase 4: Energy + Local  (Baseline Solver)

Establishing the physics of the packing problem.

### Energy Function (MVP)

The total energy  is defined as:



*Optional Total Potential:* 

### Deliverables

* **`src/energy.c`:**
* `outside_penalty_aabb(aabb, L)`
* `delta_energy_move_one(i, newPose)`: Calculates change in energy efficiently using neighbor candidates.


* **`src/propose.c`:** Translation and rotation proposals with a  (step size) schedule.
* **`src/accept.c`:** Greedy acceptance or Metropolis criteria (with  schedule).

### Acceptance Criteria

* Fixed  relaxation reliably reduces  from a random initialization.

---

## Phase 5: Relax / Squeeze / Kick Cycle

The core "Anytime" solver logic. We alternate between resolving collisions and shrinking the container.

### Deliverables

* **`src/solver_island.c`:**
1. **Relax:** Run `relax_steps(T_relax)` to minimize overlap.
2. **Squeeze:** `try_squeeze(eps_shrink)` followed by a short repair phase.
3. **Kick:** If stuck, perform `kick()` (subset rotation, removal-reinsertion, or  spike).


* **Tracking:** Maintain `best_L` and `best_pose`.
* **Trace Output:** CSV logging per rank (Time, Iteration, , , `best_L`, Accept Rate).

### Acceptance Criteria

* On small  (), `best_L` decreases monotonically over time.
* Solver remains stable (no explosions) when infeasibility persists.

---

## Phase 6: OpenMP (Intra-Island Parallelism)

Accelerating the evaluation of proposals within a single MPI rank.

### Deliverables

* **Batching:** Define batch size  (e.g., 32–256).
* **Parallel Eval:** Evaluate  proposals in parallel using OpenMP (read-only state access).
* **Commit Rule:**
* *Deterministic:* Choose best  (stable tie-breaking).
* *Stochastic:* Sample based on .



### Acceptance Criteria

* Demonstrable speedup vs. single-proposal loop at equivalent solution quality.

---

## Phase 7: MPI Islands + Elite Migration

The "Overnight Engine" for HPC.

### Deliverables

* **`src/mpi_migration.c`:**
* Periodic `MPI_Allgather(best_L)` to identify the global leader.
* Broadcast **Elite Pose Array** from the winner to all workers.
* **Restart Strategy:** Losers restart search near the elite solution (small perturbation).


* **CLI Flags:** `--mig_seconds`, `--time_limit_seconds`, `--rank_seed_mode`.

### Acceptance Criteria

* Multi-rank median "time-to-best" is faster than a single rank given the same total CPU budget.

---

## Phase 8: GPU Acceleration (Conditional)

*Target: P100. Implementation only if profiling confirms narrowphase is the bottleneck.*

### Strategy

* **Option A:** GPU evaluates collision checks for a moved polygon vs. its neighbors.
* **Option B (Preferred):** GPU evaluates  for massive proposal batches.

### Acceptance Criteria

*  wall-time speedup compared to the equivalent CPU configuration.

---

## Reproducibility & CLI Protocol

To ensure scientific validity and ease of debugging:

* **Deterministic Mode:**
* `seed = base_seed + rank`
* Stable iteration order & sorted candidate lists.
* Fixed batch sizes & tie-break rules.


* **Default Parameters (Overnight Run):**
* **N:** 2000
* **Initial L:** Heuristic (e.g.,  or bbox baseline)
* **Relax:** 50k proposals/cycle
* **Squeeze Repair:** 10k proposals
* **Migration:** Every 2 seconds (wallclock)


### second part. update


This is the Master Engineering Plan to evolve your current serial CPU baseline into the **Chimera N=200 GPU Solver** ready for a 4-hour HPC run.

We will execute this in **5 Phases**. You cannot skip phases; debugging a distributed GPU simulation without a verified physics engine is impossible.

---

### Phase 1: The "Truth" Audit (CPU Verification)

**Goal:** Guarantee the physics math is correct before moving to GPU. If the CPU says , it must physically mean "no overlap."

* **1.1 Implement Reflection:**
* **Action:** Modify `src/propose.c` to use reflection instead of clipping.
* **Why:** Clipping creates artificial clusters at the walls.


* **1.2 Implement Invariants Test:**
* **Action:** Create `src/test_invariants.c`.
* **Task:** Verify Symmetry () and  consistency ().
* **Why:** These bugs will be invisible on the GPU but will prevent convergence.


* **1.3 Flatten Geometry:**
* **Action:** Write a helper `serialize_geometry(ConvexDecomp *D)` in `solver.c`.
* **Task:** Convert the pointer-heavy `ConvexDecomp` struct into a single flat `float` array (max 512 floats) ready for GPU Constant Memory.



**✅ Success Criteria:** `./audit_physics` prints `PASS` for all checks.

---

### Phase 2: The GPU Port (Single Simulation)

**Goal:** Get **one** simulation running on the GPU that matches the CPU exactly.

* **2.1 Data Structures (SoA):**
* **Action:** Create `src/gpu_data.h`.
* **Structure:** Allocate flat arrays for `X`, `Y`, `Angle`, `Energy`, `Temperature` of size `MAX_CHAINS * N`.


* **2.2 Geometry Loading:**
* **Action:** Copy the flat geometry array from Phase 1.3 into CUDA `__constant__` memory.


* **2.3 The "Device" Physics Engine:**
* **Action:** Port `check_overlap` and `propose_move` to `__device__` functions in CUDA.
* **Optimization:** Hardcode loops for SAT checks (unrolling) for speed.


* **2.4 The Verification Kernel:**
* **Action:** Write a kernel that takes a specific pose, calculates Energy, and returns it.
* **Test:** Compare CPU Energy vs. GPU Energy for the same input. They must be identical (within float precision).



**✅ Success Criteria:** `assert(abs(cpu_energy - gpu_energy) < 1e-5)` passes.

---

### Phase 3: The "Hydra" Engine (Scaling to 5,120 Chains)

**Goal:** Run 5,120 simulations in parallel.

* **3.1 The SoA Layout:**
* **Action:** Implement the "Strided" memory access pattern.
* **Rule:** Thread `tid` accesses polygon `i` at `d_X[i * 5120 + tid]`. This ensures memory coalescing (High Bandwidth).


* **3.2 The Annealing Kernel:**
* **Action:** Implement `k_anneal_n200`.
* **Logic:**
* Each thread manages 1 simulation ().
* Thread loads  (temperature).
* Performs 1000 Monte Carlo steps in a loop.
* Writes final state and energy back to global memory.




* **3.3 Host Driver:**
* **Action:** Launch the kernel with `<<< 40 blocks, 128 threads >>>` (Total 5,120 threads).



**✅ Success Criteria:** You can run 5,120 chains for 10 seconds and see the energy decrease on all of them.

---

### Phase 4: The Chimera Logic (Meta-Optimization)

**Goal:** Connect the chains so they work together (Parallel Tempering) and solve the problem (Shrinking).

* **4.1 Replica Exchange (CPU Side):**
* **Action:** After every kernel launch (e.g., every 50k steps):
* Download `d_Energy` and `d_Temp` to CPU.
* Loop through chains  and .
* Swap temperatures based on the Metropolis criterion.
* Upload new `d_Temp` to GPU.




* **4.2 The "Hydraulic Press" (Outer Loop):**
* **Action:** Implement the adaptive controller.
* **Logic:**
* `global_min_E = min(all_chain_energies)`
* If `global_min_E < 0.0001`: `box_size *= 0.999`. (Squeeze).
* Else: Wait.




* **4.3 Reseeding:**
* **Action:** If a chain finds a valid packing () at the new smaller size, copy its positions to the "hot" chains to restart them in this good configuration.



**✅ Success Criteria:** The solver automatically shrinks the box size over time without crashing.

---

### Phase 5: HPC Production Readiness

**Goal:** Prepare for the 4-hour unattended Slurm run.

* **5.1 Checkpointing:**
* **Action:** Every 10 minutes, dump the entire state (`X`, `Y`, `Ang`, `box_size`, `Temp`) to a binary file `checkpoint.bin`.
* **Action:** On startup, check if `checkpoint.bin` exists. If so, load it instead of random initialization.


* **5.2 CLI Arguments:**
* **Action:** Add flags: `--n 200`, `--hours 4`, `--out results.json`.


* **5.3 Logging:**
* **Action:** Print a CSV-style log to stdout: `Timestamp, BoxSize, MinEnergy, AcceptanceRate`. This allows you to plot progress while the job is running.



**✅ Success Criteria:** You can kill the program with `Ctrl+C`, restart it, and it resumes exactly where it left off.

---

### Immediate Next Step

We start at **Phase 1**.

**Do you want the code for `test_invariants.c` (Phase 1.2) to begin the verification?**