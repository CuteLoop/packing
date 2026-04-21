# GPU Roadmap v5.0 — Native CUDA, Phased Parallelization

**Project:** Polygon Packing SA — HPC Acceleration  
**Builds on:** CPU Roadmap v2.1, Phase 0B flat `State` + `Workspace` interface  
**Strategy:** MS first → Hybrid PT → ER-MS (data-driven only)  
**Architecture:** Native CUDA `.cu`, host-orchestrated, kernel-boundary sync

---

## Critique of Prior Roadmaps

Before the roadmap itself, a synthesis of what prior versions got wrong.

**What all versions agreed on (keep):**
- MS is the correct first GPU target — zero inter-replica synchronization
- 1 block per replica is the right kernel mapping
- Host orchestrates barriers (bisection, swaps, resampling) — not device
- Hybrid PT is correct: GPU grinds epochs, CPU executes swap math
- ER-MS last; only implement if data justifies it

**What the OpenMP-target approach (v3.0) got wrong:**
- OpenMP offload is *lowest rewrite cost*, not *lowest friction*. Compiler support across GCC/Clang/NVHPC varies. Debugging is opaque. Performance tuning is difficult. Native CUDA gives fine-grained control over shared memory and warp behavior that this stochastic geometry workload specifically requires. Choose CUDA.

**Critical gaps in all prior versions:**

| Issue | Impact | Fix |
|---|---|---|
| No memory budget audit before writing kernels | May find R=64 doesn't fit in VRAM after months of work | Add Phase G1.5 |
| No profiling gate before optimization phase | May optimize the wrong bottleneck | Add Phase G3.5 |
| Early-stop via naked write to shared flag | Race condition; threads overwrite each other | Use `atomicOr`; check at epoch boundary, not per-proposal |
| "One thread per replica" in first drafts | Leaves 127 of 128 threads idle per block | Threads within block parallelize over N polygons |
| RNG: same seed across replicas | Correlated trajectories across all replicas | Seed as `hash(global_seed, replica_id)` |
| Data transfer frequency unspecified | Risk of per-proposal cudaMemcpy destroying throughput | Transfer only at slice/epoch boundaries — never per proposal |
| `Workspace` has heap pointers | Deep-copy of pointer structs to device is undefined behavior | Flatten to fixed-capacity arrays before writing any kernel |

---

## Architecture Overview

```
CPU (orchestrator)                GPU (compute engine)
──────────────────                ──────────────────────────────────────────
bisection logic              →    DeviceStateBank [R × N] SoA layout
slice dispatch               →    ms_slice_kernel <<< R, BLOCK_SZ >>>
swap / resample math         ←    energy + feasibility flags only
logging / CSV                      one block per replica
Slurm / job control                threads within block parallelize over N
```

**Kernel boundary = synchronization barrier.**  
CPU launching a new kernel is the only safe global sync point on GPU. All barrier mechanics (PT swaps, ER-MS resampling) happen on host between kernel launches.

---

## Phase G0 — Freeze the CPU Baseline

**Goal:** Lock the CPU implementation as the correctness oracle. All GPU phases validate against this.

### Tasks

1. Create a git tag `cpu_baseline_ms_v1` from the current passing state.
2. Run the existing golden-case suite (N=10, N=20, N=50) and save:
   - `seed`, `L`, `best_energy`, `feasibility`, `proposal_count`, `final_pose_dump`
3. Capture these per method (ms, erms, pt) into `analysis/baseline_cpu_metrics.csv`.
4. Confirm that re-running the tagged branch within tolerance is automated: `make golden`.

### Deliverables

- `tags/cpu_baseline_ms_v1`
- `tests/golden_cases/` — one subdirectory per instance
- `analysis/baseline_cpu_metrics.csv`
- `Makefile` target: `make golden`

### Gate G0

- `make golden` passes on the tagged branch within floating-point tolerance.
- No GPU code exists yet. This phase is purely protective.

---

## Phase G1 — GPU Kernel Boundary + Data Flow Specification

**Goal:** Before writing any `.cu` file, precisely define what belongs on device and what stays on host.

### Device responsibilities (hot loop — everything inside a slice)

- Per-replica proposal generation (draw Δx, Δy, Δθ from per-replica RNG)
- Transform polygons → recompute AABBs
- Spatial hash rebuild from AABBs
- Overlap + outside penalty evaluation
- Metropolis accept/reject
- Track per-replica best-so-far state and energy
- Set early-stop atomic flag on feasibility

### Host responsibilities (orchestrator — between slices)

- Bisection logic (probe scheduling, bracket updates)
- PT: adjacent swap accept/reject math
- ER-MS: ranking, elite protection, clone/perturb
- Warm-start: downscale translations, repair pass
- Logging (every 2s)
- Slurm/job management
- Final MS-polish

### Data transfer rule — locked

Transfer data **only at these points:**

| Event | Direction | What |
|---|---|---|
| Slice start | host → device | initial `State[r]` for this probe (once per probe, not per slice) |
| Slice end | device → host | `energy[r]`, `feasibility[r]`, `best_replica` pose |
| PT epoch boundary | device → host | `energy[r]` only (not full poses unless swap accepted) |
| ER-MS epoch boundary | device → host | `energy[r]` + `State[top_25%]` |

**Never transfer per-proposal. Never transfer per-step.**

### Deliverables

- `docs/GPU_BOUNDARY.md` — copy of the two tables above plus the transfer rule
- `docs/HOST_DEVICE_DATAFLOW.md` — sequence diagram: `[host: probe loop] → [device: slice kernel] → [host: bisection update]`
- Stub function signature: `void run_slice_ms_gpu(DeviceStateBank *bank, SliceParams *p, SliceResult *out);`

### Gate G1

- `docs/GPU_BOUNDARY.md` exists and is reviewed
- The stub function signature compiles (empty body, no GPU code yet)

---

## Phase G1.5 — Memory Budget Audit

**Goal:** Compute the per-replica memory footprint before writing a single kernel. Discovering an OOM at Phase G4 wastes weeks.

### Tasks

1. With flat (pointer-free) `Workspace` in mind, compute exact bytes per replica:

```c
// Fill in actual sizes from your codebase:
size_t bytes_state     = sizeof(State);                          // pose + annealing + RNG + cached eval
size_t bytes_workspace = sizeof(int)   * MAX_GRID_CELLS          // grid_heads
                       + sizeof(int)   * MAX_N                   // grid_next
                       + sizeof(AABB)  * MAX_N                   // bounds
                       + sizeof(Poly)  * MAX_N                   // transformed_polys
                       + sizeof(Contact) * MAX_CONTACTS;         // contacts (biggest unknown)

size_t bytes_per_replica = bytes_state + bytes_workspace;
```

2. Compute maximum R for the target GPU:

```
max_R_global = GPU_VRAM_bytes / bytes_per_replica
max_R_shared = shared_mem_per_SM / bytes_workspace   // if workspace goes in shared mem
```

3. For the P100 (16 GB global, 64 KB shared per block):
   - Global: almost certainly not a constraint at R=20..256
   - **Shared memory is the real constraint.** If `bytes_workspace` > 32 KB, shared staging is impossible and must go in global.

4. Document worst-case contact count `MAX_CONTACTS`. This is the variable most likely to blow shared memory budget.

### Deliverables

- `docs/GPU_MEMORY_BUDGET.md` — table with field-level byte counts, totals, max R estimates

### Gate G1.5

- `bytes_per_replica` is computed and documented
- `MAX_CONTACTS` has a justified bound (not a guess)
- Target: R=64 fits comfortably in VRAM. If it does not, the Workspace design must change before Phase G2.

---

## Phase G2 — Flatten State, SoA Device Layout

**Goal:** Produce a GPU-native memory layout. The CPU `State` struct from Phase 0B stays untouched. A parallel `DeviceStateBank` in SoA layout lives only on device.

### Locked design decisions

1. CPU-side `State[R]` (AoS) is unchanged — CPU logic stays readable.
2. Introduce `DeviceStateBank` in SoA for device kernels — coalesced access across replicas.
3. `Workspace` must be fully flattened (no heap pointers) before mapping to device.
4. Pack/unpack functions are the only site for host↔device data movement.

### SoA layout

```c
// include/device_state.cuh
typedef struct {
    // Pose  [r * N + i]
    double *x, *y, *theta;       // length R*N each

    // Per-replica SA state  [r]
    double *temp, *step_pos, *step_theta;
    uint64_t *rng;               // xorshift64* state, one per replica
    double *energy, *overlap, *outside;
    int    *feasible;
    int    *stop_flag;           // single shared atomic early-stop

    int R, N;
} DeviceStateBank;
```

**Index macro — use everywhere, no raw arithmetic in kernels:**

```c
#define IDX(r, i, N)  ((r) * (N) + (i))
```

### Workspace flattening

Replace all pointer fields with fixed arrays. Sizes come from Phase G1.5 audit:

```c
// Before (CPU, heap pointers — cannot map to device):
typedef struct { int *grid_heads; Contact *contacts; ... } Workspace;

// After (GPU-ready, flat POD):
typedef struct {
    int     grid_heads[MAX_GRID_CELLS];
    int     grid_next[MAX_N];
    AABB    bounds[MAX_N];
    Poly    transformed[MAX_N];
    Contact contacts[MAX_CONTACTS];
} FlatWorkspace;
```

If `sizeof(FlatWorkspace)` exceeds 32 KB (per Phase G1.5), allocate workspace in global memory rather than shared memory and document this decision.

### RNG seeding — locked

```c
// Each replica gets an independent stream, fully deterministic from global seed:
d_rng[r] = xorshift64_mix(global_seed, (uint64_t)r * 6364136223846793005ULL);
```

**Never use `rand()` or any libc RNG on device. Never share an RNG state between replicas.**

### Deliverables

- `include/device_state.cuh` — `DeviceStateBank`, `IDX` macro, `FlatWorkspace`
- `src/device_state.cu` — `bank_alloc`, `bank_free`, `bank_pack(State *cpu, DeviceStateBank *dev)`, `bank_unpack`
- `include/gpu_rng.cuh` — `__device__ xorshift64_next()`, `__device__ uniform_double()`
- `tests/test_pack_unpack.cu` — round-trip for R=20, N=200; assert byte-exact equality

### Gate G2

- Round-trip pack/unpack test passes for R=20, N=200
- No `cudaMemcpy` calls outside `device_state.cu`
- `sizeof(FlatWorkspace)` documented and decision (shared vs global) recorded in `docs/GPU_MEMORY_BUDGET.md`

---

## Phase G3 — Naive GPU MS (Correctness First)

**Goal:** Get a correct GPU MS implementation before any optimization. A slow correct kernel is far more valuable than a fast incorrect one.

### Kernel mapping — locked

```
Grid:   R blocks  (one block = one replica)
Block:  BLOCK_SZ threads  (threads cooperate over N polygons within a replica)
```

Inside each block, threads handle polygon work in stride:

```c
// Thread tid handles polygons: tid, tid+blockDim, tid+2*blockDim, ...
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    // transform polygon i, update AABB i, insert into spatial hash
}
__syncthreads();
// ... reduction of penalties ...
```

### Early-stop — locked

Do **not** check the stop flag every proposal. This causes warp divergence and wastes atomics. Check at the epoch boundary (every `K_check = 200` proposals):

```c
if (proposals % K_check == 0) {
    if (atomicOr(d_stop, 0)) return;   // read-only check
}
// On feasibility:
if (is_feasible) atomicOr(d_stop, 1); // write
```

After kernel returns, host checks whether early-stop was triggered.

### Kernel signature

```c
__global__ void ms_slice_kernel(
    DeviceStateBank *bank,
    int K,               // proposals per replica per slice
    int N,               // polygons
    double L,            // container side length
    EvalParams ep,
    SAParams sa
);
```

### Deliverables

- `src/eval_kernel.cu` — penalty evaluation as a cooperative block operation; validated against CPU `evaluate_state` first before any proposal kernel
- `src/proposal_kernel.cu` — Metropolis propose+accept on device
- `src/method_ms_gpu.cu` — slice runner calling the above kernels
- `tests/test_eval_vs_cpu.cu` — compare GPU eval to CPU eval on 100 random states; assert `|Δ| < 1e-9`
- `tests/test_ms_gpu_vs_cpu.cu` — same seed, same R: GPU and CPU MS bisection converge to same bracket within tolerance

**Implementation order inside this phase:**
1. `eval_kernel` only — validate against CPU before touching proposals
2. Add `proposal_kernel` — validate determinism (same seed → same trajectory)
3. Assemble `ms_slice_kernel` — validate end-to-end

### Gate G3

- GPU and CPU MS bisection brackets match on N=20, R=10, fixed seed
- No NaNs, no illegal memory accesses (`cuda-memcheck` or `compute-sanitizer`)
- `make test` passes including GPU tests

---

## Phase G3.5 — Profiling Sanity Check

**Goal:** Measure before optimizing. Never assume the kernel is the bottleneck.

### Tasks

1. Run Nsight Systems on a medium case (N=50, R=20, 30-second budget):

```bash
nsys profile --trace=cuda,osrt ./run --backend gpu --N 50 --R 20 --time 30
```

2. From the Nsight timeline, measure:
   - Fraction of wall time inside GPU kernels vs. CPU overhead
   - Fraction of kernel time in `eval_kernel` vs. `proposal_kernel`
   - PCIe transfer time (should be negligible)

3. Run Nsight Compute on `ms_slice_kernel`:

```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
              l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
              smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared.sum \
    ./run --backend gpu --N 50 --R 1
```

4. Document: achieved occupancy, SM utilization, global memory throughput.

### Deliverables

- `docs/MS_GPU_PROFILE_BASELINE.md` — kernel time breakdown, occupancy, bandwidth
- `analysis/profile_ms_gpu.py` — parse Nsight CSV exports and plot

### Gate G3.5

- ≥ 80% of wall time is inside GPU kernels (if < 80%, the CPU orchestration loop is the bottleneck — fix that before any kernel optimization)
- Bottleneck identified: is it memory bandwidth or compute?
- Occupancy baseline documented (accept any value here; will optimize in G4)

---

## Phase G4 — Optimize MS

**Goal:** Maximize throughput of the already-correct MS kernel. Do not change behavior — only performance.

### Optimization ladder (work top-to-bottom, stop when speedup plateaus)

**1. Block size tuning**

Start `BLOCK_SZ = 128`. Try 64, 128, 256. Measure occupancy and proposals/sec at each. The sweet spot for N=200 on P100 is likely 128 or 256.

**2. Shared memory staging for per-replica pose data**

If `sizeof(FlatWorkspace) < 32 KB`, move workspace into `__shared__`. This eliminates repeated global reads for the polygons that are accessed many times per proposal.

```c
__shared__ AABB s_bounds[MAX_N];
__shared__ double s_x[MAX_N], s_y[MAX_N], s_theta[MAX_N];
// Load once per proposal batch, reuse across penalty checks
```

**3. Fuse transform + AABB into a single pass**

Currently two separate loops. One loop reads pose, writes transformed poly and AABB simultaneously — halves global writes.

**4. Reduce branch divergence in overlap detection**

If threads early-exit on AABB miss, divergence is high when most pairs don't overlap. Restructure: first pass builds a compact candidate-pair list cooperatively, second pass checks full collision on that compact list.

**5. Persistent kernel (multi-step slice)**

Remove the per-epoch kernel launch overhead by running the full slice in one long kernel with internal loop. Only valid after correctness is confirmed.

### Metrics to track at each step

| Metric | Tool | Target |
|---|---|---|
| Proposals/sec | timing | > 10× CPU baseline |
| Kernel occupancy | ncu | > 50% |
| Global mem bandwidth | ncu | > 40% peak |
| Acceptance rate | log CSV | stable (no divergence artifact) |

### Deliverables

- Updated `src/method_ms_gpu.cu`
- `docs/MS_GPU_OPT_NOTES.md` — table: each optimization step, measured speedup, kept or reverted
- `analysis/profile_ms_gpu.py` updated with post-optimization profiles

### Gate G4

- GPU MS is decisively faster than CPU MS for R ≥ 20 on N=100 (target: ≥ 10× wall-clock)
- Optimization notes document which changes were kept and which were reverted and why

---

## Phase G5 — MS Production Graph Suite on GPU

**Goal:** Run the full graph-suite studies using GPU MS. Get real data before adding PT.

### Tasks

1. Port `scripts/run_graph_suite.slurm` → `scripts/run_graph_suite_gpu.slurm`:
   - Add `#SBATCH --gres=gpu:1`
   - Add `--backend gpu` CLI flag
2. Sweep: R ∈ {10, 20, 50}, N ∈ {20, 50, 100, 200}, 2 seeds each
3. Compare against CPU baseline from Phase G0

### Deliverables

- `scripts/run_graph_suite_gpu.slurm`
- `analysis/compare_cpu_gpu_ms.py` — overlay bisection traces, η, proposals/sec
- Results CSV committed to `analysis/results/gpu_ms_graph_suite.csv`

### Gate G5

- GPU MS produces valid bisection + log CSVs for all sweep points
- Convergence plots show GPU MS reaches same solution quality as CPU MS in less wall-clock time
- η values match between CPU and GPU runs on same seed (within tolerance)

---

## Phase G6 — Hybrid GPU PT

**Goal:** Add parallel tempering without global device synchronization. The GPU grinds epochs; the CPU controls the temperature ladder.

### Architecture — locked (hybrid)

```
[host] initialize temperature ladder T[0] < T[1] < ... < T[R-1]
[host] map State bank to device

loop until slice budget exhausted:
    [device] pt_epoch_kernel <<<R, BLOCK_SZ>>> (run K_swap = 200*N proposals)
    [host]   cudaDeviceSynchronize()
    [host]   cudaMemcpy energy[0:R] from device   ← energies only, not full poses
    [host]   for each adjacent pair (a, b) in deterministic order:
                 compute ΔE*(1/Ta - 1/Tb)
                 if metropolis accept: flag swap (a, b)
    [host]   if any swaps: cudaMemcpy swapped State pairs host→device
    [host]   quench check: if any d_feasible[r]:
                 copy feasible replica to replica 0
                 run short repair on device
                 re-evaluate; if feasible → mark probe done

[host] unpack best state
```

### Why host-side swap math

Swap decisions require reading two replicas' energies and temperatures. On device this requires inter-block communication — either a global barrier or a separate kernel. On host it requires a 4R-float memcpy and a 5-line loop. The host approach eliminates an entire class of synchronization bugs with negligible performance cost (swaps happen every 200*N proposals, not every step).

### Temperature ladder — locked

```c
// Geometric spacing; replica 0 is always the cold (lowest temperature) replica
T[r] = T_max * pow(T_min / T_max, (double)r / (R - 1));
// Store in __constant__ memory for fast kernel reads
```

### Deliverables

- `src/method_pt_gpu.cu`
- `src/pt_swap_host.c` — the 5-line host swap logic (not in a CUDA file)
- `tests/test_pt_gpu_swap.cu` — verify temperature ladder preserved after 10 epochs; swap acceptance rate in (0.1, 0.5)
- `tests/test_pt_quench_gpu.cu` — feasible replica correctly promotes to cold replica

### Gate G6

- PT GPU at N=100, R=20 produces nonzero swap acceptance rate
- Temperature ladder is preserved after 100 swap epochs
- On at least one hard case (N=100), PT reaches feasibility faster than MS wall-clock

---

## Phase G7 — ER-MS (Optional, Data-Driven)

**Goal:** Implement evolutionary resampling only if Phase G5/G6 data shows PT is insufficient on the hardest packing topologies.

**Do not start this phase** unless:
- GPU PT is fully validated (Gate G6 passed)
- There exist specific hard cases where PT fails and ER-MS is hypothesized to help
- The expected quality gain justifies the added orchestration complexity

### Architecture (if implemented)

Same hybrid pattern as PT: GPU runs `K_resample` proposals per epoch, CPU performs all resampling logic between kernel launches.

```
[device] ms_epoch_kernel <<<R, BLOCK_SZ>>> (K_resample proposals)
[host]   cudaMemcpy energy[0:R] + feasible[0:R]
[host]   sort replicas by energy
[host]   protect top 25%; for bottom 75%: select source, copy State, reseed RNG
[host]   cudaMemcpy updated State[0:R] to device
[device] next ms_epoch_kernel
```

### Gate G7

- ER-MS outperforms Hybrid PT on at least one documented hard case
- Data is published in `analysis/erms_vs_pt_comparison.py`

---

## Sprint Schedule

| Sprint | Phases | Goal |
|---|---|---|
| 1 | G0, G1, G1.5 | Baseline locked, boundary defined, memory budget known |
| 2 | G2, G3 | Naive GPU MS correct, validated against CPU oracle |
| 3 | G3.5, G4 | Profile → optimize → documented speedup |
| 4 | G5 | Full graph suite on GPU, production data |
| 5 | G6 | Hybrid PT running and validated |
| 6 | G7 (if justified) | ER-MS |

---

## Test Taxonomy

Every phase must pass tests in all four categories before the gate is considered met.

### Correctness tests

- Same seed + same R → reproducible GPU behavior within numeric tolerance
- GPU penalty evaluation matches CPU on 100 random states (`|Δ| < 1e-9`)
- No NaN/Inf in any energy, pose, or RNG output
- Feasibility flag consistent with penalty values
- Best-so-far energy never increases without a corresponding proposal acceptance
- Host↔device round-trip preserves poses bit-exactly

### Physics / geometry tests

- Transformed polygons match CPU transforms on canned states
- Overlap penalties match CPU on canned states with known hand-computed answers
- Outside penalties match CPU on canned states

### Performance tests

- Proposals/sec vs. R (scaling curve)
- Slice wall-clock vs. CPU MS (speedup table)
- Kernel time fraction (from Nsight; must be ≥ 80%)
- Occupancy at production block size

### Reliability tests

- 10-minute run with `compute-sanitizer` on N=100, R=20: no illegal accesses, no race conditions
- 10-minute run: no energy drift into NaN
- Multiple seeds produce qualitatively consistent bisection behavior (no pathological outliers)

---

## Quick-Reference: Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Pointer in struct mapped to device | segfault inside kernel | Flatten all `Workspace` fields to fixed arrays (Phase G2) |
| Shared RNG seed across replicas | All replicas follow identical trajectory | `rng[r] = mix(seed, r)` (Phase G2) |
| Per-proposal `cudaMemcpy` | 99% of time in PCIe transfer | Transfer only at slice/epoch boundary (Phase G1 rule) |
| Naked write to early-stop flag | Race condition, inconsistent stops | `atomicOr` only (Phase G3) |
| Early-stop check every proposal | Severe warp divergence | Check every `K_check=200` proposals (Phase G3) |
| Optimize before profiling | May optimize the wrong kernel | Gate G3.5 must pass before G4 starts |
| Port all three methods at once | Debugging three interacting systems | MS only until Gate G5; add PT in G6 |
| One thread per replica | 127/128 threads idle per block | Threads parallelize over N polygons within a block |