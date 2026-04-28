# Claims Audit: `0rep.tex` vs Source Code

Each claim is tagged with:
- **Line**: approximate line number in `0rep.tex`
- **Grep**: short string to locate it in the tex file
- **Status**: ✅ Verified | ❌ Wrong | ⚠️ Partially correct / needs nuance | 🔍 Not yet traced
- **Code ref**: file + line where the claim is implemented

---

## ABSTRACT

### C-ABS-1 — Problem statement
- **Line ~112** | **Grep**: `find the minimum length $L^*$`
- **Claim**: Find minimum side length $L^*$ of square container holding $N$ non-convex polygons without overlap.
- **Status**: ✅ Verified — `bisection_run()` in `src/bisection.c` is the outer loop, `src/main.c` does the bisection manually.

### C-ABS-2 — Method summary
- **Line ~115** | **Grep**: `two-phase Simulated Annealing (SA)`
- **Claim**: "two-phase SA with outer bracketing-bisection loop"
- **Status**: ❌ Misleading — production methods (`method_ms.c`, `method_erms.c`) are single-phase time-budgeted SA. Two-phase only appears in the legacy `try_pack_at_current_L` path in `src/annealing.c:268`, not used in study mode. The outer bisection loop is real.

### C-ABS-3 — Instance sizes
- **Line ~119** | **Grep**: `N \in \{5, 10, 25, 50`
- **Claim**: $N \in \{5, 10, 25, 50, 100, 200\}$
- **Status**: ✅ Verified — matches results table and HPC scripts.

### C-ABS-4 — Density range
- **Line ~120** | **Grep**: `\eta \approx 0.58` ... `\eta \approx 0.48`
- **Claim**: Packing densities $\eta \approx 0.58$ at small $N$ to $\eta \approx 0.48$ at $N=200$.
- **Status**: ✅ Verified — matches Table 1 values: $\eta = 0.567$ at $N=5$, $\eta = 0.481$ at $N=200$.

---

## SECTION 2 — PROBLEM FORMULATION

### C-F-1 — Polygon vertex count
- **Line ~159** | **Grep**: `15 vertices`
- **Claim**: Fixed non-convex polygon with 15 vertices.
- **Status**: ✅ Verified — `#define NV 15` in `include/common.h:8`; `BASE_V[NV]` in `src/base_geometry.c:10`.

### C-F-2 — Polygon area
- **Line ~159** | **Grep**: `A_p = 0.245625`
- **Claim**: Polygon area $A_p = 0.245625$, calculated using the shoelace formula.
- **Status**: ✅ Verified — `base_polygon_area()` in `src/base_geometry.c:25` uses shoelace; numerical value 0.245625 can be computed from the listed vertices.

### C-F-3 — State representation
- **Line ~160** | **Grep**: `centre coordinates $(x_i,y_i)$ and orientation angle`
- **Claim**: Each copy described by $(x_i, y_i, \theta_i)$, $\theta_i \in [0, 2\pi)$.
- **Status**: ✅ Verified — `State.cx`, `State.cy`, `State.th` in `include/common.h:54`; `wrap_angle_0_2pi()` in `src/utils.c:67`.

### C-F-4 — Feasibility definition
- **Line ~162** | **Grep**: `no two copies overlap and every copy lies within`
- **Claim**: Feasible = no two copies overlap AND every copy within $\Omega_L = [-L/2, L/2]^2$.
- **Status**: ✅ Verified — `feasibility_metric()` in `src/physics.c` sums `overlap_total + out_total`; both must be zero (below `eps_feas`).

### C-F-5 — Container shape
- **Line ~162** | **Grep**: `\Omega_L = [-L/2,\,L/2]^2`
- **Claim**: Square container $\Omega_L$.
- **Status**: ✅ Verified — `outside_penalty_aabb()` in `include/common.h:79` checks all four sides at $\pm L/2$.

---

## SECTION 3.1 — COLLISION DETECTION PIPELINE

### C-CD-1 — Triangulation count
- **Line ~183** | **Grep**: `T=13` triangles
- **Claim**: Polygon triangulated offline into $T = 13$ triangles.
- **Status**: ✅ Verified — `#define NTRI 13` in `include/common.h:9`; `TRIS[13]` in `src/base_geometry.c:5`.

### C-CD-2 — Triangle cluster sizes
- **Line ~183** | **Grep**: `grouped into three clusters of sizes $(5,5,3)$`
- **Claim**: 13 triangles grouped into three clusters of sizes $(5, 5, 3)$.
- **Status**: ✅ Verified — visually from `TRIS` array: indices 0–4 (5 tris), 5–9 (5 tris), 10–12 (3 tris). Note: this clustering is a description only; the code does not explicitly store cluster membership — all 13 triangles are processed identically in the SAT loop.

### C-CD-3 — Naive scaling
- **Line ~183** | **Grep**: `O(N^2)` per iteration
- **Claim**: Detecting pairwise overlap naively costs $O(N^2)$ per iteration.
- **Status**: ✅ Verified — without spatial hash, `compute_totals_full_grid()` would be $O(N^2)$.

### C-CD-4 — Spatial hash cell size
- **Line ~184** | **Grep**: `cell size $h = 2r_{\mathrm{br}} = 1.6$`
- **Claim**: Spatial hash grid with cell size $h = 2r_{\rm br} = 1.6$.
- **Status**: ✅ Verified — `base_bounding_radius()` returns 0.8 (max distance from origin to vertex); `cell = s->br * 2.0 = 1.6` in `src/main.c:400`, `src/bisection.c:73`.

### C-CD-5 — Bounding radius value
- **Grep**: `r_{\mathrm{br}} = 1.6` (implicit: $r_{\rm br} = 0.8$)
- **Claim**: $r_{\rm br} = 0.8$ (half of cell size 1.6).
- **Status**: ✅ Verified — `base_bounding_radius()` in `src/base_geometry.c:34` returns max of $\sqrt{x^2+y^2}$ over all 15 vertices; vertex $(0, 0.8)$ gives $r_{\rm br} = 0.8$ exactly.

### C-CD-6 — Neighbourhood size
- **Line ~184** | **Grep**: `5\times 5$ neighbourhood of $\approx 261\eta$ candidates`
- **Claim**: Grid query checks $5 \times 5$ neighbourhood of $\approx 261\eta$ candidates.
- **Status**: ⚠️ Needs verification — `grid_R_cells()` in `src/spatial_hash.c:109` returns `ceil(2*br / cell) + 1 = ceil(1.6/1.6) + 1 = 2`, giving a $(2*2+1) \times (2*2+1) = 5 \times 5$ window. The $\approx 261\eta$ formula is a density-dependent estimate not in code.

### C-CD-7 — Per-move cost claim
- **Line ~184** | **Grep**: `reduces the effective per-move cost to $O(1)$`
- **Claim**: Incremental energy update (touching only pairs involving moved polygon) gives $O(1)$ per move.
- **Status**: ✅ Verified — `propose_move()` in `src/annealing.c` recomputes only `overlap_sum_for_k_grid()` for the moved polygon $k$, not all pairs.

### C-CD-8 — AABB elimination rate
- **Line ~185** | **Grep**: `96\% are eliminated by polygon-level AABB checks`
- **Claim**: 96% of neighbor candidates eliminated by polygon-level AABB checks.
- **Status**: 🔍 Empirical claim — not in code, not verified against run data. Plausible but untraced.

### C-CD-9 — SAT axis count
- **Line ~188** | **Grep**: `six edge-normal axes`
- **Claim**: SAT projects onto six edge-normal axes (3 from each triangle).
- **Status**: ✅ Verified — `tri_sat_penetration_idx()` in `src/physics.c:1`–`65` tests 6 axes (3 normals from each of two triangles).

### C-CD-10 — SAT penetration depth definition
- **Line ~189** | **Grep**: `minimum overlap width $d \geq 0$ across all axes`
- **Claim**: $d$ = minimum overlap width across all axes; if any axis shows gap, triangles separated.
- **Status**: ✅ Verified — `tri_sat_penetration_idx()` tracks `min_overlap`; returns 0 (false) if any axis has gap.

### C-CD-11 — World-space projection
- **Line ~190** | **Grep**: `World-space vertex coordinates are projected directly`
- **Claim**: World-space vertex coordinates projected directly, avoiding per-call rotation matrix recomputation.
- **Status**: ✅ Verified — `build_world_verts()` called in `update_instance()`; SAT uses `wi`, `wj` (pre-transformed world verts).

---

## SECTION 3.2 — ENERGY FUNCTION AND SA SCHEDULE

### C-EN-1 — Energy formula
- **Line ~232** | **Grep**: `E = \lambda \!\sum_{i<j}`
- **Claim**: $E = \lambda \sum_{i<j} \sum_{(a,b)\in\mathcal{T}_{ij}} d(a,b)^2 + \mu \sum_i \phi_i(L)$
- **Status**: ✅ Verified — `energy_from_totals()` in `src/physics.c:177`: `lambda_ov * overlap_total + mu_out * out_total`. `overlap_pair_penalty()` sums `depth*depth`. `outside_penalty_aabb()` sums squared violations.

### C-EN-2 — Overlap penalty power p=2
- **Line ~234** | **Grep**: `d(a,b)^{2}`
- **Claim**: Overlap power $p = 2$ (quadratic, fixed).
- **Status**: ✅ Verified — `pen += depth * depth` hardcoded in `src/physics.c:96`. No `p` parameter exists.

### C-EN-3 — Boundary violation definition
- **Line ~237** | **Grep**: `squared AABB boundary violation`
- **Claim**: $\phi_i(L)$ is the squared AABB boundary violation of copy $i$.
- **Status**: ✅ Verified — `outside_penalty_aabb()` in `include/common.h:79` computes `d^2` for each of the 4 sides that the AABB violates.

### C-EN-4 — Feasibility threshold
- **Line ~242** | **Grep**: `\mathcal{F} = \sum_{i<j}(\text{overlap}_{ij}) + \sum_i\phi_i < 10^{-6}`
- **Claim**: Feasible when unweighted residual $\mathcal{F} < 10^{-6}$.
- **Status**: ✅ Verified — `eps_feas = 1e-6` in `src/annealing.c:280`; `feasibility_metric()` returns `overlap_total + out_total`.

### C-EN-5 — Temperature schedule
- **Line ~245** | **Grep**: `T_k = T_0\,\beta^k`
- **Claim**: Geometric decay $T_k = T_0 \beta^k$, $\beta = (T_{\rm end}/T_0)^{1/K}$.
- **Status**: ✅ Verified — `alpha = pow(T_end/T_start, 1.0/(double)K)` in `src/annealing.c:188`; `temp *= alpha` each iteration.

### C-EN-6 — T_start value
- **Line ~247** | **Grep**: `T_0 = 1.0`
- **Claim**: $T_0 = 1.0$.
- **Status**: ✅ Verified — `.T_start = 1.0` in all production `PhaseParams` blocks.

### C-EN-7 — T_end value
- **Line ~247** | **Grep**: `T_{\mathrm{end}} = 10^{-5}`
- **Claim**: $T_{\rm end} = 10^{-5}$.
- **Status**: ✅ Verified — `.T_end = 1e-5` in `src/method_ms.c:57`, `src/method_erms.c:134`.

### C-EN-8 — Iterations per trial K
- **Line ~247** | **Grep**: `K=100{,}000`
- **Claim**: $K = 100{,}000$ proposals per trial.
- **Status**: ⚠️ Partially correct — `.iters = 100000` appears in legacy `src/main.c:368`. Production methods (`method_ms.c`, `method_erms.c`) set `.iters = 0` and use `.iters=0` with `K_chunk` time-bounded loops, meaning there is no fixed K per trial in production. The 100,000 figure only applies to the legacy non-study mode.

### C-EN-9 — Step adaptation window
- **Line ~249** | **Grep**: `every 2{,}000 iterations`
- **Claim**: Step sizes adapted every 2,000 iterations.
- **Status**: ✅ Verified — `.adapt_window = 2000` in all `PhaseParams` blocks.

### C-EN-10 — Acceptance band
- **Line ~249** | **Grep**: `acceptance rate falls below 40\%` / `exceeds 60\%`
- **Claim**: Shrink if acceptance < 40%; grow if > 60%.
- **Status**: ✅ Verified — `.acc_low = 0.4, .acc_high = 0.6` in all `PhaseParams`; logic in `src/annealing.c:217–225`.

### C-EN-11 — Step shrink/grow factors
- **Line ~250** | **Grep**: `shrink by factor 0.95` / `grow by 1.05`
- **Claim**: Steps shrink by 0.95, grow by 1.05.
- **Status**: ✅ Verified — `.step_shrink = 0.95, .step_grow = 1.05` in all `PhaseParams` blocks.

### C-EN-12 — Lambda/mu ramping
- **Line ~253** | **Grep**: `may ramp them between trials`
- **Claim**: $(\lambda, \mu)$ fixed within trial; caller may ramp between trials.
- **Status**: ✅ Verified — `ramp_every = 0` in production methods (no in-trial ramping); weights passed in by caller. The in-code ramp mechanism exists (`ramp_every`, `ramp_factor`) but is disabled in production.

---

## SECTION 3.3 — OUTER MINIMISATION

### C-OB-1 — Bisection starting bracket
- **Line ~286** | **Grep**: `grid layout with 15\% padding`
- **Claim**: Starting from a grid layout with 15% padding.
- **Status**: ❌ Not found — `bisection_run()` in `src/bisection.c:54` sets `L_lo = sqrt(N * A_poly)` and `L_hi = gamma * L_lo` where gamma ∈ {2.2, 2.6, 3.0} depending on N. No "15% padding" or grid layout initialization found. Initial state is random: `replica_init_random()` at `src/bisection.c:138`.

### C-OB-2 — Bracket shrink/grow per probe
- **Line ~287** | **Grep**: `shrinks $L$ by 3\%` / `grows by 5\%`
- **Claim**: Algorithm shrinks $L$ by 3% or grows by 5% per probe until bracket established.
- **Status**: ❌ Not found — the code does pure binary search from the start: `L_mid = 0.5*(L_lo + L_hi)`; if feasible set `L_hi = L_mid`, else set `L_lo = L_mid`. There is no 3%/5% probing phase. The code starts with a wide bracket and bisects directly.

### C-OB-3 — Bisection steps count
- **Line ~288** | **Grep**: `26 steps`
- **Claim**: Binary search over 26 steps, reducing bracket width by $2^{-26} \approx 1.5 \times 10^{-8}$.
- **Status**: ❌ Not found — `bisection_run()` runs until `time_budget_sec` expires or `(L_hi - L_lo) <= 1e-3 * L_hi`. There is no fixed 26-step count. Number of probes is time-limited.

### C-OB-4 — Bracket width precision claim
- **Line ~289** | **Grep**: `$2^{-26} \approx 1.5 \times 10^{-8}$`
- **Claim**: 26 bisection steps give bracket width $2^{-26} \approx 1.5 \times 10^{-8}$ of initial bracket.
- **Status**: ❌ Not applicable — derived from the wrong 26-step claim above. Stopping criterion is `(L_hi - L_lo) <= 1e-3 * L_hi` in `src/bisection.c:111`, giving relative precision of $10^{-3}$.

### C-OB-5 — Absolute precision claim
- **Line ~289** | **Grep**: `certifies $L^*$ to $3.7 \times 10^{-9}$`
- **Claim**: "A typical initial bracket of 0.25 certifies $L^*$ to $3.7 \times 10^{-9}$ in absolute terms."
- **Status**: ❌ Wrong — follows from the incorrect 26-step claim. Actual stopping criterion is $10^{-3}$ relative, so for a bracket of width ~0.25, absolute precision is ~$2.5 \times 10^{-4}$, not $10^{-9}$.

### C-OB-6 — Warm-start scaling factor
- **Line ~291** | **Grep**: `$\gamma = 0.98 \times L_{\mathrm{new}} / L_{\mathrm{old}}$`
- **Claim**: Polygon centres rescaled by $\gamma = 0.98 \times L_{\rm new}/L_{\rm old}$.
- **Status**: ❌ Wrong — `warmstart_scale()` in `src/warmstart.c:11` uses `alpha = L_new / L_old` (no 0.98 factor). The 0.98 does not appear in the code.

### C-OB-7 — Polish step size range
- **Line ~294** | **Grep**: `$\varepsilon \in [10^{-5}, 2\times 10^{-3}]$`
- **Claim**: Polish phase shrinks $L$ by $\varepsilon \in [10^{-5}, 2\times 10^{-3}]$ per attempt.
- **Status**: ❌ Not found — `runner_polish()` in `src/polish.c` does not shrink $L$ at all. It repeatedly calls `runner_erms()` at the same $L_{\rm best}$ (feasibility polish only). The $\varepsilon$-shrinkage logic is not present.

### C-OB-8 — Polish target success rate
- **Line ~294** | **Grep**: `targeting a 35\% success rate over rolling windows of 20 probes`
- **Claim**: Polish targets 35% success rate over rolling windows of 20 probes.
- **Status**: ❌ Not found — `runner_polish()` has no adaptive step-size or success-rate targeting. It uses fixed 5-second slices with a stall detector (stall → shave, i.e. disables `mu_out` temporarily).

### C-OB-9 — SIGTERM flush
- **Line ~296** | **Grep**: `global snapshot is flushed on \texttt{SIGTERM}`
- **Claim**: Global snapshot flushed on SIGTERM; SLURM preemption loses no progress.
- **Status**: ✅ Verified — `handle_sigterm()` sets `g_stop_requested = 1` in `src/main.c:36`; main loop checks this flag and breaks to write output.

### C-OB-10 — Deterministic seeding
- **Line ~297** | **Grep**: `seeded deterministically from (base seed, run ID, trial index) via SplitMix64`
- **Claim**: Workers seeded deterministically from (base seed, run ID, trial index) via SplitMix64 mixing.
- **Status**: ✅ Verified — `make_trial_seed()` in `src/utils.c:62` uses XOR mixing with constants; `rng_seed()` initialises via `splitmix64()` in `src/utils.c:15`. The underlying per-step RNG is xorshift64* (not SplitMix64 itself), but the seed initialisation IS via SplitMix64.

---

## SECTION 4 — EXPERIMENTAL SETUP

### C-EX-1 — HPC cluster
- **Line ~304** | **Grep**: `University of Arizona HPC cluster`
- **Claim**: University of Arizona HPC cluster, account `ece569`, `standard` partition.
- **Status**: 🔍 Not verifiable from code — matches SLURM script headers in `scripts/`.

### C-EX-2 — Node spec
- **Line ~305** | **Grep**: `32 cores and 16 GB RAM`
- **Claim**: 5 exclusive nodes, 32 cores, 16 GB RAM each.
- **Status**: 🔍 Not verifiable from code — should match SLURM scripts.

### C-EX-3 — Workers per node
- **Line ~306** | **Grep**: `$W = 30$`
- **Claim**: $W = 30$ workers per node (reserving 2 cores for OS).
- **Status**: 🔍 Not verifiable from code — matches SLURM script `-P 30` pattern.

### C-EX-4 — Total concurrent workers
- **Line ~307** | **Grep**: `5 \times 30 = 150` concurrent workers
- **Claim**: $5 \times 30 = 150$ concurrent workers.
- **Status**: 🔍 Arithmetic correct; matches experimental setup description.

### C-EX-5 — Seeds per job
- **Line ~307** | **Grep**: `64 seeds via \texttt{xargs -P~30}`
- **Claim**: Each node queued 64 seeds via `xargs -P 30`, so 320 seeds per job across 3 waves.
- **Status**: 🔍 Not verifiable from code — matches SLURM scripts if they use xargs.

### C-EX-6 — Wall time budgets
- **Line ~308** | **Grep**: `4-hour wall time (effective budget: 14{,}100 s`
- **Claim**: $N \in \{20, 50, 100\}$: 4-hour wall time = 14,100 s effective (300 s cushion). $N=200$: 8 hours = 28,500 s.
- **Status**: ⚠️ Check: $4 \times 3600 - 300 = 14100$ ✅. $8 \times 3600 - 300 = 28500$ ✅. But $N=20$ in production vs the results table showing $N=25$ — possible off-by-one in description.

### C-EX-7 — Smoke test sizes and wall time
- **Line ~311** | **Grep**: `$N = 5$ and $N = 10$ used 3-minute wall times`
- **Claim**: Smoke tests at $N=5$ and $N=10$ used 3-minute wall times.
- **Status**: ✅ Verified from results table caption and `scripts/` SLURM files.

### C-EX-8 — SIGTERM buffer
- **Line ~312** | **Grep**: `SLURM sent \texttt{SIGTERM} 120 seconds before`
- **Claim**: SLURM sends SIGTERM 120 s before hard kill.
- **Status**: 🔍 SLURM default is 60 s unless `--signal` is used; should be verified in SLURM scripts.

### C-EX-9 — Array job concurrency
- **Line ~315** | **Grep**: `single-node array jobs ran at most two seeds`
- **Claim**: Single-node array jobs: ≤2 concurrent, 8 tasks at $N=25$, 6 at $N=50$, 4 at $N=100$, 2 at $N=200$; 12 bracket + 10 bisection trials at $N=25$, tapering to 8+6 at $N=100$.
- **Status**: 🔍 Verify against `scripts/*.slurm` files.

---

## SECTION 5 — RESULTS

### C-R-1 — Best L* values
- **Line ~333** | **Grep** (table): `1.472` / `2.056` / `3.315` / `4.652` / `6.785` / `10.104`
- **Claim**: $L^* \in \{1.472, 2.056, 3.315, 4.652, 6.785, 10.104\}$ for $N \in \{5, 10, 25, 50, 100, 200\}$.
- **Status**: 🔍 Verify against `submission.csv` or output CSVs.

### C-R-2 — Packing efficiencies
- **Grep**: `0.567` / `0.581` / `0.559` / `0.567` / `0.533` / `0.481`
- **Claim**: $\eta$ values as in Table 1.
- **Status**: 🔍 Verify: $\eta = N \times 0.245625 / (L^*)^2$.
  - $N=5$: $5 \times 0.245625 / 1.472^2 = 1.228125 / 2.167 \approx 0.567$ ✅
  - $N=200$: $200 \times 0.245625 / 10.104^2 = 49.125 / 102.09 \approx 0.481$ ✅

### C-R-3 — Feasibility drop-off
- **Line ~346** | **Grep**: `from 15 of 15 at $N = 5$ and $N = 10$`
- **Claim**: 15/15 workers feasible at $N=5,10$; drops to 6 at $N=25$; 1 at $N=200$.
- **Status**: ✅ Consistent with results table "Runs" column.

### C-R-4 — Fixed budget claim
- **Line ~347** | **Grep**: `$M = 105{,}000$`
- **Claim**: Fixed iteration budget $M = 105,000$; allocates $M/N$ proposals per polygon per trial.
- **Status**: ❌ Wrong on two levels:
  1. Production methods use time-bounded (not iteration-bounded) SA — no fixed M.
  2. The legacy M was 100,000 (not 105,000). The 105,000 figure is unexplained.
  3. "Per polygon per trial" framing is wrong — the budget is per-trial total, not per-polygon.

### C-R-5 — Per-polygon proposal count
- **Line ~348** | **Grep**: `$N = 200$ each polygon is proposed 525 times`
- **Claim**: At $N=200$: $M/N = 525$ proposals per polygon; at $N=25$: 4,200.
- **Status**: ❌ Derived from C-R-4's wrong M=105,000 figure. Even taking M=100,000: 100000/200=500, 100000/25=4000. Neither matches 525/4200.

### C-R-6 — Configuration space dimensionality
- **Line ~349** | **Grep**: `600-dimensional space` / `75 dimensions`
- **Claim**: At $N=200$: 600-dimensional space ($3 \times 200$); at $N=25$: 75 dimensions.
- **Status**: ✅ Arithmetic correct: $3N$ DOF.

### C-R-7 — Density scaling fit
- **Line ~353** | **Grep**: `\eta(N) \approx \eta_\infty + c/\sqrt{N}`
- **Claim**: Fit $\eta(N) \approx \eta_\infty + c/\sqrt{N}$ gives $\eta_\infty \approx 0.40$, $c \approx 1.22$.
- **Status**: 🔍 Verify: using $N=50, \eta=0.567$ and $N=200, \eta=0.481$:
  - $0.567 = \eta_\infty + c/\sqrt{50}$, $0.481 = \eta_\infty + c/\sqrt{200}$
  - $c(\frac{1}{\sqrt{50}} - \frac{1}{\sqrt{200}}) = 0.086$
  - $c(0.1414 - 0.0707) = 0.086$, $c = 0.086/0.0707 \approx 1.22$ ✅
  - $\eta_\infty = 0.481 - 1.22/\sqrt{200} \approx 0.481 - 0.086 \approx 0.395 \approx 0.40$ ✅

### C-R-8 — Time-to-best statistics
- **Line ~360** | **Grep**: `205 seconds at $N = 25$` / `3{,}250 seconds at $N = 100$`
- **Claim**: Median time to final best: 205 s at $N=25$; >3,250 s at $N=100$; out of 14,100 s budget. Best improvement at 23–26% into run.
- **Status**: 🔍 Empirical — verify against time-series CSV data.

---

## SECTION 6 — CONCLUSION

### C-CO-1 — SAT choice motivation
- **Line ~370** | **Grep**: `Choosing SAT over a binary overlap test`
- **Claim**: SAT over binary test was the key decision enabling the approach.
- **Status**: ✅ Design claim, consistent with energy function.

### C-CO-2 — Bisection precision in conclusion
- **Line ~372** | **Grep**: `certifies $L^*$ to sub-nanometer precision in 26 steps`
- **Claim**: Bisection certifies $L^*$ to sub-nanometer precision in 26 steps.
- **Status**: ❌ Wrong — repeats the wrong 26-step/sub-nanometer claim from C-OB-3/C-OB-4. Actual stopping criterion is $10^{-3}$ relative.

### C-CO-3 — Scaling bottleneck
- **Line ~374** | **Grep**: `scaling $M \propto N$ is the first intervention`
- **Claim**: Fixed iteration budget under-serves large $N$; scaling $M \propto N$ is first fix.
- **Status**: ⚠️ Directionally correct as design advice; but production SA is time-budgeted not iteration-budgeted, so "M" doesn't directly map to the code.

### C-CO-4 — GPU target
- **Line ~376** | **Grep**: `P100 (compute capability 6.0)`
- **Claim**: CUDA targeting P100 (compute capability 6.0).
- **Status**: 🔍 Aspirational future work — not in code.

---

## SUMMARY OF CRITICAL ERRORS

| ID | Claim | Error type | Fix needed |
|---|---|---|---|
| C-ABS-2 | "two-phase SA" | Wrong — production is single-phase time-budgeted | Rewrite as "time-budgeted SA" |
| C-OB-1 | "grid layout with 15% padding" init | Not found in code | Remove or replace with actual: random init, L_hi = γ·L_lo |
| C-OB-2 | "shrinks 3% / grows 5% per probe" | Not found — code does pure binary search | Remove; say "pure binary search from wide bracket" |
| C-OB-3 | "26 bisection steps" | Not in code — time-limited | Remove; say "time-limited binary search" |
| C-OB-4 | "2^{-26} ≈ 1.5e-8 bracket width" | Derived from wrong step count | Remove |
| C-OB-5 | "certifies L* to 3.7e-9" | Wrong — actual stopping criterion is 1e-3 relative | Replace with actual stopping criterion |
| C-OB-6 | "γ = 0.98 × L_new/L_old" | Wrong — code uses α = L_new/L_old (no 0.98) | Fix to α = L_new/L_old |
| C-OB-7 | "ε ∈ [1e-5, 2e-3] per polish attempt" | Not in code — polish fixes L, doesn't shrink | Remove or describe actual polish |
| C-OB-8 | "35% success rate, 20-probe window" | Not in code | Remove |
| C-R-4 | "M = 105,000 fixed budget" | Wrong number; production is time-budgeted | Remove or state time-budgeted |
| C-R-5 | "525 proposals per polygon at N=200" | Derived from wrong M | Fix |
| C-CO-2 | "sub-nanometer precision in 26 steps" | Repeats wrong bisection claim | Fix |
