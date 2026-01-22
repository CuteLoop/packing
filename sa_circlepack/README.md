# `src/` — Simulated Annealing Circle Packing (Naive C Baseline)

This directory contains a **minimal, correctness-first implementation** of simulated annealing (SA) for packing identical circles in a 2D square of fixed side length `L`.

The goal of this baseline is **trustworthiness, not speed**:

* each theoretical component of simulated annealing is implemented explicitly;
* stochasticity is fully deterministic under a fixed seed;
* all relevant diagnostics are logged step-by-step;
* the code is structured to support inquiry, testing, and later optimization
  (OpenMP, CUDA, spatial hashing, etc.).

This implementation is intentionally *naive* and *transparent*. Performance
optimizations are deferred until after correctness and observability are established.

---

## Model (Theory → Code)

### State

A state consists of:

* `X ∈ R^{N×2}`: circle centers,
* `L > 0`: container side length (fixed during a run in this baseline).

All circles have identical radius `r`.

---

### Energy

The total energy is

[
E(X,L) = E_\text{pair}(X) + E_\text{wall}(X,L) + \alpha L,
]

where:

**Pairwise overlap penalty**
[
E_\text{pair}(X)
= \sum_{i<j} \max(0, 2r - |x_i - x_j|)^2
]

**Wall penalty**
[
E_\text{wall}(X,L)
= \sum_i \Bigl[
\max(0, r - x_i)^2

* \max(0, x_i - (L - r))^2
* \max(0, r - y_i)^2
* \max(0, y_i - (L - r))^2
  \Bigr].
  ]

Properties enforced by construction and tested explicitly:

* `E ≥ 0` always;
* feasible configurations (no overlaps, fully inside walls) satisfy
  `E_pair = E_wall = 0`.

---

### Metropolis Acceptance

Given a proposal with energy change `ΔE` at temperature `T`:

* accept unconditionally if `ΔE ≤ 0`;
* accept with probability `exp(-ΔE / T)` otherwise.

Edge behavior:

* if `T ≤ 0`, all uphill moves are rejected.

---

### Cooling Schedule

A **geometric cooling schedule** is used:

[
T_{t+1} = \gamma T_t, \qquad \gamma \in (0,1).
]

Thus:
[
T_t = T_0 \gamma^t.
]

This is not asymptotically optimal, but is sufficient and transparent for a baseline.

---

## Files and Responsibilities

### `rng.c` / `rng.h`

Deterministic pseudorandom number generation.

Provided functions:

* `rng_u01()` → uniform `U ∈ [0,1)`,
* `rng_normal()` → standard normal `N(0,1)` via Box–Muller (with caching).

Design notes:

* small xorshift-style generator;
* no global state;
* RNG object is passed explicitly to all stochastic components;
* **same seed ⇒ same trajectory**, enforced by tests.

---

### `energy.c` / `energy.h`

Implements the energy model.

Functions:

* `energy_pair(X, N, r)`
  Computes pairwise overlap penalty (O(N²)).

* `energy_wall(X, N, r, L)`
  Computes wall penalty (O(N)).

* `energy_total(X, N, r, L, alpha)`
  Full energy (used mainly for validation and diagnostics).

* `delta_energy_move_one(X, N, i, new_pos, r, L)`
  Computes the **exact energy change** when moving a *single* circle `i`
  to `new_pos` while keeping all others fixed (O(N)).

Important notes:

* `alpha * L` cancels in `delta_energy_move_one` because `L` is fixed;
* all circles are assumed to have identical radius `r`;
* incremental energy updates are central to making the SA inner loop O(N).

---

### `propose.c` / `propose.h`

Proposal kernel.

Behavior:

* choose exactly one circle index uniformly;
* propose a Gaussian step `δ ~ N(0, σ² I)` in 2D;
* clip the proposed position into `[0, L] × [0, L]`.

Design notes:

* clipping is a pragmatic baseline choice to avoid numerical blow-up;
* later variants may use reflection, rejection, or more geometric proposals;
* locality + randomness ensure ergodicity.

---

### `metropolis.c` / `metropolis.h`

Acceptance rule:

* `metropolis_accept(dE, T, rng)` implements the Metropolis criterion.

The function is pure and side-effect free aside from RNG usage.

---

### `anneal.c` / `anneal.h`

Annealing driver.

Responsibilities:

* run `n_steps` iterations of:
  propose → compute `ΔE` → accept/reject → update state;
* apply geometric cooling at every step;
* maintain a **best-so-far** configuration with monotone best energy;
* record a complete diagnostic trace.

#### Trace logging

At **every Metropolis step**, the following are logged:

* `step`
* `E`        — total energy
* `E_pair`   — pairwise penalty (diagnostic)
* `E_wall`   — wall penalty (diagnostic)
* `T`        — temperature
* `accepted` — 0/1
* `moved`    — index of moved circle

Logging is intentionally dense (one row per step) to support
debugging, invariants, and post-hoc analysis.

#### Return type

`anneal_run()` returns an `AnnealResult` containing:

* `X_best`, `E_best`,
* overall acceptance rate,
* a `Trace` struct with all logged arrays.

Memory ownership:

* `anneal_run()` allocates all buffers;
* caller **must** call `anneal_free_result()`.

---

## Diagnostics and Plotting

The trace CSV (`*_trace.csv`) is designed to support standard diagnostics:

1. **Energy vs step** (with best-so-far)
2. **Temperature vs step** (log scale)
3. **Rolling acceptance rate**
4. **Penalty decomposition** (`E_pair` vs `E_wall`)
5. **ΔE distribution**
6. **Moved-index histogram**

A companion plotting script (`tools/plot_trace.py`) generates:

* a **2×2 main diagnostic figure**:

  * Energy,
  * Temperature,
  * Acceptance rate,
  * Penalty decomposition;
* separate histograms for:

  * `ΔE`,
  * moved index.

Plots are designed to be:

* readable at long runs (downsampling applied),
* minimal in labeling,
* suitable for inclusion in LaTeX reports.

---

## Build and Test

From the repository root (`sa_circlepack/`):

```bash
make
make test
```

All tests are deterministic and correctness-oriented, enforcing:

* RNG reproducibility,
* energy non-negativity and symmetry,
* correctness of incremental `ΔE`,
* Metropolis acceptance behavior,
* best-so-far monotonicity.

---

## Design Philosophy (Why This Looks “Verbose”)

This code prioritizes:

* explicitness over cleverness,
* invariants over speed,
* diagnostics over minimalism.

Every array, log, and function exists to make the **theory observable in code**.
Once correctness and understanding are locked in, performance work becomes
safe, local, and mechanical.

---

If you want next steps, natural continuations are:

* adaptive proposal scaling,
* outer-loop optimization of `L`,
* spatial acceleration of pair energies,
* OpenMP parallelism,
* CUDA kernels for energy evaluation.

But this baseline is intentionally complete *before* any of that.