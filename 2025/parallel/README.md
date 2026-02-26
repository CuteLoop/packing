# `src/` — Simulated Annealing Circle Packing (Naive C Baseline)

This directory contains a minimal, correctness-first implementation of simulated annealing (SA) for packing identical circles in a 2D square of side length `L`.

The goal of this baseline is **trustworthiness**, not speed:
- each theoretical component from the SA model is implemented in an explicit module;
- the code supports deterministic reproducibility via an explicit RNG;
- diagnostics (energy, temperature, accept/reject decisions) are recorded step-by-step;
- the implementation is intended as a clean starting point for later optimization (OpenMP/CUDA, spatial hashing, etc.).

## Model (Theory → Code)

We model a state as:
- `X ∈ R^{N×2}`: circle centers
- `L > 0`: container side length (fixed during a run in the baseline)

Energy:
\[
E(X,L) = E_\text{pair}(X) + E_\text{wall}(X,L) + \alpha L
\]
with overlap penalty
\[
E_\text{pair}(X) = \sum_{i<j} \max(0,2r-\|x_i-x_j\|)^2
\]
and wall penalty
\[
E_\text{wall}(X,L)=\sum_i\left(\max(0,r-x_i)^2+\max(0,x_i-(L-r))^2+\max(0,r-y_i)^2+\max(0,y_i-(L-r))^2\right).
\]

Metropolis acceptance at temperature `T`:
- accept always if `ΔE <= 0`;
- accept with probability `exp(-ΔE/T)` otherwise.

Cooling schedule:
- geometric: `T <- gamma * T`, `gamma ∈ (0,1)`.

## Files and Responsibilities

### `rng.c` / `rng.h`
Deterministic pseudorandom number generation:
- `rng_u01()` gives uniform `U ∈ [0,1)`.
- `rng_normal()` gives standard normal `N(0,1)` via Box–Muller with caching.
- determinism: same seed → same trajectory.

Design notes:
- We use a small xorshift-based generator (fast, deterministic, minimal dependencies).
- We keep the RNG as an explicit object passed into all stochastic components.

### `energy.c` / `energy.h`
Implements the energy model:
- `energy_pair(X,N,r)` computes the pairwise overlap penalty (O(N²)).
- `energy_wall(X,N,r,L)` computes the wall penalty (O(N)).
- `energy_total(X,N,r,L,alpha)` adds penalties plus `alpha*L`.

Incremental update:
- `delta_energy_move_one(X,N,i,new_pos,r,L)` computes the exact energy change when
  moving **one** circle `i` to `new_pos` while keeping `L` fixed (O(N)).
  This supports an SA inner loop that is O(N) per proposal.

Important:
- `alpha*L` cancels in `delta_energy_move_one` because `L` is unchanged during a move.
- The baseline assumes identical radii for all circles.

### `propose.c` / `propose.h`
Proposal kernel:
- picks exactly one circle index uniformly;
- adds Gaussian step `N(0, sigma^2 I)`;
- clips the candidate into the bounding box `[0, L]×[0, L]` to prevent numerical blow-up.

Design notes:
- Clipping is a pragmatic baseline choice; later versions may use reflection,
  periodic boundaries (for experiments), or reject-outside policies.

### `metropolis.c` / `metropolis.h`
Acceptance rule:
- `metropolis_accept(dE, T, rng)` implements the Metropolis criterion.

Edge behavior:
- `T <= 0` rejects all uphill moves (and always accepts downhill moves).

### `anneal.c` / `anneal.h`
Annealing driver:
- runs `n_steps` iterations of propose → ΔE → accept/reject → record trace;
- stores an energy trace `E[t]`, temperature trace `T[t]`, accept decisions, and moved indices;
- tracks a **best-so-far** configuration (monotone best energy).

Return type:
- `AnnealResult` includes:
  - `X_best`, `E_best`,
  - `accept_rate`,
  - `Trace` arrays for debugging and plotting.

Memory ownership:
- `anneal_run()` allocates result buffers.
- caller must call `anneal_free_result()`.

## Build and Use

From repository root sa_circlepack/ :

```bash
make
make test
```