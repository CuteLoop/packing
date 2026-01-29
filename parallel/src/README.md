# `src/` — Simulated Annealing Circle Packing (Naive C Baseline)

This directory contains a minimal, correctness-first implementation of simulated annealing (SA)
for packing identical circles in a 2D square container of side length `L`.

The priority is **trustworthiness**:
- deterministic reproducibility via an explicit RNG and seed,
- simple modules matching the theory,
- unit tests for invariants,
- full instrumentation via step-by-step traces.

## Theory → Code mapping

State:
- `X ∈ R^{N×2}`: circle centers
- `L > 0`: square side length (fixed during a run in this baseline)

Energy:
$$
E(X,L) = E_{\\text{pair}}(X) + E_{\\text{wall}}(X,L) + \\alpha L
$$
where

Pair overlap penalty:
$$
E_{\\text{pair}}(X) = \\sum_{i<j} \\max(0,2r-\\|x_i-x_j\\|)^2
$$

Wall penalty:
$$
E_{\\text{wall}}(X,L)=\\sum_i\\left(\\max(0,r-x_i)^2+\\max(0,x_i-(L-r))^2+\\max(0,r-y_i)^2+\\max(0,y_i-(L-r))^2\\right).
$$

Metropolis acceptance at temperature `T`:
- accept always if `ΔE <= 0`
- accept with probability `exp(-ΔE/T)` if `ΔE > 0`

Cooling schedule:
- geometric: `T <- gamma * T`, with `0 < gamma < 1`

## Modules

### `rng.c` / `rng.h`
Deterministic randomness:
- `rng_u01()` returns uniform `U ∈ [0,1)`
- `rng_normal()` returns standard normal `N(0,1)` (Box–Muller)
- same seed ⇒ identical sequences ⇒ reproducible SA traces.

### `energy.c` / `energy.h`
Energy model:
- `energy_pair(X,N,r)` : O(N²) overlap energy
- `energy_wall(X,N,r,L)` : O(N) wall energy
- `energy_total(X,N,r,L,alpha)` : total energy

Incremental move update:
- `delta_energy_move_one(X,N,i,new_pos,r,L)` computes the exact `ΔE`
  when moving one circle `i` to `new_pos`, holding `L` fixed.
  This supports O(N) updates in the SA inner loop.

### `propose.c` / `propose.h`
Proposal kernel:
- choose exactly one index uniformly
- step `δ ~ N(0, sigma² I)`
- candidate is clipped to `[0,L]²` to prevent numerical blow-up.

### `metropolis.c` / `metropolis.h`
Metropolis accept rule:
- downhill moves always accepted
- uphill moves accepted with probability `exp(-ΔE/T)`.

### `anneal.c` / `anneal.h`
Annealing driver:
- propose → compute `ΔE` → accept/reject → update state → record trace
- tracks a **best-so-far** configuration (best energy is monotone).

### `io.c` / `io.h`
Output utilities:
- trace CSV writer
- best centers CSV writer
- best configuration SVG writer.

## Outputs and Trace Format

The runner writes:

- `<prefix>_trace.csv` : per-step diagnostics
- `<prefix>_best.csv`  : best configuration centers
- `<prefix>_best.svg`  : visualization of best configuration

### Trace CSV columns

`*_trace.csv` includes:

- `step` : integer step index
- `E` : total energy at that step
- `T` : temperature at that step
- `E_pair` : pair overlap energy at that step (diagnostic)
- `E_wall` : wall penalty energy at that step (diagnostic)
- `accepted` : 0/1 whether the proposal was accepted
- `moved` : which circle index was proposed/moved

**Logging frequency:** one row per Metropolis step (no subsampling).

### Notes on diagnostics

`E_pair` and `E_wall` are logged for interpretability and plotting.
They are computed explicitly from the current state (clarity-first);
this can be optimized later if needed.

## Build

From repo root:

```bash
make
make sa_run
```

Run a smoke test via your `run/` scripts, then visualize traces with the plotting script.

## Intentional baseline limitations

* No optimization over `L` during a run (fixed `L`)
* No adaptive tuning of `sigma`
* No spatial hashing / neighbor lists
* Trace logging is full-resolution (can be decimated later)
