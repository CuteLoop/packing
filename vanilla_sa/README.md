Below is a compact, practical documentation block you can drop into `docs/vanilla_sa.md` (or as a header comment in `sa_pack.c`). It is written to match how the code actually behaves.

---

# `vanilla_sa` / `sa_pack.c` — Simulated Annealing Disk Packing (Square)

## Purpose

This program uses **simulated annealing (SA)** to place **N circles** (disks) of fixed radii inside a **square container of fixed side length `L`**, minimizing constraint violations (overlaps and boundary breaches). It is designed as a **fast baseline** with a few high-return upgrades:

* **Incremental energy updates**: (O(N)) per move (only pairs involving the moved circle).
* **Adaptive step size**: acceptance-rate controller over a sliding window.
* **Two-phase schedule**: exploratory search followed by feasibility enforcement.
* **Multi-start trials**: many medium runs rather than one long run.
* **Teleport / reinsert moves**: occasional global jumps to escape local minima.

The code is currently configured for **feasibility** (not optimizing `L`), but contains a hook (`alpha_L`) for length optimization experiments.

---

## Build and Run

### Compile

```bash
gcc -O2 -std=c11 -Wall -Wextra -pedantic sa_pack.c -o sa_pack -lm
```

### Run

```bash
./sa_pack
```

Artifacts produced in the working directory:

* `best_circles.csv` — best configuration found (by feasibility score)
* `best.svg` — visualization of that configuration

---

## Problem Definition

### State (`State`)

* `N`: number of circles
* `L`: side length of the square (fixed in this demo)
* `x[i], y[i]`: center coordinates of disk `i`
* `r[i]`: radius of disk `i`

The container is the axis-aligned square:
[
x, y \in \left[-\frac{L}{2}, \frac{L}{2}\right]
]
Disk `i` is feasible w.r.t. walls if:
[
|x_i| \le \frac{L}{2} - r_i,\quad |y_i| \le \frac{L}{2} - r_i
]

Disk pair `(i,j)` is non-overlapping if:
[
|c_i - c_j| \ge r_i + r_j
]

---

## Objective (Energy)

The annealer evaluates an energy
[
E = \alpha_L L + \lambda_{ov}, \text{OV} + \mu_{out}, \text{OUT}
]

Where:

* `OV` is the sum over all pairs `i<j` of squared overlap depth:

  * overlap depth = ((r_i + r_j) - d_{ij}) if disks overlap, else 0
  * contribution = overlap_depth²
* `OUT` is the sum over disks of squared boundary violation:

  * if `|x_i|` exceeds allowable `lim = L/2 - r_i`, add ((|x_i|-lim)^2)
  * same for `y_i`

### Feasibility metric

The code tracks “best” solutions using:
[
\text{feas} = \text{OV} + \text{OUT}
]
This is **unweighted**, so it stays comparable even when penalty weights ramp.

---

## Key Implementation Notes

### Incremental bookkeeping (`Totals`)

`Totals` stores running sums:

* `overlap_total = Σ_{i<j} overlap_pair(i,j)`
* `out_total = Σ_i outside_for_k(i)`

During a move of disk `k`, the program recomputes only:

* all overlaps involving `k` (sum over `j != k`)
* the boundary penalty of `k`

This yields (O(N)) per move.

---

## Move Set

Moves are applied to one randomly chosen disk `k`:

1. **Local jiggle** (most moves)

   * `x[k] += U(-step, step)`
   * `y[k] += U(-step, step)`

2. **Teleport / reinsert** (with probability `p_reinsert`)

   * re-sample `x[k], y[k]` uniformly inside the wall-feasible region for that disk

If a move is rejected, the state and totals are rolled back via `undo_move`.

---

## Acceptance Rule (Metropolis)

Given energy change (\Delta E = E_{new} - E):

* Always accept if (\Delta E \le 0)
* Else accept with probability:
  [
  \exp(-\Delta E / T)
  ]

`T` cools exponentially from `T_start` to `T_end` over each phase.

---

## Two-Phase Schedule

The program runs **two SA phases per trial**:

### Phase A: Explore

* Higher temperature
* Larger starting step
* Low teleport rate (but nonzero)
* Penalty ramp typically off (or mild)

### Phase B: Enforce feasibility

* Reheats slightly, then cools lower than Phase A
* Smaller step size targets constraint cleanup
* Penalties ramp aggressively (`ramp_every`, `ramp_factor`) up to `lambda_max`, `mu_max`
* Lower teleport rate

The best state is tracked by `feas` across both phases.

---

## Adaptive Step Size

Every `adapt_window` moves, the acceptance rate in that window is computed:

* if `acc < acc_low`: `step *= step_shrink`
* if `acc > acc_high`: `step *= step_grow`

Then clamp to `[step_min, step_max]`.

This stabilizes exploration: too many rejects shrinks moves; too many accepts grows them.

---

## Multi-Start Trials

`main()` runs `trials` independent runs, each with a different RNG seed and fresh random initialization.

The globally best configuration is selected by minimal `feas`.

---

## Outputs

### `best_circles.csv`

* Header includes `L`, `best_feas`, and `N`
* One row per circle: `i,x,y,r`

### `best.svg`

* Draws the container square and circles
* Simple coordinate map:

  * world `[-L/2,L/2]` mapped to pixels with y-axis flipped

---

## How to Modify for Experiments

Common edits (all in `main()` unless noted):

* **Change problem size**

  * `int N = ...;`
  * `s.L = ...;`
  * radii assignment loop `s.r[i] = ...;`

* **Improve feasibility hardness**

  * in Phase B: raise `lambda_max`, `mu_max` (e.g. `1e7`)
  * or increase `ramp_factor`, decrease `ramp_every`

* **Change teleport behavior**

  * `A.p_reinsert`, `B.p_reinsert`

* **Optimize `L` (not currently active)**

  * set `w.alpha_L > 0` and add a move that perturbs `s.L`
  * must update `outside_for_k` accordingly and revalidate totals carefully

---

## Logging

Each phase prints periodic status lines with:

* iteration, temperature, step size
* energy, overlap total, outside total, feasibility
* acceptance rate
* current penalties (`lam`, `mu`)

