Here is the finalized, consolidated engineering document reflecting all critiques, constraints, and refinements.

You can drop this directly into your repo as:

`ENGINEERING_SPEC_v1_3.md`

---

# ENGINEERING_SPEC_v1_3.md

## Class-Scale Parallel Optimization Study + N=200 Hero Run

### HPC_DEMO — Non-Convex Polygon Packing

---

# 1. Purpose

This study evaluates three parallel optimization strategies for minimizing the side length (L) of a square containing (N) non-convex polygons.

We operate under strict class-project constraints:

* **Max 20 CPU cores**
* **≤ 4 hours wall-clock per method (comparison suite)**
* One **N=200 hero run** demonstrating a near-optimal packing
* Produce clear, interpretable convergence plots

This is a **demonstrative algorithms study**, not a statistically exhaustive HPC campaign.

---

# 2. Methods Compared

We evaluate three coordination strategies:

### Method A — MS (Multi-Start)

Independent parallel simulated annealing workers (no communication).

### Method B — ER-MS (Elite Resample Multi-Start)

Parallel workers with periodic elite selection and restart with perturbation.

### Method C — PT (Parallel Tempering)

Replica exchange across a geometric temperature ladder.

---

# 3. Experimental Structure

We split the study into:

## 3.1 Graph Suite (Comparison Study)

Used for algorithm comparison and convergence plots.

[
N \in {5, 10, 20, 50, 100}
]

## 3.2 Hero Suite (Demonstration Run)

[
N = 200
]

The N=200 run is a **demonstration of capability**, not a statistical comparison.

---

# 4. Resource Model

## 4.1 Hardware

* 20 CPUs maximum

## 4.2 Parallel Layout

### Graph Suite

* R = 10 threads per run
* Run 2 seeds concurrently (2 × 10 = 20 CPUs)

### Hero Run

* R = 20 threads
* Single run using all CPUs

---

# 5. Metrics

## 5.1 Primary Metric

[
L_{\text{best}}(t)
]
Best feasible (L) discovered up to wall time (t).

## 5.2 Secondary Metrics

* Final (L_{\text{best}}(T))
* Final bracket width (L_{hi} - L_{lo})
* Success rate
* Minimum penalty per probe (E_{\min})

## 5.3 Hero Metric

Packing efficiency:

[
\eta = \frac{N A_{poly}}{L^2}
]

---

# 6. Feasibility Definition

A configuration is feasible if:

* `outside_penalty <= EPS_FEAS`
* `overlap_penalty <= EPS_FEAS`

`EPS_FEAS` is a fixed constant defined in configuration and must not change during the study.

---

# 7. Determinism

* RNG state is part of configuration state.
* Swaps and clones copy RNG state.
* Seeds are deterministic per:

  ```
  seed_r = splitmix64(global_seed ^ hash(N, run_id, probe_idx, r))
  ```

---

# 8. Bracket Initialization

[
L_{lo} = \sqrt{N \cdot A_{poly}}
]

[
L_{hi} = \gamma(N) \cdot L_{lo}
]

Where:

| N       | γ   |
| ------- | --- |
| 5,10,20 | 2.2 |
| 50      | 2.6 |
| 100,200 | 3.0 |

---

## 8.1 Emergency Expansion (N=200 Only)

If first 3 probes at (L = L_{hi}) are infeasible:

[
L_{hi} \leftarrow 1.5 \cdot L_{hi}
]

Triggered at most once.

---

# 9. Bisection Schedule

Repair time counts within slice:

[
t_{repair} = \min(0.5s,; 0.05 \cdot t_{slice})
]

---

## 9.1 Graph Suite Slice Schedule

Base slice time:

| N       | t_base |
| ------- | ------ |
| 5,10,20 | 6s     |
| 50      | 10s    |
| 100     | 15s    |

Probe buckets:

* Probes 1–5: 1 × t_base
* Probes 6–10: 2 × t_base
* Probes 11+: 4 × t_base

Stop if:

[
L_{hi} - L_{lo} \le 10^{-3} L_{hi}
]

or time expires.

---

## 9.2 N=200 Hero Slice Schedule

* Probes 1–6: 60s
* Probes 7–12: 120s
* Probes 13+: 240s

---

# 10. Method Definitions

## 10.1 MS

* R independent workers
* Early stop if any worker feasible

## 10.2 ER-MS

Resample frequency:
[
K_{resample} = 200 \cdot N
]

Protect top 25% workers.

Elite pool:
[
k' = \lceil 0.25R \rceil
]

Noise schedule within probe (elapsed fraction f):

[
\sigma_{pos} = (0.02 - 0.015f) L
]

[
\sigma_{\theta} = (5^\circ - 4^\circ f)
]

Anti-collapse trigger (relative):

[
\frac{E_{max}-E_{min}}{\max(1, |E_{mean}|)} < 10^{-6}
]

for two resamples → double noise next resample only.

---

## 10.3 PT

* Geometric ladder
* Numeric Tmin and Tmax defined explicitly in config
* Swap frequency:

[
K_{swap} = 200 \cdot N
]

* Adjacent swaps only
* Early stop only if **coldest replica feasible**

### PT Pilot Tuning

* 10-min pilot at N=100, R=10
* Acceptance must fall in [0.15, 0.60]
* Adjust Tmax/Tmin by ×2 if needed
* Lock values afterward

Additional 5-min sanity pilot at N=200.

---

# 11. Standardized Polish Procedure

Polish uses MS-polish (R=20) regardless of original method.

Graph suite: optional 5-minute polish.

Hero run:

* 50-minute polish at fixed (L_{best})

### Thermal Pulse Rule

If no improvement for 10 minutes:

* Temperature ×10 for 30 seconds
* Ramp back down

---

# 12. Time Allocation

## 12.1 Graph Suite (Per Method)

| N   | seeds | minutes/seed |
| --- | ----- | ------------ |
| 5   | 5     | 2            |
| 10  | 5     | 2            |
| 20  | 5     | 3            |
| 50  | 5     | 6            |
| 100 | 5     | 10           |

Total runtime per method ≈ 115 minutes
Wall time ≈ 58 minutes (2 concurrent runs)

---

## 12.2 Hero Run (Single Selected Method)

* 150 minutes bisection
* 50 minutes polish
* R = 20

Total ≈ 200 minutes (3h20m)

---

# 13. Reporting Rules

Graph suite:

* Plot all 5 seed traces per method
* Optional mean overlay
* No statistical confidence intervals claimed

Hero:

* Report final L
* Report packing efficiency η
* Show SVG

---

# 14. Limitations (Explicitly Acknowledged)

* R differs between graph suite (10) and hero (20)
* Small seed counts → illustrative trends, not statistical claims
* Hero run is demonstration, not formal comparison

---

# 15. Deliverables

For each run:

* `_bisection.csv`
* `_log.csv`
* Best configuration CSV
* SVG

Final report includes:

* Convergence plots
* Method comparison discussion
* Hero packing visualization
* Efficiency metric η

