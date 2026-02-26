# CSV Schema - Bisection Study Logging

## File: `{prefix}_{method}_N{NNN}_s{seed}_bisection.csv`

One row per bisection probe.

| Column | Type | Description |
|--------|------|-------------|
| run_id | uint64 | Run identifier |
| seed | uint64 | RNG base seed |
| method | string | "ms", "erms", or "pt" |
| N | int | Number of polygons |
| R | int | Number of replicas |
| probe_idx | int | 0-indexed probe number |
| wall_sec_start | float | Wall seconds since run start (probe begin) |
| wall_sec_end | float | Wall seconds since run start (probe end) |
| L_lo | float | Lower bracket bound after this probe |
| L_hi | float | Upper bracket bound after this probe |
| L_mid | float | L tested in this probe |
| slice_budget_sec | float | Allocated time for this probe |
| slice_used_sec | float | Actual time consumed |
| feasible | int | 1 if feasible found, 0 otherwise |
| min_energy | float | Minimum energy seen in slice |
| min_feas | float | Minimum feasibility metric in slice |
| resample_events | int | ER-MS: resample events in slice (MS sets to 0) |
| L_best | float or empty | Best feasible L found so far |
| bracket_width | float | L_hi - L_lo |

Always written, even for infeasible probes (min_energy, min_feas still populated).

## File: `{prefix}_{method}_N{NNN}_s{seed}_log.csv`

Progress snapshots. One row per event (probe_start, probe_end, tick).

| Column | Type | Description |
|--------|------|-------------|
| run_id | uint64 | Run identifier |
| seed | uint64 | RNG base seed |
| method | string | Method name |
| N | int | Polygon count |
| R | int | Replica count |
| wall_sec | float | Seconds since run start |
| probe_idx | int | Current probe index |
| L_current | float | L being probed |
| best_energy | float | Best energy seen globally |
| best_feas | float | Best feasibility metric globally |
| feasible_ever | int | 1 if any feasible found |
| L_best | float or empty | Best feasible L |
| event | string | "probe_start", "probe_end", "tick" |
