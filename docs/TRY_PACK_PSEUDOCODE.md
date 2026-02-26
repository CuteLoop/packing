# Phase 0A.5 — Pseudocode (try_pack_at_current_L / run_phase)

Functions covered
-----------------
- run_phase(...)
- try_pack_at_current_L(...)

run_phase(...) outline
----------------------
```
function run_phase(s, t, w, rng, pp, step_xy, step_th):
    temp = pp.T_start
    alpha = (pp.iters > 0) ? pow(pp.T_end / pp.T_start, 1/pp.iters) : 1
    accepts = 0
    adapt_window = max(pp.adapt_window, 1)

    for i in 0 .. pp.iters-1:
        m = propose_move(s, w, rng, step_xy, step_th)
            // propose_move does:
            // - pick k
            // - compute ov_before = overlap_sum_for_k_grid(s, k)
            // - compute out_before = outside_penalty_aabb(aabb[k], L)
            // - update cx/cy/th for k
            // - update_instance(s, k)
            // - grid_update(s.grid, k)
            // - compute ov_after, out_after
            // - m.dE = w.lambda_ov * d_ov + w.mu_out * d_out

        accept = (m.dE <= 0) OR (rng_u01 < exp(-m.dE / temp))
        if accept:
            t.overlap_total += m.d_ov
            t.out_total += m.d_out
            accepts += 1
        else:
            undo_move(s, m)
                // undo_move restores cx/cy/th, update_instance, grid_update

        temp *= alpha

        if (i+1) % adapt_window == 0:
            rate = accepts / adapt_window
            accepts = 0
            if rate < pp.acc_low:  step_xy *= pp.step_shrink; step_th *= pp.step_shrink
            if rate > pp.acc_high: step_xy *= pp.step_grow;   step_th *= pp.step_grow
            clamp step_xy, step_th to [min,max]

        if pp.ramp_every > 0 and (i+1) % pp.ramp_every == 0:
            w.lambda_ov = min(pp.lambda_max, w.lambda_ov * pp.ramp_factor)
            w.mu_out    = min(pp.mu_max,    w.mu_out * pp.ramp_factor)
```

try_pack_at_current_L(...) outline
----------------------------------
```
function try_pack_at_current_L(s, rng, A, B, trials, seed, run_id, out_cx, out_cy, out_th, verbose):
    allocate trial_best_cx/cy/th arrays

    // evaluation
    tot = compute_totals_full_grid(s)
    w.alpha_L = 0
    w.lambda_ov = A.lambda_start
    w.mu_out = A.mu_start
    best_feas = feasibility_metric(tot)

    // warm-start baseline
    trial_best_* = s.*

    // grid ops
    grid_rebuild(s.grid, s.N, s.L, s.grid.cell, s.cx, s.cy)
    tot = compute_totals_full_grid(s)
    best_feas = feasibility_metric(tot)

    for tr in 0 .. trials-1:
        trial_seed = make_trial_seed(seed, run_id, tr+1)
        rng_seed(rng, trial_seed)

        if tr > 0:
            // warm-start from best
            s.cx/cy/th = out_* or trial_best_*
            update_instance(s, i) for all i
            grid_rebuild(s.grid, s.N, s.L, s.grid.cell, s.cx, s.cy)
            tot = compute_totals_full_grid(s)

        if verbose:
            print ov/out/feas using feasibility_metric(tot)

        // run_phase A then B
        step_xy = A.step_xy_start; step_th = A.step_th_start
        run_phase(s, tot, w, rng, A, step_xy, step_th)

        w.lambda_ov = B.lambda_start; w.mu_out = B.mu_start
        step_xy = B.step_xy_start; step_th = B.step_th_start
        run_phase(s, tot, w, rng, B, step_xy, step_th)

        feas = feasibility_metric(tot)
        if feas < best_feas:
            best_feas = feas
            trial_best_* = s.*
            if out_* pointers set: copy trial_best_* into out_*

    // restore best
    s.* = trial_best_*
    update_instance(s, i) for all i
    grid_rebuild(s.grid, s.N, s.L, s.grid.cell, s.cx, s.cy)

    free trial_best arrays
    return best_feas
```

Where specific operations occur
-------------------------------
- Proposal loop: run_phase for-loop (pp.iters iterations).
- State mutation per move: propose_move updates s.cx/s.cy/s.th for polygon k, then update_instance and grid_update.
- Evaluation:
  - compute_totals_full_grid called before trials and after grid rebuild.
  - feasibility_metric called after compute_totals_full_grid and after phases.
  - energy_from_totals is defined in physics.c but not used by try_pack_at_current_L.
- Grid ops:
  - grid_update in propose_move/undo_move.
  - grid_rebuild at start, between trials, and after restoring best.
- Geometry:
  - update_instance in propose_move/undo_move and in full rebuild loops.
- Logging/snapshots:
  - Only verbose printf in try_pack_at_current_L.
  - CSV/SVG logging is handled in main.c, not in try_pack_at_current_L.
```

Gate check
----------
With this doc, an agent can answer:
- "Where does the proposal loop live?" -> run_phase for-loop.
- "What state is mutated per move?" -> s.cx/s.cy/s.th, s.world/aabb/tri_aabb via update_instance, s.grid via grid_update.
