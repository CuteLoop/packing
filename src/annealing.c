// Annealing engine: propose single-polygon moves, perform incremental geometry
// updates and grid updates, and accept/reject with Metropolis.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../include/annealing.h"
#include "../include/physics.h"
#include "../include/geometry.h"
#include "../include/spatial_hash.h"
#include "../include/utils.h"
#include "../include/replica.h"

typedef struct {
    int k;
    double old_cx, old_cy, old_th;
    double dE;
    double d_ov;
    double d_out;
} Move;

static Move propose_move_state(State *s, const Weights *w, RNG *rng, double step_xy, double step_th) {
    Move m;
    m.k = (int)(rng_u01(rng) * (double)s->N);
    if (m.k < 0) m.k = 0;
    if (m.k >= s->N) m.k = s->N - 1;

    m.old_cx = s->cx[m.k];
    m.old_cy = s->cy[m.k];
    m.old_th = s->th[m.k];

    double ov_before = overlap_sum_for_k_grid(s, m.k);
    double out_before = outside_penalty_aabb(&s->aabb[m.k], s->L);

    s->cx[m.k] += rng_uniform(rng, -step_xy, step_xy);
    s->cy[m.k] += rng_uniform(rng, -step_xy, step_xy);
    s->th[m.k] += rng_uniform(rng, -step_th, step_th);
    s->th[m.k] = wrap_angle_0_2pi(s->th[m.k]);

    update_instance(s, m.k);
    grid_update(&s->grid, m.k, s->cx[m.k], s->cy[m.k]);

    double ov_after = overlap_sum_for_k_grid(s, m.k);
    double out_after = outside_penalty_aabb(&s->aabb[m.k], s->L);

    m.d_ov = ov_after - ov_before;
    m.d_out = out_after - out_before;
    m.dE = w->lambda_ov * m.d_ov + w->mu_out * m.d_out;

    return m;
}

static void undo_move_state(State *s, const Move *m) {
    int k = m->k;
    s->cx[k] = m->old_cx;
    s->cy[k] = m->old_cy;
    s->th[k] = m->old_th;
    update_instance(s, k);
    grid_update(&s->grid, k, s->cx[k], s->cy[k]);
}

static void run_phase_state(State *s, Totals *t, Weights *w, RNG *rng, const PhaseParams *pp, double *step_xy, double *step_th) {
    double temp = pp->T_start;
    double alpha = 1.0;
    if (pp->iters > 0) alpha = pow(pp->T_end / pp->T_start, 1.0 / (double)pp->iters);

    long long accepts = 0;
    int adapt_window = pp->adapt_window > 0 ? pp->adapt_window : 1;

    for (int i = 0; i < pp->iters; i++) {
        Move m = propose_move_state(s, w, rng, *step_xy, *step_th);

        int accept = 0;
        if (m.dE <= 0.0) accept = 1;
        else if (rng_u01(rng) < exp(-m.dE / temp)) accept = 1;

        if (accept) {
            t->overlap_total += m.d_ov;
            t->out_total += m.d_out;
            accepts++;
        } else {
            undo_move_state(s, &m);
        }

        temp *= alpha;

        if ((i + 1) % adapt_window == 0) {
            double rate = (double)accepts / (double)adapt_window;
            accepts = 0;
            if (rate < pp->acc_low) {
                *step_xy *= pp->step_shrink;
                *step_th *= pp->step_shrink;
            } else if (rate > pp->acc_high) {
                *step_xy *= pp->step_grow;
                *step_th *= pp->step_grow;
            }
            if (*step_xy < pp->step_xy_min) *step_xy = pp->step_xy_min;
            if (*step_xy > pp->step_xy_max) *step_xy = pp->step_xy_max;
            if (*step_th < pp->step_th_min) *step_th = pp->step_th_min;
            if (*step_th > pp->step_th_max) *step_th = pp->step_th_max;
        }

        if (pp->ramp_every > 0 && ((i+1) % pp->ramp_every) == 0) {
            w->lambda_ov = fmin(pp->lambda_max, w->lambda_ov * pp->ramp_factor);
            w->mu_out = fmin(pp->mu_max, w->mu_out * pp->ramp_factor);
        }
    }
}

static inline RNG rng_from_replica(const ReplicaState *r) {
    RNG rng;
    rng.s = r->rng_s;
    return rng;
}

static inline void rng_to_replica(ReplicaState *r, const RNG *rng) {
    r->rng_s = rng->s;
}

static void populate_bridge_state(State *bridge, ReplicaState *r, Workspace *w, int N, double L) {
    bridge->N = N;
    bridge->L = L;
    bridge->cx = r->cx;
    bridge->cy = r->cy;
    bridge->th = r->th;
    bridge->world = w->world;
    bridge->aabb = w->aabb;
    bridge->tri_aabb = w->tri_aabb;
    bridge->br = w->br;
    bridge->grid = w->grid;
}

static Move propose_move(ReplicaState *r, Workspace *w, State *bridge, const Weights *weights,
                         int N, double L, double step_xy, double step_th)
{
    Move m;
    RNG rng = rng_from_replica(r);
    m.k = (int)(rng_u01(&rng) * (double)N);
    if (m.k < 0) m.k = 0;
    if (m.k >= N) m.k = N - 1;

    m.old_cx = r->cx[m.k];
    m.old_cy = r->cy[m.k];
    m.old_th = r->th[m.k];

    double ov_before = overlap_sum_for_k_grid(bridge, m.k);
    double out_before = outside_penalty_aabb(&w->aabb[m.k], L);

    r->cx[m.k] += rng_uniform(&rng, -step_xy, step_xy);
    r->cy[m.k] += rng_uniform(&rng, -step_xy, step_xy);
    r->th[m.k] += rng_uniform(&rng, -step_th, step_th);
    r->th[m.k] = wrap_angle_0_2pi(r->th[m.k]);

    update_instance_rw(r->cx, r->cy, r->th, w->world, w->aabb, w->tri_aabb, m.k);
    grid_update(&w->grid, m.k, r->cx[m.k], r->cy[m.k]);

    double ov_after = overlap_sum_for_k_grid(bridge, m.k);
    double out_after = outside_penalty_aabb(&w->aabb[m.k], L);

    m.d_ov = ov_after - ov_before;
    m.d_out = out_after - out_before;
    m.dE = weights->lambda_ov * m.d_ov + weights->mu_out * m.d_out;

    rng_to_replica(r, &rng);
    return m;
}

static void undo_move(ReplicaState *r, Workspace *w, int k, const Move *m) {
    r->cx[k] = m->old_cx;
    r->cy[k] = m->old_cy;
    r->th[k] = m->old_th;
    update_instance_rw(r->cx, r->cy, r->th, w->world, w->aabb, w->tri_aabb, k);
    grid_update(&w->grid, k, r->cx[k], r->cy[k]);
}

int run_sa_epoch(ReplicaState *r, Workspace *w, int N, double L,
                 Weights *weights, double eps_feas,
                 const PhaseParams *pp, int K, volatile int *early_stop)
{
    State bridge;
    populate_bridge_state(&bridge, r, w, N, L);

    double temp = pp->T_start;
    double alpha = 1.0;
    if (K > 0) alpha = pow(pp->T_end / pp->T_start, 1.0 / (double)K);

    long long accepts = 0;
    int adapt_window = pp->adapt_window > 0 ? pp->adapt_window : 1;
    int steps = 0;

    for (int i = 0; i < K; i++) {
        if (early_stop && (i % 1000 == 0) && *early_stop) break;

        Move m = propose_move(r, w, &bridge, weights, N, L, r->step_xy, r->step_th);

        RNG rng = rng_from_replica(r);
        int accept = 0;
        if (m.dE <= 0.0) accept = 1;
        else if (rng_u01(&rng) < exp(-m.dE / temp)) accept = 1;
        rng_to_replica(r, &rng);

        if (accept) {
            r->totals.overlap_total += m.d_ov;
            r->totals.out_total += m.d_out;
            accepts++;
        } else {
            undo_move(r, w, m.k, &m);
        }

        temp *= alpha;
        steps++;

        if ((i + 1) % adapt_window == 0) {
            double rate = (double)accepts / (double)adapt_window;
            accepts = 0;
            if (rate < pp->acc_low) {
                r->step_xy *= pp->step_shrink;
                r->step_th *= pp->step_shrink;
            } else if (rate > pp->acc_high) {
                r->step_xy *= pp->step_grow;
                r->step_th *= pp->step_grow;
            }
            if (r->step_xy < pp->step_xy_min) r->step_xy = pp->step_xy_min;
            if (r->step_xy > pp->step_xy_max) r->step_xy = pp->step_xy_max;
            if (r->step_th < pp->step_th_min) r->step_th = pp->step_th_min;
            if (r->step_th > pp->step_th_max) r->step_th = pp->step_th_max;
        }

        if (pp->ramp_every > 0 && ((i + 1) % pp->ramp_every) == 0) {
            weights->lambda_ov = fmin(pp->lambda_max, weights->lambda_ov * pp->ramp_factor);
            weights->mu_out = fmin(pp->mu_max, weights->mu_out * pp->ramp_factor);
        }
    }

    r->temp = temp;
    r->overlap_penalty = r->totals.overlap_total;
    r->outside_penalty = r->totals.out_total;
    r->feas = feasibility_metric(&r->totals);
    r->is_feasible = (r->feas <= eps_feas) ? 1 : 0;
    r->energy = energy_from_totals(&bridge, weights, &r->totals);
    r->proposals_done += (uint64_t)steps;
    r->epoch_proposals += (uint64_t)steps;

    return steps;
}

static void replica_from_state(ReplicaState *r, const State *s, const RNG *rng) {
    memset(r, 0, sizeof(*r));
    for (int i = 0; i < s->N; i++) {
        r->cx[i] = s->cx[i];
        r->cy[i] = s->cy[i];
        r->th[i] = s->th[i];
    }
    r->rng_s = rng->s;
}

static void state_from_replica(State *s, const ReplicaState *r) {
    for (int i = 0; i < s->N; i++) {
        s->cx[i] = r->cx[i];
        s->cy[i] = r->cy[i];
        s->th[i] = r->th[i];
    }
}

double try_pack_at_current_L(State *s, RNG *rng, const PhaseParams *A, const PhaseParams *B, int trials,
                             uint64_t seed, uint64_t run_id,
                             double *out_cx, double *out_cy, double *out_th, int verbose)
{
    const double eps_feas = 1e-6;
    int N = s->N;
    double *trial_best_cx = (double*)malloc((size_t)N * sizeof(double));
    double *trial_best_cy = (double*)malloc((size_t)N * sizeof(double));
    double *trial_best_th = (double*)malloc((size_t)N * sizeof(double));
    if (!trial_best_cx || !trial_best_cy || !trial_best_th) {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }

    double cell = (s->grid.cell > 0.0) ? s->grid.cell : (s->br * 2.0);

    ReplicaState r;
    replica_from_state(&r, s, rng);

    Workspace w;
    workspace_init(&w, N, s->L, cell);
    rebuild_derived(&r, &w, N, s->L, cell);

    Weights weights;
    weights.alpha_L = 0.0;
    weights.lambda_ov = A->lambda_start;
    weights.mu_out = A->mu_start;
    evaluate_full(&r, &w, N, s->L, &weights, eps_feas);

    double best_feas = r.feas;
    for (int i = 0; i < N; i++) {
        trial_best_cx[i] = r.cx[i];
        trial_best_cy[i] = r.cy[i];
        trial_best_th[i] = r.th[i];
    }

    for (int tr = 0; tr < trials; tr++) {
        uint64_t trial_seed = make_trial_seed(seed, run_id, (uint64_t)(tr + 1));
        RNG trial_rng;
        rng_seed(&trial_rng, trial_seed);
        r.rng_s = trial_rng.s;

        if (tr > 0) {
            for (int i = 0; i < N; i++) {
                r.cx[i] = out_cx ? out_cx[i] : trial_best_cx[i];
                r.cy[i] = out_cy ? out_cy[i] : trial_best_cy[i];
                r.th[i] = out_th ? out_th[i] : trial_best_th[i];
            }
            rebuild_derived(&r, &w, N, s->L, cell);
            evaluate_full(&r, &w, N, s->L, &weights, eps_feas);
        }

        if (verbose) {
            printf("  - SA trial %d/%d (seed=%llu) start: ov=%.2e out=%.2e feas=%.2e\n",
                   tr + 1, trials, (unsigned long long)trial_seed,
                   r.totals.overlap_total, r.totals.out_total, r.feas);
            fflush(stdout);
        }

        weights.lambda_ov = A->lambda_start;
        weights.mu_out = A->mu_start;
        r.step_xy = A->step_xy_start;
        r.step_th = A->step_th_start;
        r.epoch_proposals = 0;
        run_sa_epoch(&r, &w, N, s->L, &weights, eps_feas, A, A->iters, NULL);

        weights.lambda_ov = B->lambda_start;
        weights.mu_out = B->mu_start;
        r.step_xy = B->step_xy_start;
        r.step_th = B->step_th_start;
        r.epoch_proposals = 0;
        run_sa_epoch(&r, &w, N, s->L, &weights, eps_feas, B, B->iters, NULL);

        double feas = r.feas;
        if (feas < best_feas) {
            best_feas = feas;
            for (int i = 0; i < N; i++) {
                trial_best_cx[i] = r.cx[i];
                trial_best_cy[i] = r.cy[i];
                trial_best_th[i] = r.th[i];
            }
            if (out_cx && out_cy && out_th) {
                for (int i = 0; i < N; i++) {
                    out_cx[i] = trial_best_cx[i];
                    out_cy[i] = trial_best_cy[i];
                    out_th[i] = trial_best_th[i];
                }
            }
        }
    }

    state_from_replica(s, &r);
    for (int i = 0; i < N; i++) update_instance(s, i);
    grid_rebuild(&s->grid, s->N, s->L, cell, s->cx, s->cy);

    rng->s = r.rng_s;
    workspace_free(&w);

    free(trial_best_cx);
    free(trial_best_cy);
    free(trial_best_th);
    return best_feas;
}

double try_pack_at_current_L_old(State *s, RNG *rng, const PhaseParams *A, const PhaseParams *B, int trials,
                                 uint64_t seed, uint64_t run_id,
                                 double *out_cx, double *out_cy, double *out_th, int verbose)
{
    int N = s->N;
    double *trial_best_cx = (double*)malloc((size_t)N * sizeof(double));
    double *trial_best_cy = (double*)malloc((size_t)N * sizeof(double));
    double *trial_best_th = (double*)malloc((size_t)N * sizeof(double));
    if (!trial_best_cx || !trial_best_cy || !trial_best_th) {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }

    Totals tot = compute_totals_full_grid(s);
    Weights w;
    w.alpha_L = 0.0;
    w.lambda_ov = A->lambda_start;
    w.mu_out = A->mu_start;

    double best_feas = feasibility_metric(&tot);
    for (int i = 0; i < N; i++) {
        trial_best_cx[i] = s->cx[i];
        trial_best_cy[i] = s->cy[i];
        trial_best_th[i] = s->th[i];
    }

    // Ensure grid is populated
    grid_rebuild(&s->grid, s->N, s->L, s->grid.cell, s->cx, s->cy);
    tot = compute_totals_full_grid(s);
    best_feas = feasibility_metric(&tot);

    for (int tr = 0; tr < trials; tr++) {
        uint64_t trial_seed = make_trial_seed(seed, run_id, (uint64_t)(tr + 1));
        rng_seed(rng, trial_seed);

        if (tr > 0) {
            // warm-start: keep best found so far
            for (int i = 0; i < N; i++) {
                s->cx[i] = out_cx ? out_cx[i] : trial_best_cx[i];
                s->cy[i] = out_cy ? out_cy[i] : trial_best_cy[i];
                s->th[i] = out_th ? out_th[i] : trial_best_th[i];
            }
            for (int i = 0; i < N; i++) update_instance(s, i);
            grid_rebuild(&s->grid, s->N, s->L, s->grid.cell, s->cx, s->cy);
            tot = compute_totals_full_grid(s);
        }

        if (verbose) {
            printf("  - SA trial %d/%d (seed=%llu) start: ov=%.2e out=%.2e feas=%.2e\n",
                   tr + 1, trials, (unsigned long long)trial_seed, tot.overlap_total, tot.out_total, feasibility_metric(&tot));
            fflush(stdout);
        }

        // Phase A (explore)
        double step_xy = A->step_xy_start;
        double step_th = A->step_th_start;
        run_phase_state(s, &tot, &w, rng, A, &step_xy, &step_th);

        // Phase B (enforce)
        w.lambda_ov = B->lambda_start;
        w.mu_out = B->mu_start;
        step_xy = B->step_xy_start;
        step_th = B->step_th_start;
        run_phase_state(s, &tot, &w, rng, B, &step_xy, &step_th);

        // Record trial best
        double feas = feasibility_metric(&tot);
        if (feas < best_feas) {
            best_feas = feas;
            for (int i = 0; i < N; i++) {
                trial_best_cx[i] = s->cx[i];
                trial_best_cy[i] = s->cy[i];
                trial_best_th[i] = s->th[i];
            }
            if (out_cx && out_cy && out_th) {
                for (int i = 0; i < N; i++) {
                    out_cx[i] = trial_best_cx[i];
                    out_cy[i] = trial_best_cy[i];
                    out_th[i] = trial_best_th[i];
                }
            }
        }
    }

    // restore best into state
    for (int i = 0; i < N; i++) {
        s->cx[i] = trial_best_cx[i];
        s->cy[i] = trial_best_cy[i];
        s->th[i] = trial_best_th[i];
    }
    for (int i = 0; i < N; i++) update_instance(s, i);
    grid_rebuild(&s->grid, s->N, s->L, s->grid.cell, s->cx, s->cy);

    free(trial_best_cx); free(trial_best_cy); free(trial_best_th);
    return best_feas;
}
