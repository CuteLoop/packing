// Annealing engine: propose single-polygon moves, perform incremental geometry
// updates and grid updates, and accept/reject with Metropolis.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/annealing.h"
#include "../include/physics.h"
#include "../include/geometry.h"
#include "../include/spatial_hash.h"
#include "../include/utils.h"

typedef struct {
    int k;
    double old_cx, old_cy, old_th;
    double dE;
    double d_ov;
    double d_out;
} Move;

static Move propose_move(State *s, const Weights *w, RNG *rng, double step_xy, double step_th) {
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

static void undo_move(State *s, const Move *m) {
    int k = m->k;
    s->cx[k] = m->old_cx;
    s->cy[k] = m->old_cy;
    s->th[k] = m->old_th;
    update_instance(s, k);
    grid_update(&s->grid, k, s->cx[k], s->cy[k]);
}

static void run_phase(State *s, Totals *t, Weights *w, RNG *rng, const PhaseParams *pp, double *step_xy, double *step_th) {
    double temp = pp->T_start;
    double alpha = 1.0;
    if (pp->iters > 0) alpha = pow(pp->T_end / pp->T_start, 1.0 / (double)pp->iters);

    long long accepts = 0;
    int adapt_window = pp->adapt_window > 0 ? pp->adapt_window : 1;

    for (int i = 0; i < pp->iters; i++) {
        Move m = propose_move(s, w, rng, *step_xy, *step_th);

        int accept = 0;
        if (m.dE <= 0.0) accept = 1;
        else if (rng_u01(rng) < exp(-m.dE / temp)) accept = 1;

        if (accept) {
            t->overlap_total += m.d_ov;
            t->out_total += m.d_out;
            accepts++;
        } else {
            undo_move(s, &m);
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

double try_pack_at_current_L(State *s, RNG *rng, const PhaseParams *A, const PhaseParams *B, int trials,
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
        run_phase(s, &tot, &w, rng, A, &step_xy, &step_th);

        // Phase B (enforce)
        // optionally ramp weights
        w.lambda_ov = B->lambda_start;
        w.mu_out = B->mu_start;
        step_xy = B->step_xy_start;
        step_th = B->step_th_start;
        run_phase(s, &tot, &w, rng, B, &step_xy, &step_th);

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
