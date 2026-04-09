#include "../include/bisection.h"
#include "../include/annealing.h"
#include "../include/config.h"
#include "../include/utils.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void warmstart_scale(ReplicaState *r, int N, double L_old, double L_new) {
    double alpha = L_new / L_old;
    double half = 0.5 * L_new;

    for (int i = 0; i < N; i++) {
        r->cx[i] *= alpha;
        r->cy[i] *= alpha;

        if (r->cx[i] < -half) r->cx[i] = -half;
        if (r->cx[i] >  half) r->cx[i] =  half;
        if (r->cy[i] < -half) r->cy[i] = -half;
        if (r->cy[i] >  half) r->cy[i] =  half;
    }

    r->energy = 1e30;
    r->overlap_penalty = 1e30;
    r->outside_penalty = 1e30;
    r->feas = 1e30;
    r->is_feasible = 0;
}

void warmstart_repair(ReplicaState *r, Workspace *w, int N, double L,
                      const Weights *weights, double eps_feas,
                      double max_repair_sec)
{
    (void)max_repair_sec;

    Weights local_w = *weights;

    PhaseParams pp = {
        .iters = 0,
        .T_start = 1.0, .T_end = 1e-4,
        .step_xy_start = 0.05, .step_th_start = 0.5,
        .adapt_window = 200, .acc_low = 0.4, .acc_high = 0.6,
        .step_shrink = 0.95, .step_grow = 1.05,
        .step_xy_min = 1e-5, .step_xy_max = 1.0,
        .step_th_min = 1e-4, .step_th_max = M_PI,
        .lambda_start = local_w.lambda_ov,
        .mu_start = local_w.mu_out,
        .ramp_every = 0, .ramp_factor = 1.0,
        .lambda_max = local_w.lambda_ov,
        .mu_max = local_w.mu_out,
        .p_reinsert = 0.0, .p_rotmix = 0.0,
        .log_every = 0
    };

    if (r->step_xy <= 0.0) r->step_xy = pp.step_xy_start;
    if (r->step_th <= 0.0) r->step_th = pp.step_th_start;

    evaluate_full(r, w, N, L, &local_w, eps_feas);

    int K_repair = 50 * N;
    volatile int stop = 0;
    run_sa_epoch(r, w, N, L, &local_w, eps_feas, &pp, K_repair, &stop);

    evaluate_full(r, w, N, L, &local_w, eps_feas);
}
