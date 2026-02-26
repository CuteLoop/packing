#include "../include/methods.h"
#include "../include/replica.h"
#include "../include/annealing.h"
#include "../include/geometry.h"
#include "../include/utils.h"
#include <omp.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double pt_swap_accept_prob(double Ei, double Ej, double Ti, double Tj) {
    if (Ti <= 0.0 || Tj <= 0.0) return 0.0;
    double beta_i = 1.0 / Ti;
    double beta_j = 1.0 / Tj;
    double d = (beta_i - beta_j) * (Ej - Ei);
    if (d >= 0.0) return 1.0;
    double a = exp(d);
    return a < 0.0 ? 0.0 : a;
}

static int replica_better_than(const ReplicaState *a, const ReplicaState *b) {
    if (a->is_feasible && !b->is_feasible) return 1;
    if (!a->is_feasible && b->is_feasible) return 0;

    double fa = a->overlap_penalty + a->outside_penalty;
    double fb = b->overlap_penalty + b->outside_penalty;
    if (fa < fb) return 1;
    if (fa > fb) return 0;

    if (a->energy < b->energy) return 1;
    if (a->energy > b->energy) return 0;

    return a->replica_id < b->replica_id;
}

static void build_temp_ladder(double *temps, int R, double Tmin, double Tmax) {
    if (R <= 1) {
        temps[0] = Tmin;
        return;
    }
    double ratio = Tmax / Tmin;
    for (int i = 0; i < R; i++) {
        double t = (double)i / (double)(R - 1);
        temps[i] = Tmin * pow(ratio, t);
    }
}

void runner_pt(
    const StudyConfig *cfg,
    double L,
    double slice_budget_sec,
    int probe_idx,
    const ReplicaState *warm_init,
    SliceResult *out)
{
    memset(out, 0, sizeof(*out));
    int N = cfg->N;
    int R = cfg->R;
    double cell = base_bounding_radius() * 2.0;

    ReplicaState *replicas = (ReplicaState *)calloc((size_t)R, sizeof(ReplicaState));
    double *temps = (double *)calloc((size_t)R, sizeof(double));
    if (!replicas || !temps) {
        free(replicas);
        free(temps);
        return;
    }

    double Tmin = 1.0;
    double Tmax = 25.0;
    build_temp_ladder(temps, R, Tmin, Tmax);

    for (int r = 0; r < R; r++) {
        uint64_t trial_id = (uint64_t)probe_idx * 1000000ULL + (uint64_t)r;
        uint64_t rseed = make_trial_seed(cfg->seed, cfg->run_id, trial_id);
        if (warm_init) {
            replica_clone(&replicas[r], warm_init, rseed);
            replicas[r].replica_id = r;
        } else {
            replica_init_random(&replicas[r], N, L, rseed, temps[r], r);
        }
        replicas[r].temp = temps[r];
    }

    int early_stop = 0;
    int confirmed_feasible = 0;
    int global_stop = 0;
    double t_start = now_seconds();

    RNG swap_rng;
    rng_seed(&swap_rng, make_trial_seed(cfg->seed, cfg->run_id, (uint64_t)probe_idx * 4242ULL));

#pragma omp parallel
    {
        Workspace w;
        workspace_init(&w, N, L, cell);

#pragma omp for schedule(static)
        for (int r = 0; r < R; r++) {
            rebuild_derived(&replicas[r], &w, N, L, cell);
            evaluate_full(&replicas[r], &w, N, L, &cfg->weights, cfg->eps_feas);
        }

        int parity = 0;
        int K_epoch = 200 * N;
        int K_chunk = 20 * N;
        if (K_chunk > K_epoch) K_chunk = K_epoch;

        while (1) {
            #pragma omp single
            {
                int stop = 0;
                if (now_seconds() - t_start >= slice_budget_sec) stop = 1;
                if (early_stop && confirmed_feasible) stop = 1;
                global_stop = stop;
            }
            if (global_stop) break;

#pragma omp for schedule(static)
            for (int r = 0; r < R; r++) {
                PhaseParams pp = {
                    .iters = 0,
                    .T_start = temps[r], .T_end = temps[r],
                    .step_xy_start = 0.05, .step_th_start = 0.5,
                    .adapt_window = 2000, .acc_low = 0.4, .acc_high = 0.6,
                    .step_shrink = 0.95, .step_grow = 1.05,
                    .step_xy_min = 1e-5, .step_xy_max = 2.0,
                    .step_th_min = 1e-4, .step_th_max = M_PI,
                    .lambda_start = cfg->weights.lambda_ov,
                    .mu_start = cfg->weights.mu_out,
                    .ramp_every = 0, .ramp_factor = 1.0,
                    .lambda_max = cfg->weights.lambda_ov,
                    .mu_max = cfg->weights.mu_out,
                    .p_reinsert = 0.0, .p_rotmix = 0.0,
                    .log_every = 0
                };

                replicas[r].temp = temps[r];
                Weights local_w = cfg->weights;
                run_sa_epoch(&replicas[r], &w, N, L,
                             &local_w, cfg->eps_feas,
                             &pp, K_chunk, &early_stop);
                evaluate_full(&replicas[r], &w, N, L, &local_w, cfg->eps_feas);

                if (replicas[r].is_feasible) {
#pragma omp atomic write
                    early_stop = 1;
                }
            }

#pragma omp single
            {
                int stop_check;
#pragma omp atomic read
                stop_check = early_stop;

                if (stop_check && !confirmed_feasible) {
                    int best_idx = -1;
                    for (int i = 0; i < R; i++) {
                        if (!replicas[i].is_feasible) continue;
                        if (best_idx < 0 || replica_better_than(&replicas[i], &replicas[best_idx])) {
                            best_idx = i;
                        }
                    }

                    if (best_idx >= 0) {
                        replicas[0] = replicas[best_idx];
                        replicas[0].replica_id = 0;
                        replicas[0].temp = temps[0];

                        PhaseParams pp_cold = {
                            .iters = 0,
                            .T_start = temps[0], .T_end = temps[0],
                            .step_xy_start = 0.05, .step_th_start = 0.5,
                            .adapt_window = 2000, .acc_low = 0.4, .acc_high = 0.6,
                            .step_shrink = 0.95, .step_grow = 1.05,
                            .step_xy_min = 1e-5, .step_xy_max = 2.0,
                            .step_th_min = 1e-4, .step_th_max = M_PI,
                            .lambda_start = cfg->weights.lambda_ov,
                            .mu_start = cfg->weights.mu_out,
                            .ramp_every = 0, .ramp_factor = 1.0,
                            .lambda_max = cfg->weights.lambda_ov,
                            .mu_max = cfg->weights.mu_out,
                            .p_reinsert = 0.0, .p_rotmix = 0.0,
                            .log_every = 0
                        };

                        Weights local_w = cfg->weights;
                        rebuild_derived(&replicas[0], &w, N, L, cell);
                        run_sa_epoch(&replicas[0], &w, N, L,
                                     &local_w, cfg->eps_feas,
                                     &pp_cold, 50 * N, &early_stop);
                        evaluate_full(&replicas[0], &w, N, L, &local_w, cfg->eps_feas);

                        if (replicas[0].is_feasible) {
                            confirmed_feasible = 1;
#pragma omp atomic write
                            early_stop = 1;
                        } else {
#pragma omp atomic write
                            early_stop = 0;
                        }
                    }
                }

                int stop = 0;
                if (now_seconds() - t_start >= slice_budget_sec) stop = 1;
                if (early_stop && confirmed_feasible) stop = 1;
                global_stop = stop;
            }

            if (global_stop) break;

#pragma omp single
            {
                if (!confirmed_feasible && !global_stop) {
                    int start = parity ? 1 : 0;
                    for (int i = start; i + 1 < R; i += 2) {
                        int j = i + 1;
                        double Ti = temps[i];
                        double Tj = temps[j];
                        double Ei = replicas[i].energy;
                        double Ej = replicas[j].energy;
                        double acc = pt_swap_accept_prob(Ei, Ej, Ti, Tj);
                        if (rng_u01(&swap_rng) < acc) {
                            ReplicaState tmp = replicas[i];
                            replicas[i] = replicas[j];
                            replicas[j] = tmp;
                            replicas[i].replica_id = i;
                            replicas[j].replica_id = j;
                            replicas[i].temp = temps[i];
                            replicas[j].temp = temps[j];
                        }
                    }
                    parity = 1 - parity;
                }
            }
        }

        workspace_free(&w);
    }

    double global_min_energy = 1e30;
    double global_min_feas = 1e30;
    int global_feasible = 0;
    ReplicaState global_best;
    memset(&global_best, 0, sizeof(global_best));
    global_best.energy = 1e30;
    global_best.replica_id = -1;

    for (int r = 0; r < R; r++) {
        if (replicas[r].energy < global_min_energy)
            global_min_energy = replicas[r].energy;
        double fm = replicas[r].overlap_penalty + replicas[r].outside_penalty;
        if (fm < global_min_feas)
            global_min_feas = fm;
        if (replicas[r].is_feasible)
            global_feasible = 1;

        if (global_best.replica_id < 0 ||
            replica_better_than(&replicas[r], &global_best)) {
            global_best = replicas[r];
        }
    }

    out->feasible = global_feasible;
    out->min_energy = global_min_energy;
    out->min_feas = global_min_feas;
    out->best_state = global_best;
    out->has_state = (global_best.replica_id >= 0);
    out->slice_used_sec = now_seconds() - t_start;
    out->resample_events = 0;

    free(replicas);
    free(temps);
}
