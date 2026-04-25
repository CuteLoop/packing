#include "../include/methods.h"
#include "../include/replica.h"
#include "../include/annealing.h"
#include "../include/geometry.h"
#include "../include/utils.h"
#include <omp.h>
#include <stdio.h>
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

/*
 * Canonical PT exchange energy.
 * Replica exchange must be computed from a common, temperature-independent
 * energy across all replicas. Using a replica-local annealing objective can
 * violate detailed balance and lead to pathological always-accept behavior.
 */
static double pt_exchange_energy(const ReplicaState *s) {
    return s->overlap_penalty + s->outside_penalty;
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

    double Tmin = (cfg->pt_Tmin > 0.0) ? cfg->pt_Tmin : 1.0;
    double Tmax = (cfg->pt_Tmax > 0.0) ? cfg->pt_Tmax : 25.0;
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
        /*
         * PT BUG FIX: replica_init_random and replica_clone both leave step_xy
         * and step_th at 0 (memset/copy from uninitialised source). run_sa_epoch
         * uses r->step_xy/step_th directly as the move size, so zero step sizes
         * mean every proposed move perturbs by exactly 0 -- replicas never move,
         * all energies stay identical, and swap acceptance is always 1.
         * Set sane defaults matching PhaseParams.step_xy_start / step_th_start.
         */
        if (replicas[r].step_xy <= 0.0) replicas[r].step_xy = 0.05;
        if (replicas[r].step_th <= 0.0) replicas[r].step_th = 0.5;
    }

    int early_stop = 0;
    int confirmed_feasible = 0;
    int global_stop = 0;
    double t_start = now_seconds();

    RNG swap_rng;
    rng_seed(&swap_rng, make_trial_seed(cfg->seed, cfg->run_id, (uint64_t)probe_idx * 4242ULL));

    // Swap statistics
    int swap_attempts = 0;
    int swap_accepts = 0;
#ifdef PT_DEBUG
    int debug_logged_swap_this_probe = 0;
#endif

#pragma omp parallel
    {
        Workspace w;
        workspace_init(&w, N, L, cell);

#pragma omp for schedule(static)
        for (int r = 0; r < R; r++) {
            rebuild_derived(&replicas[r], &w, N, L, cell);

            /*
             * PT diversification: when all replicas are cloned from the same
             * warm_init, they start at identical positions. Apply a small
             * per-polygon random perturbation using each replica's own RNG so
             * that replicas explore different regions from the first epoch.
             * Replica 0 is left unperturbed as the reference cold chain.
             */
            if (warm_init && r > 0) {
                RNG div_rng;
                div_rng.s = replicas[r].rng_s;
                double scale = 0.05;
                for (int k = 0; k < N; k++) {
                    replicas[r].cx[k] += rng_uniform(&div_rng, -scale, scale);
                    replicas[r].cy[k] += rng_uniform(&div_rng, -scale, scale);
                    double dth = rng_uniform(&div_rng, -0.5, 0.5);
                    replicas[r].th[k] = fmod(replicas[r].th[k] + dth + 2.0 * M_PI, 2.0 * M_PI);
                }
                replicas[r].rng_s = div_rng.s;
                rebuild_derived(&replicas[r], &w, N, L, cell);
            }

            evaluate_full(&replicas[r], &w, N, L, &cfg->weights, cfg->eps_feas);
        }

#ifdef PT_DEBUG
        if (probe_idx == 0) {
#pragma omp single
            {
                printf("PT DEBUG INIT probe=%d R=%d\n", probe_idx, R);
                for (int dbg_r = 0; dbg_r < R; dbg_r++) {
                    double E = pt_exchange_energy(&replicas[dbg_r]);
                    printf("PT DEBUG INIT r=%d T=%.4f step_xy=%.6f step_th=%.6f "
                           "E=%.6f ov=%.6f out=%.6f cx0=%.4f cy0=%.4f th0=%.4f\n",
                           dbg_r, replicas[dbg_r].temp,
                           replicas[dbg_r].step_xy, replicas[dbg_r].step_th,
                           E, replicas[dbg_r].overlap_penalty, replicas[dbg_r].outside_penalty,
                           N > 0 ? replicas[dbg_r].cx[0] : 0.0,
                           N > 0 ? replicas[dbg_r].cy[0] : 0.0,
                           N > 0 ? replicas[dbg_r].th[0] : 0.0);
                }
            }
        }
#endif

        int parity = 0;
        int K_epoch = (cfg->pt_K_epoch > 0) ? cfg->pt_K_epoch : 200 * N;
        int K_chunk = K_epoch / 10;
        if (K_chunk < 1) K_chunk = 1;
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
                /*
                 * Rebuild the workspace from this replica's current positions
                 * before local search. With R > num_threads a single thread
                 * processes multiple replicas sequentially; without a rebuild
                 * the workspace (grid, aabb, world) retains the previous
                 * replica's geometry and evaluate_full would compute the wrong
                 * energy for this replica.
                 */
                rebuild_derived(&replicas[r], &w, N, L, cell);
                /*
                 * Temperature dependency: pp.T_start = pp.T_end = temps[r],
                 * so run_sa_epoch runs isothermally at this replica's ladder
                 * temperature. The Metropolis criterion inside run_sa_epoch is
                 * exp(-dE / temp) where temp = temps[r], giving each replica a
                 * distinct acceptance rate. Higher-temperature replicas accept
                 * uphill moves more readily, allowing barrier-crossing; the
                 * cold replica (temps[0]) converges toward low-energy states.
                 */
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
                    /* PT_DEBUG: first-swap log fires once per probe; flag is set on first log and never reset. */
#ifdef PT_DEBUG
                    if (probe_idx == 0 && !debug_logged_swap_this_probe) {
                        printf("PT DEBUG PRE-SWAP probe=%d parity=%d all replicas:\n", probe_idx, parity);
                        for (int dbg_r = 0; dbg_r < R; dbg_r++) {
                            printf("  r=%d T=%.4f E=%.6f ov=%.6f out=%.6f cx0=%.4f cy0=%.4f th0=%.4f\n",
                                   dbg_r, temps[dbg_r], pt_exchange_energy(&replicas[dbg_r]),
                                   replicas[dbg_r].overlap_penalty, replicas[dbg_r].outside_penalty,
                                   N > 0 ? replicas[dbg_r].cx[0] : 0.0,
                                   N > 0 ? replicas[dbg_r].cy[0] : 0.0,
                                   N > 0 ? replicas[dbg_r].th[0] : 0.0);
                        }
                    }
#endif
                    for (int i = start; i + 1 < R; i += 2) {
                        int j = i + 1;
                        double Ti = temps[i];
                        double Tj = temps[j];
                        double Ei = pt_exchange_energy(&replicas[i]);
                        double Ej = pt_exchange_energy(&replicas[j]);
                        double acc = pt_swap_accept_prob(Ei, Ej, Ti, Tj);
#ifdef PT_DEBUG
                        double beta_i = (Ti > 0.0) ? (1.0 / Ti) : 0.0;
                        double beta_j = (Tj > 0.0) ? (1.0 / Tj) : 0.0;
                        double delta = (beta_i - beta_j) * (Ej - Ei);
#endif
                        double u = rng_u01(&swap_rng);
                        int accepted = (u < acc);
                        swap_attempts++;
                        if (accepted) {
                            swap_accepts++;
                            ReplicaState tmp = replicas[i];
                            replicas[i] = replicas[j];
                            replicas[j] = tmp;
                            replicas[i].replica_id = i;
                            replicas[j].replica_id = j;
                            replicas[i].temp = temps[i];
                            replicas[j].temp = temps[j];
                        }
#ifdef PT_DEBUG
                        if (!debug_logged_swap_this_probe) {
                            printf("PT DEBUG probe=%d parity=%d pair=(%d,%d) Ti=%.6f Tj=%.6f Ei=%.6f Ej=%.6f beta_i=%.6f beta_j=%.6f delta=%.6f acc=%.6f u=%.6f accepted=%d\n",
                                   probe_idx, parity, i, j, Ti, Tj, Ei, Ej,
                                   beta_i, beta_j, delta, acc, u, accepted);
                            debug_logged_swap_this_probe = 1;
                        }
#endif
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
        double ex = pt_exchange_energy(&replicas[r]);
        if (ex < global_min_energy)
            global_min_energy = ex;
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

#ifdef PT_DEBUG
    {
        double minE = 1e300, maxE = -1e300, sumE = 0.0;
        int feasible_count = 0;
        int distinct_energy_count = 0;
        for (int r = 0; r < R; r++) {
            double E = pt_exchange_energy(&replicas[r]);
            if (E < minE) minE = E;
            if (E > maxE) maxE = E;
            sumE += E;
            if (replicas[r].is_feasible) feasible_count++;
            int seen = 0;
            for (int k = 0; k < r; k++) {
                double Ek = pt_exchange_energy(&replicas[k]);
                if (fabs(E - Ek) < 1e-12) {
                    seen = 1;
                    break;
                }
            }
            if (!seen) distinct_energy_count++;
        }
        double meanE = (R > 0) ? (sumE / (double)R) : 0.0;
        double coldE = (R > 0) ? pt_exchange_energy(&replicas[0]) : 0.0;
        double accept_rate = (swap_attempts > 0) ? ((double)swap_accepts / (double)swap_attempts) : 0.0;
        printf("PT DEBUG SUMMARY probe=%d attempts=%d accepts=%d rate=%.6f "
               "minE=%.6f maxE=%.6f meanE=%.6f distinctE=%d "
               "coldE=%.6f feasible_count=%d cold_feasible=%d\n",
               probe_idx, swap_attempts, swap_accepts, accept_rate,
               minE, maxE, meanE, distinct_energy_count,
               coldE, feasible_count, (R > 0) ? replicas[0].is_feasible : 0);
    }
#else
    printf("PT probe %d: swap_attempts=%d swap_accepts=%d\n", probe_idx, swap_attempts, swap_accepts);
#endif

    free(replicas);
    free(temps);
}
