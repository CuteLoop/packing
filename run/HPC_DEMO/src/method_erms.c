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

static int erms_K_epoch(int N) {
    return 200 * N;
}

static int erms_k_elite(int R) {
    int k = (int)ceil(0.25 * (double)R);
    if (k < 1) k = 1;
    if (k > R) k = R;
    return k;
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

static double gaussian_sample(RNG *rng, double mean, double stddev) {
    double u1 = rng_u01(rng);
    double u2 = rng_u01(rng);
    if (u1 < 1e-30) u1 = 1e-30;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + stddev * z;
}

static void perturb_replica(ReplicaState *r, int N, double L,
                            double sigma_pos, double sigma_theta)
{
    RNG rng;
    rng.s = r->rng_s;
    double half = 0.5 * L;

    for (int i = 0; i < N; i++) {
        r->cx[i] += gaussian_sample(&rng, 0.0, sigma_pos);
        r->cy[i] += gaussian_sample(&rng, 0.0, sigma_pos);
        r->th[i] += gaussian_sample(&rng, 0.0, sigma_theta);

        if (r->cx[i] < -half) r->cx[i] = -half;
        if (r->cx[i] >  half) r->cx[i] =  half;
        if (r->cy[i] < -half) r->cy[i] = -half;
        if (r->cy[i] >  half) r->cy[i] =  half;

        r->th[i] = fmod(r->th[i], 2.0 * M_PI);
        if (r->th[i] < 0.0) r->th[i] += 2.0 * M_PI;
    }

    r->rng_s = rng.s;
    r->energy = 1e30;
    r->overlap_penalty = 1e30;
    r->outside_penalty = 1e30;
    r->feas = 1e30;
    r->is_feasible = 0;
}

typedef struct {
    int consecutive_collapsed;
    int boost_next;
} AntiCollapseState;

static void check_anti_collapse(AntiCollapseState *ac,
                                const ReplicaState *replicas,
                                const int *sorted_idx, int k_elite)
{
    if (k_elite < 2) return;

    double E_min = replicas[sorted_idx[0]].energy;
    double E_max = replicas[sorted_idx[k_elite - 1]].energy;
    double E_mean = 0.0;
    for (int i = 0; i < k_elite; i++) {
        E_mean += replicas[sorted_idx[i]].energy;
    }
    E_mean /= (double)k_elite;

    double denom = fabs(E_mean);
    if (denom < 1.0) denom = 1.0;
    double spread = (E_max - E_min) / denom;

    if (spread < 1e-6) {
        ac->consecutive_collapsed++;
        if (ac->consecutive_collapsed >= 2) {
            ac->boost_next = 1;
            ac->consecutive_collapsed = 0;
        }
    } else {
        ac->consecutive_collapsed = 0;
        ac->boost_next = 0;
    }
}

void runner_erms(
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
    int K_epoch = erms_K_epoch(N);
    int K_chunk = 20 * N;
    if (K_chunk > K_epoch) K_chunk = K_epoch;

    int k_elite = erms_k_elite(R);
    int k_protect = k_elite;

    PhaseParams pp = {
        .iters = 0,
        .T_start = 1.0, .T_end = 1e-5,
        .step_xy_start = 0.05, .step_th_start = 0.5,
        .adapt_window = 2000, .acc_low = 0.4, .acc_high = 0.6,
        .step_shrink = 0.95, .step_grow = 1.05,
        .step_xy_min = 1e-5, .step_xy_max = 2.0,
        .step_th_min = 1e-4, .step_th_max = 3.14159265358979323846,
        .lambda_start = cfg->weights.lambda_ov,
        .mu_start = cfg->weights.mu_out,
        .ramp_every = 0, .ramp_factor = 1.0,
        .lambda_max = cfg->weights.lambda_ov,
        .mu_max = cfg->weights.mu_out,
        .p_reinsert = 0.0, .p_rotmix = 0.0,
        .log_every = 0
    };

    ReplicaState *replicas = (ReplicaState *)calloc((size_t)R, sizeof(ReplicaState));
    if (!replicas) return;

    for (int r = 0; r < R; r++) {
        uint64_t trial_id = (uint64_t)probe_idx * 1000000ULL + (uint64_t)r;
        uint64_t rseed = make_trial_seed(cfg->seed, cfg->run_id, trial_id);
        if (warm_init) {
            replica_clone(&replicas[r], warm_init, rseed);
            replicas[r].replica_id = r;
        } else {
            replica_init_random(&replicas[r], N, L, rseed, 1.0, r);
        }
    }

    int early_stop = 0;
    int global_stop = 0;
    int resample_count = 0;
    AntiCollapseState anti_collapse = {0, 0};

    int *sorted_idx = (int *)calloc((size_t)R, sizeof(int));
    int *victims = (int *)calloc((size_t)R, sizeof(int));
    int n_victims = 0;

    double t_start = now_seconds();

#pragma omp parallel
    {
        Workspace w;
        workspace_init(&w, N, L, cell);

#pragma omp for schedule(static)
        for (int r = 0; r < R; r++) {
            rebuild_derived(&replicas[r], &w, N, L, cell);
            evaluate_full(&replicas[r], &w, N, L, &cfg->weights, cfg->eps_feas);
        }

        while (1) {
    #pragma omp single
                {
                    int stop = 0;
                    if (now_seconds() - t_start >= slice_budget_sec) stop = 1;
                    if (early_stop) stop = 1;
                    global_stop = stop;
                }
                if (global_stop) break;

#pragma omp for schedule(static)
            for (int r = 0; r < R; r++) {
                Weights local_w = cfg->weights;
                run_sa_epoch(&replicas[r], &w, N, L,
                             &local_w, cfg->eps_feas,
                             &pp, K_chunk, &early_stop);
                evaluate_full(&replicas[r], &w, N, L,
                              &local_w, cfg->eps_feas);

                if (replicas[r].is_feasible) {
#pragma omp atomic write
                    early_stop = 1;
                }
            }

#pragma omp single
            {
                int stop = 0;
                if (now_seconds() - t_start >= slice_budget_sec) stop = 1;
                if (early_stop) stop = 1;
                global_stop = stop;

                if (!global_stop && k_protect < R) {
                    resample_count++;

                    for (int i = 0; i < R; i++) sorted_idx[i] = i;
                    for (int i = 1; i < R; i++) {
                        int key = sorted_idx[i];
                        int j = i - 1;
                        while (j >= 0 &&
                               replica_better_than(&replicas[key],
                                                   &replicas[sorted_idx[j]])) {
                            sorted_idx[j + 1] = sorted_idx[j];
                            j--;
                        }
                        sorted_idx[j + 1] = key;
                    }

                    double f = (now_seconds() - t_start) / slice_budget_sec;
                    if (f > 1.0) f = 1.0;
                    double sigma_pos = (0.02 - 0.015 * f) * L;
                    double sigma_theta = (5.0 - 4.0 * f) * (M_PI / 180.0);

                    check_anti_collapse(&anti_collapse, replicas, sorted_idx, k_elite);
                    if (anti_collapse.boost_next) {
                        sigma_pos *= 2.0;
                        sigma_theta *= 2.0;
                        anti_collapse.boost_next = 0;
                    }

                    n_victims = R - k_protect;
                    int clone_src = 0;
                    for (int i = 0; i < n_victims; i++) {
                        int victim = sorted_idx[k_protect + i];
                        int source = sorted_idx[clone_src % k_elite];

                        uint64_t clone_seed = make_trial_seed(
                            cfg->seed,
                            cfg->run_id,
                            (uint64_t)probe_idx * 100000ULL +
                            (uint64_t)resample_count * 1000ULL +
                            (uint64_t)victim
                        );

                        replica_clone(&replicas[victim], &replicas[source], clone_seed);
                        replicas[victim].replica_id = victim;
                        perturb_replica(&replicas[victim], N, L, sigma_pos, sigma_theta);
                        replicas[victim].epoch_proposals = 0;

                        victims[i] = victim;
                        clone_src++;
                    }
                } else {
                    n_victims = 0;
                }
            }

            if (global_stop) break;

#pragma omp for schedule(static)
            for (int i = 0; i < n_victims; i++) {
                int v = victims[i];
                rebuild_derived(&replicas[v], &w, N, L, cell);
                evaluate_full(&replicas[v], &w, N, L,
                              &cfg->weights, cfg->eps_feas);
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
    out->resample_events = resample_count;

    free(replicas);
    free(sorted_idx);
    free(victims);
}
