#include "../include/methods.h"
#include "../include/replica.h"
#include "../include/annealing.h"
#include "../include/geometry.h"
#include "../include/utils.h"
#include <omp.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

void runner_ms(
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
    Workspace *workspaces = (Workspace *)calloc((size_t)R, sizeof(Workspace));
    ReplicaState *best_states = (ReplicaState *)calloc((size_t)R, sizeof(ReplicaState));
    double *best_feas_metric = (double *)calloc((size_t)R, sizeof(double));
    double *best_energy = (double *)calloc((size_t)R, sizeof(double));
    double *min_energy = (double *)calloc((size_t)R, sizeof(double));
    double *min_feas = (double *)calloc((size_t)R, sizeof(double));
    int *has_feasible = (int *)calloc((size_t)R, sizeof(int));

    if (!replicas || !workspaces || !best_states || !best_feas_metric || !best_energy ||
        !min_energy || !min_feas || !has_feasible) {
        free(replicas);
        free(workspaces);
        free(best_states);
        free(best_feas_metric);
        free(best_energy);
        free(min_energy);
        free(min_feas);
        free(has_feasible);
        return;
    }

    for (int r = 0; r < R; r++) {
        workspace_init(&workspaces[r], N, L, cell);
        best_feas_metric[r] = 1e30;
        best_energy[r] = 1e30;
        min_energy[r] = 1e30;
        min_feas[r] = 1e30;
        has_feasible[r] = 0;
    }

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

    volatile int early_stop = 0;
    double t_start = now_seconds();
    int K_batch = 500;

#pragma omp parallel num_threads(R)
    {
        int tid = omp_get_thread_num();
        ReplicaState *my_r = &replicas[tid];
        Workspace *my_w = &workspaces[tid];
        Weights local_w = cfg->weights;

        if (warm_init) {
            uint64_t rseed = make_trial_seed(cfg->seed, (uint64_t)probe_idx * 1000ULL + (uint64_t)tid, cfg->run_id);
            replica_clone(my_r, warm_init, rseed);
            my_r->replica_id = tid;
        } else {
            uint64_t rseed = make_trial_seed(cfg->seed, (uint64_t)probe_idx * 1000ULL + (uint64_t)tid, cfg->run_id);
            replica_init_random(my_r, N, L, rseed, 1.0, tid);
        }

        rebuild_derived(my_r, my_w, N, L, cell);
        evaluate_full(my_r, my_w, N, L, &local_w, cfg->eps_feas);

        if (my_r->step_xy <= 0.0) my_r->step_xy = pp.step_xy_start;
        if (my_r->step_th <= 0.0) my_r->step_th = pp.step_th_start;

        double my_min_energy = my_r->energy;
        double my_min_feas = my_r->feas;
        ReplicaState my_best = *my_r;
        double my_best_feas_metric = my_r->feas;
        int my_has_feasible = my_r->is_feasible ? 1 : 0;

        while (!early_stop) {
            double elapsed = now_seconds() - t_start;
            if (elapsed >= slice_budget_sec) break;

            run_sa_epoch(my_r, my_w, N, L, &local_w, cfg->eps_feas, &pp, K_batch, &early_stop);
            evaluate_full(my_r, my_w, N, L, &local_w, cfg->eps_feas);

            if (my_r->energy < my_min_energy) my_min_energy = my_r->energy;
            if (my_r->feas < my_min_feas) my_min_feas = my_r->feas;

            if (my_r->is_feasible) {
                if (!my_has_feasible || my_r->energy < my_best.energy) {
                    my_best = *my_r;
                }
                my_has_feasible = 1;
            } else if (!my_has_feasible && my_r->feas < my_best_feas_metric) {
                my_best = *my_r;
                my_best_feas_metric = my_r->feas;
            }

            if (my_r->is_feasible) {
                early_stop = 1;
            }
        }

        best_states[tid] = my_best;
        best_feas_metric[tid] = my_best_feas_metric;
        best_energy[tid] = my_best.energy;
        min_energy[tid] = my_min_energy;
        min_feas[tid] = my_min_feas;
        has_feasible[tid] = my_has_feasible;
    }

    ReplicaState global_best;
    memset(&global_best, 0, sizeof(global_best));
    global_best.energy = 1e30;
    global_best.overlap_penalty = 1e30;
    global_best.outside_penalty = 1e30;
    global_best.is_feasible = 0;
    global_best.replica_id = -1;

    int global_feasible = 0;
    double global_min_energy = 1e30;
    double global_min_feas = 1e30;

    for (int r = 0; r < R; r++) {
        if (min_energy[r] < global_min_energy) global_min_energy = min_energy[r];
        if (min_feas[r] < global_min_feas) global_min_feas = min_feas[r];
        if (has_feasible[r]) global_feasible = 1;

        int take = 0;
        if (has_feasible[r] && !global_best.is_feasible) {
            take = 1;
        } else if (has_feasible[r] == global_best.is_feasible) {
            double my_fm = best_feas_metric[r];
            double gb_fm = global_best.overlap_penalty + global_best.outside_penalty;
            if (my_fm < gb_fm) take = 1;
            else if (my_fm == gb_fm && best_energy[r] < global_best.energy) take = 1;
            else if (my_fm == gb_fm && best_energy[r] == global_best.energy && r < global_best.replica_id) take = 1;
        }

        if (take) {
            global_best = best_states[r];
            global_best.replica_id = r;
        }
    }

    out->feasible = global_feasible;
    out->min_energy = global_min_energy;
    out->min_feas = global_min_feas;
    out->best_state = global_best;
    out->has_state = (global_best.replica_id >= 0);
    out->slice_used_sec = now_seconds() - t_start;
    out->resample_events = 0;

    for (int r = 0; r < R; r++) {
        workspace_free(&workspaces[r]);
    }
    free(replicas);
    free(workspaces);
    free(best_states);
    free(best_feas_metric);
    free(best_energy);
    free(min_energy);
    free(min_feas);
    free(has_feasible);
}
