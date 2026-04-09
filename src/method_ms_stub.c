#include "../include/bisection.h"
#include "../include/annealing.h"
#include "../include/geometry.h"
#include "../include/utils.h"
#include <string.h>

void runner_ms_stub(
    const StudyConfig *cfg,
    double L,
    double slice_budget_sec,
    int probe_idx,
    const ReplicaState *warm_init,
    SliceResult *out)
{
    memset(out, 0, sizeof(*out));
    int N = cfg->N;

    ReplicaState r;
    if (warm_init) {
        r = *warm_init;
    } else {
        uint64_t rseed = make_trial_seed(cfg->seed, cfg->run_id, (uint64_t)probe_idx * 1000ULL);
        replica_init_random(&r, N, L, rseed, 1.0, 0);
    }

    Workspace w;
    double cell = base_bounding_radius() * 2.0;
    workspace_init(&w, N, L, cell);
    rebuild_derived(&r, &w, N, L, cell);

    Weights local_w = cfg->weights;
    evaluate_full(&r, &w, N, L, &local_w, cfg->eps_feas);

    double min_energy = r.energy;
    double min_feas = r.feas;

    ReplicaState best = r;
    double best_feas_metric = r.feas;
    int found_feasible = r.is_feasible ? 1 : 0;

    PhaseParams pp = {
        .iters = 0,
        .T_start = 1.0, .T_end = 1e-5,
        .step_xy_start = 0.05, .step_th_start = 0.5,
        .adapt_window = 2000, .acc_low = 0.4, .acc_high = 0.6,
        .step_shrink = 0.95, .step_grow = 1.05,
        .step_xy_min = 1e-5, .step_xy_max = 2.0,
        .step_th_min = 1e-4, .step_th_max = 3.14159265358979323846,
        .lambda_start = local_w.lambda_ov,
        .mu_start = local_w.mu_out,
        .ramp_every = 0, .ramp_factor = 1.0,
        .lambda_max = local_w.lambda_ov,
        .mu_max = local_w.mu_out,
        .p_reinsert = 0.0, .p_rotmix = 0.0,
        .log_every = 0
    };

    if (r.step_xy <= 0.0) r.step_xy = pp.step_xy_start;
    if (r.step_th <= 0.0) r.step_th = pp.step_th_start;

    double t_start = now_seconds();
    int K_batch = 500;
    volatile int early_stop = 0;

    while (1) {
        double elapsed = now_seconds() - t_start;
        if (elapsed >= slice_budget_sec) break;

        run_sa_epoch(&r, &w, N, L, &local_w, cfg->eps_feas, &pp, K_batch, &early_stop);
        evaluate_full(&r, &w, N, L, &local_w, cfg->eps_feas);

        if (r.energy < min_energy) min_energy = r.energy;
        if (r.feas < min_feas) min_feas = r.feas;

        if (r.is_feasible) {
            if (!found_feasible || r.energy < best.energy) {
                best = r;
            }
            found_feasible = 1;
        } else if (!found_feasible && r.feas < best_feas_metric) {
            best = r;
            best_feas_metric = r.feas;
        }

        if (found_feasible) break;
    }

    out->feasible = found_feasible;
    out->min_energy = min_energy;
    out->min_feas = min_feas;
    out->best_state = best;
    out->has_state = 1;
    out->slice_used_sec = now_seconds() - t_start;
    out->resample_events = 0;

    workspace_free(&w);
}
