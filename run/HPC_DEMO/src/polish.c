#include "../include/polish.h"
#include "../include/methods.h"
#include "../include/replica.h"
#include "../include/utils.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

void runner_polish(StudyConfig *cfg, double L_best, double total_budget_sec,
                   double stall_threshold_sec, double shave_budget_sec,
                   int use_warm_start, const ReplicaState *init_state,
                   SliceResult *out_res)
{
    memset(out_res, 0, sizeof(*out_res));

    /* Force R=20 for polish */
    int saved_R = cfg->R;
    cfg->R = 20;

    double t_start = now_seconds();
    double last_improve_time = t_start;

    /* Track the absolute best strictly feasible state */
    int have_best_feasible = 0;
    ReplicaState best_feasible_state;
    memset(&best_feasible_state, 0, sizeof(best_feasible_state));
    best_feasible_state.energy = 1e30;

    /* Current warm-start state for successive slices */
    int warm = use_warm_start;
    ReplicaState warm_state;
    if (warm && init_state) {
        warm_state = *init_state;
    }

    double saved_mu_out = cfg->weights.mu_out;
    int probe_idx = 0;

    while (1) {
        double elapsed = now_seconds() - t_start;
        double remaining = total_budget_sec - elapsed;
        if (remaining <= 0.0) break;

        /* Check for stall → trigger shave */
        double since_improve = now_seconds() - last_improve_time;
        if (since_improve >= stall_threshold_sec && stall_threshold_sec > 0.0) {
            /* --- Stochastic Shave --- */
            double shave_slice = shave_budget_sec;
            if (shave_slice > remaining) shave_slice = remaining;
            if (shave_slice <= 0.0) break;

            /* Disable outside penalty */
            cfg->weights.mu_out = 0.0;

            SliceResult shave_res;
            runner_erms(cfg, L_best, shave_slice, probe_idx++,
                        warm ? &warm_state : NULL, &shave_res);

            /* Restore outside penalty */
            cfg->weights.mu_out = saved_mu_out;

            /* Update warm-start from shave result (even if not feasible) */
            if (shave_res.has_state) {
                warm_state = shave_res.best_state;
                warm = 1;
            }

            /* Reset stall clock */
            last_improve_time = now_seconds();
            continue;
        }

        /* Normal polish slice */
        double slice_sec = 5.0;
        if (slice_sec > remaining) slice_sec = remaining;
        if (slice_sec <= 0.0) break;

        SliceResult res;
        runner_erms(cfg, L_best, slice_sec, probe_idx++,
                    warm ? &warm_state : NULL, &res);

        if (res.has_state) {
            /* Update warm-start for next slice */
            warm_state = res.best_state;
            warm = 1;

            /* Track best strictly feasible */
            if (res.feasible) {
                if (!have_best_feasible ||
                    res.min_energy < best_feasible_state.energy) {
                    best_feasible_state = res.best_state;
                    have_best_feasible = 1;
                    last_improve_time = now_seconds();
                }
            }
        }
    }

    /* Restore R */
    cfg->R = saved_R;

    /* Populate output with the best feasible state if we have one,
       otherwise the last warm state */
    if (have_best_feasible) {
        out_res->feasible = 1;
        out_res->min_energy = best_feasible_state.energy;
        out_res->min_feas = best_feasible_state.overlap_penalty +
                            best_feasible_state.outside_penalty;
        out_res->best_state = best_feasible_state;
        out_res->has_state = 1;
    } else if (warm) {
        out_res->feasible = 0;
        out_res->min_energy = warm_state.energy;
        out_res->min_feas = warm_state.overlap_penalty +
                            warm_state.outside_penalty;
        out_res->best_state = warm_state;
        out_res->has_state = 1;
    }
    out_res->slice_used_sec = now_seconds() - t_start;
    out_res->resample_events = 0;
}
