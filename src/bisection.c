#include "../include/bisection.h"
#include "../include/geometry.h"
#include "../include/utils.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static double get_t_base(int N) {
    if (N <= 20) return 6.0;
    if (N <= 50) return 10.0;
    if (N <= 100) return 15.0;
    return 60.0;
}

static double get_slice_budget(int N, int probe_idx) {
    if (N >= 200) {
        if (probe_idx < 6) return 60.0;
        if (probe_idx < 12) return 120.0;
        return 240.0;
    }

    double base = get_t_base(N);
    if (probe_idx < 5) return base;
    if (probe_idx < 10) return 2.0 * base;
    return 4.0 * base;
}

static void log_progress_row(FILE *f, const StudyConfig *cfg,
                             double wall_sec, int probe_idx,
                             double L_current, double best_energy,
                             double best_feas, int feasible_ever,
                             double L_best, const char *event)
{
    fprintf(f, "%llu,%llu,%s,%d,%d,%.2f,%d,%.6f,%.6e,%.6e,%d,",
            (unsigned long long)cfg->run_id,
            (unsigned long long)cfg->seed,
            cfg->method, cfg->N, cfg->R,
            wall_sec, probe_idx, L_current, best_energy, best_feas,
            feasible_ever);
    if (feasible_ever) {
        fprintf(f, "%.6f", L_best);
    }
    fprintf(f, ",%s\n", event);
    fflush(f);
}

BisectionResult bisection_run(const StudyConfig *cfg, method_runner_fn runner) {
    BisectionResult result;
    memset(&result, 0, sizeof(result));

    int N = cfg->N;
    double eps_feas = cfg->eps_feas;

    double A_poly = base_polygon_area();
    double L_lo = sqrt((double)N * A_poly);
    double gamma = (N <= 20) ? 2.2 : (N <= 50) ? 2.6 : 3.0;
    double L_hi = gamma * L_lo;

    ReplicaState warm_state;
    int have_warm = 0;
    double L_best = NAN;
    int feasible_ever = 0;
    double best_energy_global = 1e30;

    Workspace repair_ws;
    double cell = base_bounding_radius() * 2.0;
    workspace_init(&repair_ws, N, L_hi, cell);

    char bis_path[512];
    char log_path[512];
    snprintf(bis_path, sizeof(bis_path), "%s_%s_N%03d_s%llu_bisection.csv",
             cfg->out_prefix, cfg->method, N, (unsigned long long)cfg->seed);
    snprintf(log_path, sizeof(log_path), "%s_%s_N%03d_s%llu_log.csv",
             cfg->out_prefix, cfg->method, N, (unsigned long long)cfg->seed);

    FILE *f_bis = fopen(bis_path, "w");
    FILE *f_log = fopen(log_path, "w");
    if (!f_bis || !f_log) {
        if (f_bis) fclose(f_bis);
        if (f_log) fclose(f_log);
        workspace_free(&repair_ws);
        result.L_best = NAN;
        result.L_lo = L_lo;
        result.L_hi = L_hi;
        result.probes_done = 0;
        result.feasible_found = 0;
        return result;
    }

    fprintf(f_bis, "run_id,seed,method,N,R,probe_idx,wall_sec_start,wall_sec_end,"
                   "L_lo,L_hi,L_mid,slice_budget_sec,slice_used_sec,"
                   "feasible,min_energy,min_feas,resample_events,L_best,bracket_width\n");
    fprintf(f_log, "run_id,seed,method,N,R,wall_sec,probe_idx,"
                   "L_current,best_energy,best_feas,feasible_ever,L_best,event\n");
    fflush(f_bis);
    fflush(f_log);

    double t_run_start = now_seconds();
    int probe_idx = 0;
    double L_prev_feasible = L_hi;

    while (1) {
        double elapsed = now_seconds() - t_run_start;
        if (elapsed >= cfg->time_budget_sec) break;
        if ((L_hi - L_lo) <= 1e-3 * L_hi && feasible_ever) break;

        double L_mid = 0.5 * (L_lo + L_hi);
        double t_slice = get_slice_budget(N, probe_idx);

        double remaining = cfg->time_budget_sec - elapsed;
        if (t_slice > remaining) t_slice = remaining;
        if (t_slice < 1.0) break;

        log_progress_row(f_log, cfg, now_seconds() - t_run_start, probe_idx,
                         L_mid, best_energy_global, 0.0, feasible_ever,
                         L_best, "probe_start");

        ReplicaState init_state;
        double repair_time = 0.0;

        if (have_warm) {
            replica_clone(&init_state, &warm_state,
                          make_trial_seed(cfg->seed, cfg->run_id, (uint64_t)probe_idx * 1000ULL));
            warmstart_scale(&init_state, N, L_prev_feasible, L_mid);

            double repair_budget = fmin(0.5, 0.05 * t_slice);
            double repair_start = now_seconds();
            rebuild_derived(&init_state, &repair_ws, N, L_mid, cell);
            warmstart_repair(&init_state, &repair_ws, N, L_mid,
                             &cfg->weights, eps_feas, repair_budget);
            repair_time = now_seconds() - repair_start;
        } else {
            uint64_t rseed = make_trial_seed(cfg->seed, cfg->run_id, (uint64_t)probe_idx * 1000ULL);
            replica_init_random(&init_state, N, L_mid, rseed, 1.0, 0);
        }

        double adjusted_slice = t_slice - repair_time;
        if (adjusted_slice < 0.5) adjusted_slice = 0.5;

        double probe_wall_start = now_seconds();
        SliceResult res;
        memset(&res, 0, sizeof(res));
        res.min_energy = 1e30;
        res.min_feas = 1e30;

        runner(cfg, L_mid, adjusted_slice, probe_idx, &init_state, &res);

        double probe_wall_end = now_seconds();

        if (res.min_energy < best_energy_global) {
            best_energy_global = res.min_energy;
        }

        if (res.feasible) {
            L_hi = L_mid;
            L_best = L_mid;
            feasible_ever = 1;
            L_prev_feasible = L_mid;
            if (res.has_state) {
                replica_clone(&warm_state, &res.best_state,
                              make_trial_seed(cfg->seed, cfg->run_id, (uint64_t)probe_idx + 77777ULL));
                have_warm = 1;
            }
        } else {
            L_lo = L_mid;
        }

        if (N >= 200 && probe_idx == 2 && !feasible_ever) {
            L_hi *= 1.5;
        }

        fprintf(f_bis,
            "%llu,%llu,%s,%d,%d,"
            "%d,%.2f,%.2f,"
            "%.6f,%.6f,%.6f,"
            "%.2f,%.2f,"
            "%d,%.6e,%.6e,%d,",
                (unsigned long long)cfg->run_id,
                (unsigned long long)cfg->seed,
                cfg->method, N, cfg->R,
                probe_idx,
                probe_wall_start - t_run_start,
                probe_wall_end - t_run_start,
                L_lo, L_hi, L_mid,
                t_slice, res.slice_used_sec,
            res.feasible, res.min_energy, res.min_feas, res.resample_events);

        if (feasible_ever) {
            fprintf(f_bis, "%.6f,", L_best);
        } else {
            fprintf(f_bis, ",");
        }
        fprintf(f_bis, "%.6f\n", L_hi - L_lo);
        fflush(f_bis);

        log_progress_row(f_log, cfg, now_seconds() - t_run_start, probe_idx,
                         L_mid, best_energy_global, res.min_feas,
                         feasible_ever, L_best, "probe_end");

        probe_idx++;
    }

    fclose(f_bis);
    fclose(f_log);
    workspace_free(&repair_ws);

    result.L_best = feasible_ever ? L_best : L_hi;
    result.L_lo = L_lo;
    result.L_hi = L_hi;
    result.probes_done = probe_idx;
    result.feasible_found = feasible_ever;
    if (have_warm) result.best_state = warm_state;

    return result;
}
