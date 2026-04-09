// src/main.c
// ------------------------------------------------------------
// Refactored Entry Point
// Responsibilities: CLI Parsing, Signal Handling, Output, and High-Level Control Loop
// ------------------------------------------------------------

#define _POSIX_C_SOURCE 200809L // For sigaction, etc.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <signal.h>
#include <unistd.h>
#include <errno.h>

#include "../include/common.h"
#include "../include/config.h"
#include "../include/utils.h"
#include "../include/geometry.h"
#include "../include/spatial_hash.h"
#include "../include/physics.h"
#include "../include/annealing.h"
#include "../include/logger.h"
#include "../include/bisection.h"
#include "../include/methods.h"
#include "../include/polish.h"
#include <omp.h>

// --- Global Stop Flag for Signal Handling ---
volatile sig_atomic_t g_stop_requested = 0;

void handle_sigterm(int sig) {
    (void)sig;
    g_stop_requested = 1;
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// --- Helper: Write SVG Output ---
void write_svg(const char *path, State *s, double feas) {
    FILE *f = fopen(path, "w");
    if (!f) return;

    int W = 800, H = 800;
    fprintf(f, "<svg width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\" xmlns=\"http://www.w3.org/2000/svg\">\n", W, H, W, H);
    fprintf(f, "<rect width=\"%d\" height=\"%d\" fill=\"white\"/>\n", W, H);
    
    // Scale to fit L in the view with margin
    double scale = (W - 100) / s->L;
    double offset = 50.0;

    // Draw container box
    double box_px = s->L * scale;
    fprintf(f, "<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" fill=\"none\" stroke=\"black\" stroke-width=\"2\"/>\n", 
        offset, offset, box_px, box_px);

    // Draw Polygons
    for (int i = 0; i < s->N; i++) {
        fprintf(f, "<path d=\"");
        Vec2 *w = &s->world[i * NV];
        for (int k = 0; k < NV; k++) {
            double sx = offset + (w[k].x + s->L/2.0) * scale;
            double sy = offset + (s->L/2.0 - w[k].y) * scale; // Flip Y
            fprintf(f, "%s%.2f %.2f ", k==0?"M":"L", sx, sy);
        }
        fprintf(f, "Z\" fill=\"rgba(0,0,255,0.2)\" stroke=\"black\" stroke-width=\"1\"/>\n");
    }

    fprintf(f, "<text x=\"20\" y=\"20\" font-family=\"monospace\">N=%d L=%.6f Feas=%.2e</text>\n", s->N, s->L, feas);
    fprintf(f, "</svg>\n");
    fclose(f);
}

// --- Helper: Write CSV Output ---
void write_csv(const char *path, const char *prefix, uint64_t run_id, uint64_t seed, State *s, double feas) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "# prefix=%s run_id=%llu seed=%llu L=%.17g best_feas=%.17g N=%d\n",
            prefix ? prefix : "run",
            (unsigned long long)run_id,
            (unsigned long long)seed,
            s->L,
            feas,
            s->N);
    fprintf(f, "i,cx,cy,theta_rad\n");
    for (int i = 0; i < s->N; i++) {
        fprintf(f, "%d,%.17g,%.17g,%.17g\n", i, s->cx[i], s->cy[i], s->th[i]);
    }
    fclose(f);
}

static void write_best_csv_study(const StudyConfig *cfg, const ReplicaState *r, double L_best) {
    char path[512];
    snprintf(path, sizeof(path), "%s_%s_N%03d_s%llu_best_state.csv",
             cfg->out_prefix, cfg->method, cfg->N, (unsigned long long)cfg->seed);

    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "# prefix=%s method=%s run_id=%llu seed=%llu L=%.17g best_feas=%.17g N=%d\n",
            cfg->out_prefix,
            cfg->method,
            (unsigned long long)cfg->run_id,
            (unsigned long long)cfg->seed,
            L_best,
            r->feas,
            cfg->N);
    fprintf(f, "i,cx,cy,theta_rad\n");
    for (int i = 0; i < cfg->N; i++) {
        fprintf(f, "%d,%.17g,%.17g,%.17g\n", i, r->cx[i], r->cy[i], r->th[i]);
    }
    fclose(f);
}

static void write_best_svg_study(const StudyConfig *cfg, const ReplicaState *r, double L_best) {
    char path[512];
    snprintf(path, sizeof(path), "%s_%s_N%03d_s%llu_best_state.svg",
             cfg->out_prefix, cfg->method, cfg->N, (unsigned long long)cfg->seed);

    FILE *f = fopen(path, "w");
    if (!f) return;

    int W = 800, H = 800;
    fprintf(f, "<svg width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\" xmlns=\"http://www.w3.org/2000/svg\">\n", W, H, W, H);
    fprintf(f, "<rect width=\"%d\" height=\"%d\" fill=\"white\"/>\n", W, H);

    double scale = (W - 100) / L_best;
    double offset = 50.0;
    double box_px = L_best * scale;
    fprintf(f, "<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" fill=\"none\" stroke=\"black\" stroke-width=\"2\"/>\n",
            offset, offset, box_px, box_px);

    Vec2 world[NV];
    for (int i = 0; i < cfg->N; i++) {
        build_world_verts(world, r->cx[i], r->cy[i], r->th[i]);
        fprintf(f, "<path d=\"");
        for (int k = 0; k < NV; k++) {
            double sx = offset + (world[k].x + L_best / 2.0) * scale;
            double sy = offset + (L_best / 2.0 - world[k].y) * scale;
            fprintf(f, "%s%.2f %.2f ", k == 0 ? "M" : "L", sx, sy);
        }
        fprintf(f, "Z\" fill=\"rgba(0,0,255,0.2)\" stroke=\"black\" stroke-width=\"1\"/>\n");
    }

    fprintf(f, "<text x=\"20\" y=\"20\" font-family=\"monospace\">N=%d L=%.6f Feas=%.2e</text>\n",
            cfg->N, L_best, r->feas);
    fprintf(f, "</svg>\n");
    fclose(f);
}

void usage() {
    fprintf(stderr, "Usage: ./solver N trials [out_prefix] [seed] [run_id]\n");
    exit(1);
}

static void usage_study(void) {
    fprintf(stderr, "Study mode usage:\n");
    fprintf(stderr, "  ./bin/solver --study --method {ms,erms,pt} --R <int> --N <int> ");
    fprintf(stderr, "--time_budget_sec <sec> --seed <u64> --run_id <u64> --out_prefix <str>\n");
    fprintf(stderr, "  [--mode graph|hero] [--save_best]\n");
}

static void ensure_dir_recursive(const char *path) {
    char buf[512];
    size_t len = strlen(path);
    if (len == 0 || len >= sizeof(buf)) return;
    memcpy(buf, path, len + 1);

    for (size_t i = 1; i < len; i++) {
        if (buf[i] == '/') {
            buf[i] = '\0';
            ensure_dir(buf);
            buf[i] = '/';
        }
    }
    ensure_dir(buf);
}

static void ensure_out_prefix_dir(const char *prefix) {
    const char *slash = strrchr(prefix, '/');
    if (!slash) return;
    size_t len = (size_t)(slash - prefix);
    if (len == 0 || len >= 512) return;
    char buf[512];
    memcpy(buf, prefix, len);
    buf[len] = '\0';
    ensure_dir_recursive(buf);
}

static int run_study_mode(int argc, char **argv) {
    StudyConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.eps_feas = 1e-6;
    cfg.weights.alpha_L = 0.0;
    cfg.weights.lambda_ov = 1.0;
    cfg.weights.mu_out = 1.0;
    int save_best = 0;
    char mode[16] = "graph";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--study") == 0) {
            continue;
        } else if (strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
            strncpy(cfg.method, argv[++i], sizeof(cfg.method) - 1);
        } else if (strcmp(argv[i], "--R") == 0 && i + 1 < argc) {
            cfg.R = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--N") == 0 && i + 1 < argc) {
            cfg.N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--time_budget_sec") == 0 && i + 1 < argc) {
            cfg.time_budget_sec = atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            cfg.seed = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--run_id") == 0 && i + 1 < argc) {
            cfg.run_id = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--out_prefix") == 0 && i + 1 < argc) {
            strncpy(cfg.out_prefix, argv[++i], sizeof(cfg.out_prefix) - 1);
        } else if (strcmp(argv[i], "--eps_feas") == 0 && i + 1 < argc) {
            cfg.eps_feas = atof(argv[++i]);
        } else if (strcmp(argv[i], "--save_best") == 0) {
            save_best = 1;
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            strncpy(mode, argv[++i], sizeof(mode) - 1);
        }
    }

    if (cfg.N <= 0 || cfg.R <= 0 || cfg.time_budget_sec <= 0.0 ||
        cfg.method[0] == '\0' || cfg.out_prefix[0] == '\0') {
        fprintf(stderr, "Missing required study mode flags.\n");
        usage_study();
        return 1;
    }

    method_runner_fn runner = NULL;
    if (strcmp(cfg.method, "ms") == 0) {
        runner = runner_ms;
    } else if (strcmp(cfg.method, "erms") == 0) {
        runner = runner_erms;
    } else if (strcmp(cfg.method, "pt") == 0) {
        runner = runner_pt;
    } else {
        fprintf(stderr, "Method '%s' not yet implemented.\n", cfg.method);
        return 1;
    }

    ensure_out_prefix_dir(cfg.out_prefix);

    omp_set_num_threads(cfg.R);

    if (strcmp(mode, "hero") == 0) {
        /* Hero mode: bisection first, then polish best L */
        BisectionResult result = bisection_run(&cfg, runner);

        printf("Bisection phase: probes=%d feasible=%d L_best=%.6f bracket=[%.6f, %.6f]\n",
               result.probes_done, result.feasible_found,
               result.feasible_found ? result.L_best : -1.0,
               result.L_lo, result.L_hi);

        if (result.feasible_found) {
            double remaining = cfg.time_budget_sec - (result.probes_done * 10.0);
            if (remaining < 30.0) remaining = 30.0;

            printf("Polish phase: L_best=%.6f budget=%.1fs\n", result.L_best, remaining);

            SliceResult polish_res;
            runner_polish(&cfg, result.L_best, remaining,
                          600.0, 1.0,
                          1, &result.best_state, &polish_res);

            printf("Hero complete: feasible=%d energy=%.6e elapsed=%.1fs\n",
                   polish_res.feasible, polish_res.min_energy, polish_res.slice_used_sec);

            if (save_best && polish_res.has_state) {
                write_best_csv_study(&cfg, &polish_res.best_state, result.L_best);
                write_best_svg_study(&cfg, &polish_res.best_state, result.L_best);
                printf("Best SVG: %s_%s_N%03d_s%llu_best_state.svg\n",
                       cfg.out_prefix, cfg.method, cfg.N, (unsigned long long)cfg.seed);
            }
        } else {
            printf("Hero mode: bisection found no feasible solution; skipping polish.\n");
        }
    } else {
        /* Graph mode: bisection only */
        BisectionResult result = bisection_run(&cfg, runner);

        printf("Bisection complete: probes=%d feasible=%d L_best=%.6f bracket=[%.6f, %.6f]\n",
               result.probes_done, result.feasible_found,
               result.feasible_found ? result.L_best : -1.0,
               result.L_lo, result.L_hi);

        if (save_best && result.feasible_found) {
            write_best_csv_study(&cfg, &result.best_state, result.L_best);
            write_best_svg_study(&cfg, &result.best_state, result.L_best);
            printf("Best SVG: %s_%s_N%03d_s%llu_best_state.svg\n",
                   cfg.out_prefix, cfg.method, cfg.N, (unsigned long long)cfg.seed);
        }
    }

    return 0;
}

int main(int argc, char **argv) {
    int study_mode = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--study") == 0) {
            study_mode = 1;
            break;
        }
    }

    if (study_mode) {
        return run_study_mode(argc, argv);
    }

    if (argc < 3) usage();

    // 1. Parse Args
    int N = atoi(argv[1]);
    int trials = atoi(argv[2]);
    const char *prefix = (argc > 3) ? argv[3] : "run";
    uint64_t seed = (argc > 4) ? (uint64_t)strtoull(argv[4], NULL, 10) : 12345ULL;
    uint64_t run_id = (argc > 5) ? (uint64_t)strtoull(argv[5], NULL, 10) : 0ULL;

    // 2. Setup Signal Handling
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = handle_sigterm;
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);

    // 3. Configuration (Solver Params)
    PhaseParams pp = {
        .iters = 100000,
        .T_start = 1.0, .T_end = 1e-5,
        .step_xy_start = 0.05, .step_th_start = 0.5,
        .adapt_window = 2000, .acc_low = 0.4, .acc_high = 0.6,
        .step_shrink = 0.95, .step_grow = 1.05,
        .step_xy_min = 1e-5, .step_xy_max = 2.0,
        .step_th_min = 1e-4, .step_th_max = M_PI,
        .lambda_start = 1.0, .lambda_max = 1e6, 
        .mu_start = 1.0, .mu_max = 1e6,
        .ramp_every = 5000, .ramp_factor = 2.0,
        .log_every = 10000
    };

    // 4. Initialize State
    State s;
    s.N = N;
    s.cx = malloc(N * sizeof(double));
    s.cy = malloc(N * sizeof(double));
    s.th = malloc(N * sizeof(double));
    double *best_cx = malloc(N * sizeof(double));
    double *best_cy = malloc(N * sizeof(double));
    double *best_th = malloc(N * sizeof(double));
    s.world = malloc((size_t)N * NV * sizeof(Vec2));
    s.aabb = malloc(N * sizeof(AABB));
    s.tri_aabb = malloc((size_t)N * NTRI * sizeof(AABB));
    if (!s.cx || !s.cy || !s.th || !best_cx || !best_cy || !best_th || !s.world || !s.aabb || !s.tri_aabb) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }
    
    // Geometry Init
    s.br = base_bounding_radius();
    double cell = s.br * 2.0;

    RNG rng;
    uint64_t mix = seed ^ (run_id * 0x9e3779b97f4a7c15ULL);
    rng_seed(&rng, mix);

    // 5. Bisection Loop (High Level Control)
    double area = N * base_polygon_area();
    double L_min = sqrt(area);
    double L_max = sqrt(area) * 3.0;
    double best_L = L_max;
    double best_feas = INFINITY;
    int have_best = 0;

    printf("Starting Solver for N=%d. Est Min L=%.4f\n", N, L_min);
    ensure_dir("csv");
    ensure_dir("img");
    logger_init(prefix);

    const double FEAS_EPS = 1e-6;

    for (int t = 0; t < trials; t++) {
        if (g_stop_requested) break;

        double L = (L_min + L_max) / 2.0;
        s.L = L;
        grid_init(&s.grid, N, L, cell);

        // Randomize simple initial placement
        for (int i = 0; i < N; ++i) {
            s.cx[i] = rng_uniform(&rng, -L*0.5, L*0.5);
            s.cy[i] = rng_uniform(&rng, -L*0.5, L*0.5);
            s.th[i] = wrap_angle_0_2pi(rng_uniform(&rng, 0.0, 2.0 * M_PI));
            build_world_verts(&s.world[i * NV], s.cx[i], s.cy[i], s.th[i]);
        }

        printf("Trial %d/%d: Testing L=%.5f... ", t+1, trials, L);
        fflush(stdout);

        // Run the Solver (Annealing)
        PhaseParams A = pp;
        PhaseParams B = pp;
        double feas = try_pack_at_current_L(&s, &rng, &A, &B, 1, seed, run_id, s.cx, s.cy, s.th, 0);
        int success = (isfinite(feas) && feas <= FEAS_EPS) ? 1 : 0;

        if (feas < best_feas) {
            best_feas = feas;
            best_L = L;
            have_best = 1;
            for (int i = 0; i < N; i++) {
                best_cx[i] = s.cx[i];
                best_cy[i] = s.cy[i];
                best_th[i] = s.th[i];
            }
        }

        if (success) {
            printf("SUCCESS.\n");
            L_max = L; // Try tighter

            // Save Snapshot
            char path_svg[256], path_csv[256], path_svg_ckpt[256], path_csv_ckpt[256];
            snprintf(path_svg, sizeof(path_svg), "img/%s_best_N%03d.svg", prefix, N);
            snprintf(path_csv, sizeof(path_csv), "csv/%s_best_polys_N%03d.csv", prefix, N);
            snprintf(path_svg_ckpt, sizeof(path_svg_ckpt), "img/%s_checkpoint_N%03d.svg", prefix, N);
            snprintf(path_csv_ckpt, sizeof(path_csv_ckpt), "csv/%s_checkpoint_N%03d.csv", prefix, N);
            write_svg(path_svg, &s, feas);
            write_csv(path_csv, prefix, run_id, seed, &s, feas);
            write_svg(path_svg_ckpt, &s, feas);
            write_csv(path_csv_ckpt, prefix, run_id, seed, &s, feas);
            logger_log_trial(t, &s, feas);

        } else {
            printf("FAIL.\n");
            L_min = L; // Needs more space
        }
        
        grid_free(&s.grid);
    }

    if (have_best) {
        for (int i = 0; i < N; i++) {
            s.cx[i] = best_cx[i];
            s.cy[i] = best_cy[i];
            s.th[i] = best_th[i];
        }
        s.L = best_L;
        for (int i = 0; i < N; i++) update_instance(&s, i);
        grid_rebuild(&s.grid, s.N, best_L, cell, s.cx, s.cy);

        char path_svg[256], path_csv[256];
        snprintf(path_svg, sizeof(path_svg), "img/%s_best_N%03d.svg", prefix, N);
        snprintf(path_csv, sizeof(path_csv), "csv/%s_best_polys_N%03d.csv", prefix, N);
        write_svg(path_svg, &s, best_feas);
        write_csv(path_csv, prefix, run_id, seed, &s, best_feas);
        grid_free(&s.grid);
    }

    printf("Final Best L: %.6f (Density: %.4f)\n", best_L, area / (best_L*best_L));

    // Cleanup
    free(s.cx); free(s.cy); free(s.th);
    free(best_cx); free(best_cy); free(best_th);
    free(s.world); free(s.aabb); free(s.tri_aabb);
    logger_close();

    return 0;
}
