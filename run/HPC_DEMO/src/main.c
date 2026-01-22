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
void write_csv(const char *path, State *s, double feas) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "i,cx,cy,th,L,feas\n");
    for(int i=0; i<s->N; i++) {
        fprintf(f, "%d,%.10f,%.10f,%.10f,%.10f,%.10e\n", 
            i, s->cx[i], s->cy[i], s->th[i], s->L, feas);
    }
    fclose(f);
}

void usage() {
    fprintf(stderr, "Usage: ./solver [N] [trials] [out_prefix]\n");
    exit(1);
}

int main(int argc, char **argv) {
    if (argc < 3) usage();

    // 1. Parse Args
    int N = atoi(argv[1]);
    int trials = atoi(argv[2]);
    const char *prefix = (argc > 3) ? argv[3] : "run";

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

    Weights w = { .alpha_L = 0.0, .lambda_ov = 1.0, .mu_out = 1.0 };

    // 4. Initialize State
    State s;
    s.N = N;
    s.cx = malloc(N * sizeof(double));
    s.cy = malloc(N * sizeof(double));
    s.th = malloc(N * sizeof(double));
    s.world = malloc((size_t)N * NV * sizeof(Vec2));
    s.aabb = malloc(N * sizeof(AABB));
    s.tri_aabb = malloc((size_t)N * NTRI * sizeof(AABB));
    
    // Geometry Init
    s.br = base_bounding_radius();
    double cell = s.br * 2.0;

    RNG rng;
    rng_seed(&rng, 12345ULL + (uint64_t)getpid());

    // 5. Bisection Loop (High Level Control)
    double area = N * base_polygon_area();
    double L_min = sqrt(area);
    double L_max = sqrt(area) * 3.0;
    double best_L = L_max;

    printf("Starting Solver for N=%d. Est Min L=%.4f\n", N, L_min);
    ensure_dir("csv");
    ensure_dir("img");
    logger_init(prefix);

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
        double feas = try_pack_at_current_L(&s, &rng, &A, &B, 1, 12345ULL, 0ULL, s.cx, s.cy, s.th, 0);
        int success = (isfinite(feas) && feas < 1e200) ? 1 : 0;

        if (success) {
            printf("SUCCESS.\n");
            best_L = L;
            L_max = L; // Try tighter

            // Save Snapshot
            char path_svg[256], path_csv[256];
            snprintf(path_svg, 256, "img/%s_N%d_L%.4f.svg", prefix, N, L);
            snprintf(path_csv, 256, "csv/%s_N%d_L%.4f.csv", prefix, N, L);
            write_svg(path_svg, &s, 0.0);
            write_csv(path_csv, &s, 0.0);
            char snap[256];
            snprintf(snap, sizeof(snap), "csv/%s_N%d_L%.4f.snapshot.csv", prefix, N, L);
            logger_write_snapshot(snap, &s, 0.0);
            logger_log_trial(t, &s, 0.0);

        } else {
            printf("FAIL.\n");
            L_min = L; // Needs more space
        }
        
        grid_free(&s.grid);
    }

    printf("Final Best L: %.6f (Density: %.4f)\n", best_L, area / (best_L*best_L));

    // Cleanup
    free(s.cx); free(s.cy); free(s.th);
    free(s.world); free(s.aabb); free(s.tri_aabb);
    logger_close();

    return 0;
}
