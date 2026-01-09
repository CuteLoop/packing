// sa_pack.c
// Simulated annealing for circle packing in a square, with high-ROI upgrades:
// 1) Incremental energy updates: O(N) per move (pairs involving moved circle only)
// 2) Adaptive step size: windowed acceptance-rate controller
// 3) Two-phase schedule: Explore -> Enforce feasibility (with penalty ramp)
// 4) Multi-start: many medium trials beats one long run
// 5) Reinsert move (teleport): cheap global escape move
//
// Outputs:
//   - best_circles.csv
//   - best.svg
//
// Compile:
//   gcc -O2 -std=c11 -Wall -Wextra -pedantic sa_pack.c -o sa_pack -lm
//
// Run:
//   ./sa_pack

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------- RNG (xorshift64*) ----------------

typedef struct { uint64_t s; } RNG;

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void rng_seed(RNG *rng, uint64_t seed) {
    uint64_t x = seed;
    rng->s = splitmix64(&x);
    if (rng->s == 0) rng->s = 0xdeadbeefULL;
}

static uint64_t xorshift64star(RNG *rng) {
    uint64_t x = rng->s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static double rng_u01(RNG *rng) {
    // Uniform in [0,1), using 53 random bits (never returns 1.0)
    return (double)(xorshift64star(rng) >> 11) * (1.0 / 9007199254740992.0); // 2^53
}

static double rng_uniform(RNG *rng, double a, double b) {
    return a + (b - a) * rng_u01(rng);
}

// ---------------- Problem: circles in square ----------------

typedef struct {
    int N;
    double L;      // container side length (fixed in this demo)
    double *x;     // length N
    double *y;     // length N
    double *r;     // length N
} State;

typedef struct {
    double alpha_L;     // weight on L (if optimizing L; else 0)
    double lambda_ov;   // overlap penalty weight
    double mu_out;      // outside penalty weight
} Weights;

// Running totals for incremental updates
typedef struct {
    double overlap_total; // sum_{i<j} overlap_depth(i,j)^2
    double out_total;     // sum_i outside_amount(i)^2
} Totals;

// ---------------- Export: CSV + SVG ----------------

static int write_circles_csv(const char *path, const State *s, double best_feas) {
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    fprintf(f, "# L=%.17g best_feas=%.17g N=%d\n", s->L, best_feas, s->N);
    fprintf(f, "i,x,y,r\n");
    for (int i = 0; i < s->N; i++) {
        fprintf(f, "%d,%.17g,%.17g,%.17g\n", i, s->x[i], s->y[i], s->r[i]);
    }
    fclose(f);
    return 1;
}

// Minimal SVG renderer (square + circles). Coordinate mapping:
// world (x,y) in [-L/2, L/2] -> SVG pixels with y flipped.
static int write_best_svg(const char *path, const State *s, double best_feas,
                          int width_px, int height_px, double margin_px)
{
    FILE *f = fopen(path, "w");
    if (!f) return 0;

    const double L = s->L;
    const double half = 0.5 * L;
    const double W = (double)width_px;
    const double H = (double)height_px;

    const double sx = (W - 2.0 * margin_px) / L;
    const double sy = (H - 2.0 * margin_px) / L;
    const double scale = (sx < sy) ? sx : sy;

    const double square_px = L * scale;
    const double ox = margin_px + 0.5 * (W - 2.0 * margin_px - square_px);
    const double oy = margin_px + 0.5 * (H - 2.0 * margin_px - square_px);

    fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    fprintf(f, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">\n",
            width_px, height_px, width_px, height_px);

    fprintf(f, "  <rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" fill=\"white\"/>\n",
            width_px, height_px);

    fprintf(f, "  <text x=\"%.1f\" y=\"%.1f\" font-family=\"monospace\" font-size=\"12\">L=%.6g best_feas=%.6g N=%d</text>\n",
            10.0, 18.0, s->L, best_feas, s->N);

    fprintf(f, "  <rect x=\"%.6f\" y=\"%.6f\" width=\"%.6f\" height=\"%.6f\" fill=\"none\" stroke=\"#000\" stroke-width=\"2\"/>\n",
            ox, oy, square_px, square_px);

    for (int i = 0; i < s->N; i++) {
        double cx = ox + (s->x[i] + half) * scale;
        double cy = oy + (half - s->y[i]) * scale;
        double rr = s->r[i] * scale;

        const char *fill = (i % 2 == 0) ? "#777777" : "#999999";
        fprintf(f, "  <circle cx=\"%.6f\" cy=\"%.6f\" r=\"%.6f\" fill=\"%s\" fill-opacity=\"0.20\" stroke=\"#000\" stroke-width=\"1\"/>\n",
                cx, cy, rr, fill);
    }

    fprintf(f, "</svg>\n");
    fclose(f);
    return 1;
}

// ---------------- Utilities ----------------

static State state_alloc(int N) {
    State s;
    s.N = N;
    s.L = 1.0;
    s.x = (double*)calloc((size_t)N, sizeof(double));
    s.y = (double*)calloc((size_t)N, sizeof(double));
    s.r = (double*)calloc((size_t)N, sizeof(double));
    if (!s.x || !s.y || !s.r) {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }
    return s;
}

static void state_free(State *s) {
    free(s->x); free(s->y); free(s->r);
    s->x = s->y = s->r = NULL;
}

static double overlap_pair(const State *s, int i, int j) {
    double dx = s->x[i] - s->x[j];
    double dy = s->y[i] - s->y[j];
    double d2 = dx*dx + dy*dy;
    double need = s->r[i] + s->r[j];
    double need2 = need * need;
    if (d2 >= need2) return 0.0;
    double d = sqrt(d2);
    double ov = (need - d);
    return ov * ov;
}

// sum of overlap_pair(k,j) for all j!=k (this corresponds to “all pairs involving k”, counted once each)
static double overlap_sum_for_k(const State *s, int k) {
    double sum = 0.0;
    for (int j = 0; j < s->N; j++) {
        if (j == k) continue;
        sum += overlap_pair(s, k, j);
    }
    return sum;
}

static double outside_for_k(const State *s, int k) {
    double half = 0.5 * s->L;
    double lim = half - s->r[k];
    double pen = 0.0;

    double ax = fabs(s->x[k]) - lim;
    if (ax > 0) pen += ax * ax;

    double ay = fabs(s->y[k]) - lim;
    if (ay > 0) pen += ay * ay;

    return pen;
}

static Totals compute_totals_full(const State *s) {
    Totals t;
    t.overlap_total = 0.0;
    t.out_total = 0.0;

    for (int i = 0; i < s->N; i++) {
        t.out_total += outside_for_k(s, i);
    }
    for (int i = 0; i < s->N; i++) {
        for (int j = i+1; j < s->N; j++) {
            t.overlap_total += overlap_pair(s, i, j);
        }
    }
    return t;
}

static double energy_from_totals(const State *s, const Weights *w, const Totals *t) {
    return w->alpha_L * s->L + w->lambda_ov * t->overlap_total + w->mu_out * t->out_total;
}

static double feasibility_metric(const Totals *t) {
    // Unweighted feasibility: drives overlaps and boundary violations to zero.
    return t->overlap_total + t->out_total;
}

static void random_init(State *s, RNG *rng) {
    double half = 0.5 * s->L;
    for (int i = 0; i < s->N; i++) {
        double lim = half - s->r[i];
        if (lim < 0) lim = 0;
        s->x[i] = rng_uniform(rng, -lim, lim);
        s->y[i] = rng_uniform(rng, -lim, lim);
    }
}

// ---------------- Incremental move bookkeeping ----------------

typedef struct {
    int idx;
    double oldx, oldy;
    double d_overlap; // delta overlap_total if accepted
    double d_out;     // delta out_total if accepted
} Move;

// Local jiggle or teleport (reinsert)
static Move propose_move(State *s, Totals *tot, RNG *rng,
                         double step_xy, double p_reinsert)
{
    Move m;
    int k = (int)(rng_u01(rng) * (double)s->N);
    if (k < 0) k = 0;
    if (k >= s->N) k = s->N - 1;

    m.idx = k;
    m.oldx = s->x[k];
    m.oldy = s->y[k];

    // old local contributions
    double old_ov = overlap_sum_for_k(s, k);
    double old_out = outside_for_k(s, k);

    // apply either reinsert or local move
    double u = rng_u01(rng);
    if (u < p_reinsert) {
        double half = 0.5 * s->L;
        double lim = half - s->r[k];
        if (lim < 0) lim = 0;
        s->x[k] = rng_uniform(rng, -lim, lim);
        s->y[k] = rng_uniform(rng, -lim, lim);
    } else {
        double dx = rng_uniform(rng, -step_xy, step_xy);
        double dy = rng_uniform(rng, -step_xy, step_xy);
        s->x[k] += dx;
        s->y[k] += dy;
    }

    // new local contributions
    double new_ov = overlap_sum_for_k(s, k);
    double new_out = outside_for_k(s, k);

    m.d_overlap = (new_ov - old_ov);
    m.d_out     = (new_out - old_out);

    tot->overlap_total += m.d_overlap;
    tot->out_total     += m.d_out;

    return m;
}

static void undo_move(State *s, Totals *tot, const Move *m) {
    tot->overlap_total -= m->d_overlap;
    tot->out_total     -= m->d_out;
    s->x[m->idx] = m->oldx;
    s->y[m->idx] = m->oldy;
}

// ---------------- SA params (per phase) ----------------

typedef struct {
    int iters;

    double T_start;
    double T_end;          // define cooling from start->end over iters
    double step_start;

    // acceptance-based step adaptation
    int adapt_window;
    double acc_low;
    double acc_high;
    double step_shrink;
    double step_grow;
    double step_min;
    double step_max;

    // penalty schedule
    double lambda_start;
    double mu_start;
    int ramp_every;
    double ramp_factor;
    double lambda_max;
    double mu_max;

    // move mix
    double p_reinsert;

    int log_every;
} PhaseParams;

static double cooling_from_range(double T_start, double T_end, int iters) {
    if (iters <= 0) return 1.0;
    if (T_start <= 0) T_start = 1e-12;
    if (T_end   <= 0) T_end   = 1e-12;
    return exp(log(T_end / T_start) / (double)iters);
}

// Run one phase; updates (s, tot, w) in place.
// Tracks and updates best-by-feas (positions + feas value).
static void run_phase(State *s, Totals *tot, Weights *w, RNG *rng,
                      const PhaseParams *pp,
                      double *best_feas_io, double *bestx, double *besty)
{
    double T = pp->T_start;
    double cool = cooling_from_range(pp->T_start, pp->T_end, pp->iters);
    double step = pp->step_start;

    // initialize weights at phase start
    w->lambda_ov = pp->lambda_start;
    w->mu_out    = pp->mu_start;

    double E = energy_from_totals(s, w, tot);

    int accepts_total = 0;
    int accepts_win = 0, moves_win = 0;

    int log_every = pp->log_every;
    if (log_every < 1) log_every = 1;

    for (int t = 0; t < pp->iters; t++) {

        // ramp penalties
        if (pp->ramp_every > 0 && t > 0 && (t % pp->ramp_every) == 0) {
            w->lambda_ov *= pp->ramp_factor;
            w->mu_out    *= pp->ramp_factor;
            if (w->lambda_ov > pp->lambda_max) w->lambda_ov = pp->lambda_max;
            if (w->mu_out    > pp->mu_max)     w->mu_out    = pp->mu_max;
            E = energy_from_totals(s, w, tot);
        }

        Move m = propose_move(s, tot, rng, step, pp->p_reinsert);
        double Enew = energy_from_totals(s, w, tot);
        double dE = Enew - E;

        int accept = 0;
        if (dE <= 0.0) {
            accept = 1;
        } else {
            double u = rng_u01(rng);
            double pacc = exp(-dE / T);
            if (u < pacc) accept = 1;
        }

        if (accept) {
            E = Enew;
            accepts_total++;
            accepts_win++;

            // Track best by feasibility (unweighted), not by ramped E
            double feas = feasibility_metric(tot);
            if (feas < *best_feas_io) {
                *best_feas_io = feas;
                for (int i = 0; i < s->N; i++) { bestx[i] = s->x[i]; besty[i] = s->y[i]; }
            }
        } else {
            undo_move(s, tot, &m);
        }

        moves_win++;

        // temperature cool
        T *= cool;
        if (T < 1e-12) T = 1e-12;

        // adapt step by acceptance
        if (pp->adapt_window > 0 && moves_win >= pp->adapt_window) {
            double acc = (double)accepts_win / (double)moves_win;
            if (acc < pp->acc_low) step *= pp->step_shrink;
            else if (acc > pp->acc_high) step *= pp->step_grow;

            if (step < pp->step_min) step = pp->step_min;
            if (step > pp->step_max) step = pp->step_max;

            accepts_win = 0;
            moves_win = 0;
        }

        if ((t % log_every) == 0) {
            double acc_rate = (double)accepts_total / (double)(t + 1);
            printf("  iter=%d/%d T=%.2e step=%.2e E=%.3e ov=%.2e out=%.2e feas=%.2e acc=%.3f lam=%.2e mu=%.2e\n",
                   t, pp->iters, T, step, E, tot->overlap_total, tot->out_total,
                   feasibility_metric(tot), acc_rate, w->lambda_ov, w->mu_out);
        }
    }
}

// Run one full trial: Explore phase then Enforce phase.
// Returns best_feas for this trial and writes best positions into bestx/besty.
static double run_trial(State *s, RNG *rng,
                        const PhaseParams *A,
                        const PhaseParams *B,
                        double *bestx, double *besty)
{
    // init totals
    Totals tot = compute_totals_full(s);

    Weights w;
    w.alpha_L = 0.0; // L fixed

    // initialize best-by-feas with current state
    double best_feas = feasibility_metric(&tot);
    for (int i = 0; i < s->N; i++) { bestx[i] = s->x[i]; besty[i] = s->y[i]; }

    printf("  start: ov=%.2e out=%.2e feas=%.2e\n", tot.overlap_total, tot.out_total, best_feas);

    // Phase A: exploration
    printf("  Phase A (explore)\n");
    run_phase(s, &tot, &w, rng, A, &best_feas, bestx, besty);

    // Phase B: enforce feasibility (reheat a bit, ramp hard)
    printf("  Phase B (enforce)\n");
    run_phase(s, &tot, &w, rng, B, &best_feas, bestx, besty);

    // restore best-by-feas into state
    for (int i = 0; i < s->N; i++) { s->x[i] = bestx[i]; s->y[i] = besty[i]; }

    Totals best_tot = compute_totals_full(s);
    double best_feas_check = feasibility_metric(&best_tot);

    // numerical drift guard
    if (best_feas_check < best_feas) best_feas = best_feas_check;

    printf("  trial best: ov=%.2e out=%.2e feas=%.2e\n", best_tot.overlap_total, best_tot.out_total, best_feas);
    return best_feas;
}

// ---------------- Main demo ----------------

int main(void) {
    RNG rng;
    rng_seed(&rng, (uint64_t)time(NULL));

    int N = 25;
    State s = state_alloc(N);

    // Problem instance: equal radii
    s.L = 10.0;
    for (int i = 0; i < N; i++) s.r[i] = 0.45;

    // High-ROI defaults: many medium trials
    int trials = 50;

    // Two-phase schedule (recommended starter settings)
    PhaseParams A; // Explore
    A.iters = 105000;          // 70% of 150k
    A.T_start = 0.30;
    A.T_end   = 3e-4;          // ~1000x drop over phase
    A.step_start = 1.0;

    A.adapt_window = 1500;
    A.acc_low  = 0.25;
    A.acc_high = 0.55;
    A.step_shrink = 0.88;
    A.step_grow   = 1.12;
    A.step_min = 1e-4;
    A.step_max = 2.0;

    A.lambda_start = 1e2;
    A.mu_start     = 1e2;
    A.ramp_every   = 0;        // optional mild ramp in explore; keep off initially
    A.ramp_factor  = 1.0;
    A.lambda_max   = 1e6;
    A.mu_max       = 1e6;

    A.p_reinsert = 0.02;       // 2% teleports
    A.log_every  = A.iters / 6;

    PhaseParams B; // Enforce feasibility
    B.iters = 45000;           // 30% of 150k
    B.T_start = 0.05;          // reheat a bit
    B.T_end   = 1e-6;
    B.step_start = 0.4;

    B.adapt_window = 1500;
    B.acc_low  = 0.10;
    B.acc_high = 0.30;
    B.step_shrink = 0.90;
    B.step_grow   = 1.08;
    B.step_min = 1e-5;
    B.step_max = 1.0;

    B.lambda_start = 1e3;      // start higher
    B.mu_start     = 1e3;
    B.ramp_every   = 1500;
    B.ramp_factor  = 1.8;
    B.lambda_max   = 1e6;      // raise to 1e7 if needed
    B.mu_max       = 1e6;

    B.p_reinsert = 0.005;      // fewer teleports while tightening
    B.log_every  = B.iters / 6;

    // global best by feasibility
    double global_best_feas = 1e300;
    double *trial_best_x = (double*)malloc((size_t)N * sizeof(double));
    double *trial_best_y = (double*)malloc((size_t)N * sizeof(double));
    double *global_best_x = (double*)malloc((size_t)N * sizeof(double));
    double *global_best_y = (double*)malloc((size_t)N * sizeof(double));
    if (!trial_best_x || !trial_best_y || !global_best_x || !global_best_y) {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }

    for (int tr = 0; tr < trials; tr++) {
        // diversify seeds
        uint64_t seed = (uint64_t)time(NULL) ^ (uint64_t)(0x9E3779B97F4A7C15ULL * (uint64_t)(tr + 1));
        rng_seed(&rng, seed);

        random_init(&s, &rng);

        printf("\n=== TRIAL %d/%d (seed=%llu) ===\n", tr + 1, trials, (unsigned long long)seed);

        double best_feas = run_trial(&s, &rng, &A, &B, trial_best_x, trial_best_y);

        if (best_feas < global_best_feas) {
            global_best_feas = best_feas;
            for (int i = 0; i < N; i++) { global_best_x[i] = trial_best_x[i]; global_best_y[i] = trial_best_y[i]; }
            printf("  NEW GLOBAL BEST feas=%.2e\n", global_best_feas);
        }
    }

    // restore global best
    for (int i = 0; i < N; i++) { s.x[i] = global_best_x[i]; s.y[i] = global_best_y[i]; }

    free(trial_best_x);
    free(trial_best_y);
    free(global_best_x);
    free(global_best_y);

    Totals final_tot = compute_totals_full(&s);
    printf("\nGLOBAL BEST: ov=%.6e out=%.6e feas=%.6e\n",
           final_tot.overlap_total, final_tot.out_total, feasibility_metric(&final_tot));

    if (!write_circles_csv("best_circles.csv", &s, global_best_feas)) {
        fprintf(stderr, "Failed to write best_circles.csv\n");
    } else {
        printf("Wrote best configuration to best_circles.csv\n");
    }

    if (!write_best_svg("best.svg", &s, global_best_feas, 900, 900, 40.0)) {
        fprintf(stderr, "Failed to write best.svg\n");
    } else {
        printf("Wrote visualization to best.svg\n");
    }

    // print final centers
    for (int i = 0; i < N; i++) {
        printf("i=%d x=%.6f y=%.6f r=%.3f\n", i, s.x[i], s.y[i], s.r[i]);
    }

    state_free(&s);
    return 0;
}
