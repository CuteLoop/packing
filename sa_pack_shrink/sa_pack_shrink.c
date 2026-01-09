// sa_pack_shrink.c
// Simulated annealing for circle packing in a square, with high-ROI upgrades:
// 1) Incremental energy updates: O(N) per move (pairs involving moved circle only)
// 2) Adaptive step size: windowed acceptance-rate controller
// 3) Two-phase schedule: Explore -> Enforce feasibility (with penalty ramp)
// 4) Multi-start: many medium trials beats one long run
// 5) Reinsert move (teleport): cheap global escape move
// 6) OUTER LOOP: shrink-the-square (bracket + binary search on L) to approximate minimal feasible L
//
// Outputs:
//   - best_circles.csv
//   - best.svg
//
// Compile:
//   gcc -O2 -std=c11 -Wall -Wextra -pedantic sa_pack_shrink.c -o sa_pack_shrink -lm
//
// Run:
//   ./sa_pack_shrink

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
    double L;      // container side length
    double *x;     // length N
    double *y;     // length N
    double *r;     // length N
} State;

typedef struct {
    double alpha_L;     // weight on L (we do NOT optimize L inside SA; outer loop handles it)
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

        fprintf(f, "  <circle cx=\"%.6f\" cy=\"%.6f\" r=\"%.6f\" fill=\"#999999\" fill-opacity=\"0.18\" stroke=\"#000\" stroke-width=\"1\"/>\n",
                cx, cy, rr);
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

// Warm-start helper when L changes: scale positions inward.
static void scale_positions_for_new_L(State *s, double oldL, double newL, double safety) {
    if (oldL <= 0 || newL <= 0) return;
    double gamma = (newL / oldL) * safety;
    for (int i = 0; i < s->N; i++) {
        s->x[i] *= gamma;
        s->y[i] *= gamma;
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

    // NOTE: overlap_total is sum_{i<j}. The sum over (k,j) for j!=k corresponds
    // exactly to "pairs involving k" counted once each, so this delta is correct.
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
                      double *best_feas_io, double *bestx, double *besty,
                      int verbose)
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

        if (verbose && (t % log_every) == 0) {
            double acc_rate = (double)accepts_total / (double)(t + 1);
            printf("    iter=%d/%d T=%.2e step=%.2e E=%.3e ov=%.2e out=%.2e feas=%.2e acc=%.3f lam=%.2e mu=%.2e\n",
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
                        double *bestx, double *besty,
                        int verbose)
{
    // init totals
    Totals tot = compute_totals_full(s);

    Weights w;
    w.alpha_L = 0.0; // L fixed (outer loop handles L)

    // initialize best-by-feas with current state
    double best_feas = feasibility_metric(&tot);
    for (int i = 0; i < s->N; i++) { bestx[i] = s->x[i]; besty[i] = s->y[i]; }

    if (verbose) {
        printf("    start: ov=%.2e out=%.2e feas=%.2e\n", tot.overlap_total, tot.out_total, best_feas);
        printf("    Phase A (explore)\n");
    }
    run_phase(s, &tot, &w, rng, A, &best_feas, bestx, besty, verbose);

    if (verbose) {
        printf("    Phase B (enforce)\n");
    }
    run_phase(s, &tot, &w, rng, B, &best_feas, bestx, besty, verbose);

    // restore best-by-feas into state
    for (int i = 0; i < s->N; i++) { s->x[i] = bestx[i]; s->y[i] = besty[i]; }

    Totals best_tot = compute_totals_full(s);
    double best_feas_check = feasibility_metric(&best_tot);

    // numerical drift guard
    if (best_feas_check < best_feas) best_feas = best_feas_check;

    if (verbose) {
        printf("    trial best: ov=%.2e out=%.2e feas=%.2e\n", best_tot.overlap_total, best_tot.out_total, best_feas);
    }
    return best_feas;
}

// Try to find a feasible configuration for current s->L via multi-start SA.
// Returns best feasibility achieved, and stores best config in out_best_x/out_best_y.
static double try_pack_at_current_L(State *s, RNG *rng,
                                   const PhaseParams *A, const PhaseParams *B,
                                   int trials,
                                   double *out_best_x, double *out_best_y,
                                   int verbose_trials)
{
    int N = s->N;

    double *trial_best_x = (double*)malloc((size_t)N * sizeof(double));
    double *trial_best_y = (double*)malloc((size_t)N * sizeof(double));
    if (!trial_best_x || !trial_best_y) {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }

    double best_feas = 1e300;

    for (int tr = 0; tr < trials; tr++) {
        // diversify seeds (but stable enough if you rerun quickly)
        uint64_t seed = (uint64_t)time(NULL) ^ (uint64_t)(0x9E3779B97F4A7C15ULL * (uint64_t)(tr + 1));
        rng_seed(rng, seed);

        // For trial 0, keep current positions (warm-start). For others, randomize.
        if (tr > 0) random_init(s, rng);

        if (verbose_trials) {
            printf("  - SA trial %d/%d (seed=%llu)\n", tr + 1, trials, (unsigned long long)seed);
        }

        double feas = run_trial(s, rng, A, B, trial_best_x, trial_best_y, verbose_trials);

        if (feas < best_feas) {
            best_feas = feas;
            for (int i = 0; i < N; i++) { out_best_x[i] = trial_best_x[i]; out_best_y[i] = trial_best_y[i]; }
        }

        // restore best-so-far into s for warm-starting next trial 0-ish effect
        for (int i = 0; i < N; i++) { s->x[i] = out_best_x[i]; s->y[i] = out_best_y[i]; }
    }

    free(trial_best_x);
    free(trial_best_y);
    return best_feas;
}

// ---------------- Outer loop: bracket + binary search on L ----------------

static int is_feasible(double feas, double tol) {
    return (feas <= tol);
}

int main(void) {
    RNG rng;
    rng_seed(&rng, (uint64_t)time(NULL));

    int N = 200;
    State s = state_alloc(N);

    // Problem instance: equal radii
    s.L = 10.0;
    for (int i = 0; i < N; i++) s.r[i] = 0.45;

    // Two-phase schedule (starter settings)
    PhaseParams A; // Explore
    A.iters = 90000;
    A.T_start = 0.30;
    A.T_end   = 3e-4;
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
    A.ramp_every   = 0;
    A.ramp_factor  = 1.0;
    A.lambda_max   = 1e7;
    A.mu_max       = 1e7;

    A.p_reinsert = 0.02;
    A.log_every  = A.iters / 6;

    PhaseParams B; // Enforce feasibility
    B.iters = 45000;
    B.T_start = 0.05;
    B.T_end   = 1e-6;
    B.step_start = 0.4;

    B.adapt_window = 1500;
    B.acc_low  = 0.10;
    B.acc_high = 0.30;
    B.step_shrink = 0.90;
    B.step_grow   = 1.08;
    B.step_min = 1e-5;
    B.step_max = 1.0;

    B.lambda_start = 1e3;
    B.mu_start     = 1e3;
    B.ramp_every   = 1500;
    B.ramp_factor  = 1.8;
    B.lambda_max   = 1e7;
    B.mu_max       = 1e7;

    B.p_reinsert = 0.005;
    B.log_every  = B.iters / 6;

    // Outer loop controls
    const double feas_tol = 1e-10;     // feasibility threshold
    const int bracket_max_steps = 40;
    const int bisect_steps = 30;
    const double shrink_factor = 0.97;
    const double grow_factor   = 1.05;
    const double warm_safety   = 0.98;

    // SA effort budgets
    const int trials_bracket = 18;     // spend more effort to get a reliable bracket
    const int trials_bisect  = 12;     // moderate effort per bisection step

    // Working buffers for best configs
    double *bestx = (double*)malloc((size_t)N * sizeof(double));
    double *besty = (double*)malloc((size_t)N * sizeof(double));
    double *bestx2 = (double*)malloc((size_t)N * sizeof(double));
    double *besty2 = (double*)malloc((size_t)N * sizeof(double));
    if (!bestx || !besty || !bestx2 || !besty2) {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }

    // Initial seed config
    random_init(&s, &rng);

    // Step 1: determine if initial L is feasible
    printf("=== INITIAL PACK at L=%.6g ===\n", s.L);
    double feas0 = try_pack_at_current_L(&s, &rng, &A, &B, trials_bracket, bestx, besty, 0);
    for (int i = 0; i < N; i++) { s.x[i] = bestx[i]; s.y[i] = besty[i]; }

    printf("Initial result: L=%.6g feas=%.3e (%s)\n",
           s.L, feas0, is_feasible(feas0, feas_tol) ? "FEASIBLE" : "INFEASIBLE");

    // Step 2: bracket [L_low (infeasible), L_high (feasible)]
    double L_low, L_high;
    double L_curr = s.L;

    double best_feas_high = 1e300;
    double *bestx_high = (double*)malloc((size_t)N * sizeof(double));
    double *besty_high = (double*)malloc((size_t)N * sizeof(double));
    if (!bestx_high || !besty_high) { fprintf(stderr, "alloc failed\n"); exit(1); }

    if (is_feasible(feas0, feas_tol)) {
        // We have a feasible point; try shrinking until infeasible
        L_high = L_curr;
        best_feas_high = feas0;
        for (int i = 0; i < N; i++) { bestx_high[i] = s.x[i]; besty_high[i] = s.y[i]; }

        int found_infeas = 0;
        for (int k = 0; k < bracket_max_steps; k++) {
            double L_new = L_curr * shrink_factor;
            printf("\n=== BRACKET SHRINK: L=%.6g -> %.6g ===\n", L_curr, L_new);

            // warm-start: scale current best inward
            scale_positions_for_new_L(&s, L_curr, L_new, warm_safety);
            s.L = L_new;

            double feas = try_pack_at_current_L(&s, &rng, &A, &B, trials_bracket, bestx, besty, 0);
            for (int i = 0; i < N; i++) { s.x[i] = bestx[i]; s.y[i] = besty[i]; }

            printf("Bracket check: L=%.6g feas=%.3e (%s)\n",
                   s.L, feas, is_feasible(feas, feas_tol) ? "FEASIBLE" : "INFEASIBLE");

            if (is_feasible(feas, feas_tol)) {
                // move feasible down
                L_high = s.L;
                best_feas_high = feas;
                for (int i = 0; i < N; i++) { bestx_high[i] = s.x[i]; besty_high[i] = s.y[i]; }
                L_curr = s.L;
            } else {
                // found infeasible below
                L_low = s.L;
                found_infeas = 1;
                break;
            }
        }
        if (!found_infeas) {
            // never became infeasible (means L is still large); set a conservative low
            L_low = L_high * 0.5;
            printf("\nWARNING: did not find infeasible via shrinking; using L_low=%.6g\n", L_low);
        }
    } else {
        // Infeasible; grow until feasible
        L_low = L_curr;

        int found_feas = 0;
        for (int k = 0; k < bracket_max_steps; k++) {
            double L_new = L_curr * grow_factor;
            printf("\n=== BRACKET GROW: L=%.6g -> %.6g ===\n", L_curr, L_new);

            // warm-start: scale outward slightly (still use safety to avoid boundary slam)
            scale_positions_for_new_L(&s, L_curr, L_new, warm_safety);
            s.L = L_new;

            double feas = try_pack_at_current_L(&s, &rng, &A, &B, trials_bracket, bestx, besty, 0);
            for (int i = 0; i < N; i++) { s.x[i] = bestx[i]; s.y[i] = besty[i]; }

            printf("Bracket check: L=%.6g feas=%.3e (%s)\n",
                   s.L, feas, is_feasible(feas, feas_tol) ? "FEASIBLE" : "INFEASIBLE");

            if (is_feasible(feas, feas_tol)) {
                L_high = s.L;
                best_feas_high = feas;
                for (int i = 0; i < N; i++) { bestx_high[i] = s.x[i]; besty_high[i] = s.y[i]; }
                found_feas = 1;
                break;
            } else {
                L_low = s.L;
                L_curr = s.L;
            }
        }
        if (!found_feas) {
            fprintf(stderr, "ERROR: could not find feasible L by growing.\n");
            state_free(&s);
            return 1;
        }
    }

    printf("\n=== BRACKET FOUND ===\n");
    printf("L_low  (infeasible) ~ %.10g\n", L_low);
    printf("L_high (feasible)   ~ %.10g  (feas=%.3e)\n", L_high, best_feas_high);

    // Step 3: binary search on L
    double L_prev_feas = L_high;

    // Ensure state contains best known feasible configuration at L_high
    s.L = L_high;
    for (int i = 0; i < N; i++) { s.x[i] = bestx_high[i]; s.y[i] = besty_high[i]; }

    for (int it = 0; it < bisect_steps; it++) {
        double L_mid = 0.5 * (L_low + L_high);
        printf("\n=== BISECT %d/%d: [%.10g, %.10g] mid=%.10g ===\n", it + 1, bisect_steps, L_low, L_high, L_mid);

        // warm-start from last feasible
        scale_positions_for_new_L(&s, L_prev_feas, L_mid, warm_safety);
        s.L = L_mid;

        // run SA at this L
        double feas = try_pack_at_current_L(&s, &rng, &A, &B, trials_bisect, bestx2, besty2, 0);
        for (int i = 0; i < N; i++) { s.x[i] = bestx2[i]; s.y[i] = besty2[i]; }

        printf("Mid result: L=%.10g feas=%.3e (%s)\n",
               s.L, feas, is_feasible(feas, feas_tol) ? "FEASIBLE" : "INFEASIBLE");

        if (is_feasible(feas, feas_tol)) {
            // feasible: tighten upper bound and record best feasible config
            L_high = L_mid;
            L_prev_feas = L_mid;
            best_feas_high = feas;
            for (int i = 0; i < N; i++) { bestx_high[i] = s.x[i]; besty_high[i] = s.y[i]; }
        } else {
            // infeasible: raise lower bound
            L_low = L_mid;

            // restore last known feasible config into s for next iteration warm-start
            s.L = L_high;
            for (int i = 0; i < N; i++) { s.x[i] = bestx_high[i]; s.y[i] = besty_high[i]; }
        }
    }

    // Final: restore best feasible at smallest found L_high
    s.L = L_high;
    for (int i = 0; i < N; i++) { s.x[i] = bestx_high[i]; s.y[i] = besty_high[i]; }

    Totals final_tot = compute_totals_full(&s);
    double final_feas = feasibility_metric(&final_tot);

    printf("\n=== FINAL (BEST FEASIBLE) ===\n");
    printf("L*=%.12g\n", s.L);
    printf("ov=%.6e out=%.6e feas=%.6e (tol=%.1e)\n",
           final_tot.overlap_total, final_tot.out_total, final_feas, feas_tol);

    if (!write_circles_csv("best_circles.csv", &s, final_feas)) {
        fprintf(stderr, "Failed to write best_circles.csv\n");
    } else {
        printf("Wrote best configuration to best_circles.csv\n");
    }

    if (!write_best_svg("best.svg", &s, final_feas, 900, 900, 40.0)) {
        fprintf(stderr, "Failed to write best.svg\n");
    } else {
        printf("Wrote visualization to best.svg\n");
    }

    // print final centers
    for (int i = 0; i < N; i++) {
        printf("i=%d x=%.8f y=%.8f r=%.5f\n", i, s.x[i], s.y[i], s.r[i]);
    }

    free(bestx);
    free(besty);
    free(bestx2);
    free(besty2);
    free(bestx_high);
    free(besty_high);

    state_free(&s);
    return 0;
}
