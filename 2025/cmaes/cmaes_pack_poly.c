// cmaes_pack_poly.c
// ------------------------------------------------------------
// CMA-ES demo for non-convex polygon packing in a square container.
// Decision vars: (cx_i, cy_i, theta_i) for i=0..N-1, D=3N.
// Objective: alpha*L + lambda_ov*overlap + mu_out*outside
//
// This version FIXES L (pass --L). Later you can wrap with bracket/bisect.
//
// Compile:
//   gcc -O3 -march=native -std=c11 -Wall -Wextra -pedantic cmaes_pack_poly.c -o cmaes_pack_poly -lm
//
// Run:
//   ./cmaes_pack_poly --N 7 --L 1.50 --evals 100000 --seed 1 --lambda 32
//
// Outputs:
//   csv/cma_best_N###.csv
//   img/cma_best_N###.svg
//
// ------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------- small utils ----------------

static int streq(const char *a, const char *b) { return strcmp(a, b) == 0; }

static void ensure_dir(const char *name) {
    if (mkdir(name, 0755) == 0) return;
    if (errno == EEXIST) return;
    fprintf(stderr, "ERROR: could not create dir '%s' (errno=%d)\n", name, errno);
    exit(1);
}

static void die(const char *msg) {
    fprintf(stderr, "ERROR: %s\n", msg);
    exit(1);
}

static double clamp(double x, double a, double b) {
    if (x < a) return a;
    if (x > b) return b;
    return x;
}

static double wrap_angle_0_2pi(double th) {
    double two = 2.0 * M_PI;
    th = fmod(th, two);
    if (th < 0.0) th += two;
    return th;
}

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
    return (double)(xorshift64star(rng) >> 11) * (1.0 / 9007199254740992.0); // 2^53
}

static double rng_normal(RNG *rng) {
    // Box-Muller
    double u1 = rng_u01(rng);
    double u2 = rng_u01(rng);
    if (u1 < 1e-16) u1 = 1e-16;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// ---------------- Geometry: base polygon + triangulation ----------------

typedef struct { double x, y; } Vec2;
typedef struct { double minx, miny, maxx, maxy; } AABB;
typedef struct { int a, b, c; } Tri;

static const int NV = 15;
static const Tri TRIS[] = {
    {0, 1, 2},
    {2, 3, 4},
    {0, 2, 4},
    {4, 5, 6},
    {0, 4, 6},
    {0, 6, 7},
    {0, 7, 8},
    {0, 8, 9},
    {9, 10, 11},
    {0, 9, 11},
    {11, 12, 13},
    {0, 11, 13},
    {0, 13, 14},
};
static const int NTRI = 13;

static const Vec2 BASE_V[15] = {
    {  0.0,     0.8  },
    {  0.125,   0.5  },
    {  0.0625,  0.5  },
    {  0.2,     0.25 },
    {  0.1,     0.25 },
    {  0.35,    0.0  },
    {  0.075,   0.0  },
    {  0.075,  -0.2  },
    { -0.075,  -0.2  },
    { -0.075,   0.0  },
    { -0.35,    0.0  },
    { -0.1,     0.25 },
    { -0.2,     0.25 },
    { -0.0625,  0.5  },
    { -0.125,   0.5  },
};

static double base_bounding_radius(void) {
    double rmax2 = 0.0;
    for (int i = 0; i < NV; i++) {
        double d2 = BASE_V[i].x * BASE_V[i].x + BASE_V[i].y * BASE_V[i].y;
        if (d2 > rmax2) rmax2 = d2;
    }
    return sqrt(rmax2);
}

static Vec2 rot_trans(Vec2 v, double c, double s, double tx, double ty) {
    Vec2 out;
    out.x = c * v.x - s * v.y + tx;
    out.y = s * v.x + c * v.y + ty;
    return out;
}

static void build_world_verts(Vec2 *world, double cx, double cy, double th) {
    double c = cos(th), s = sin(th);
    for (int i = 0; i < NV; i++) world[i] = rot_trans(BASE_V[i], c, s, cx, cy);
}

static AABB aabb_of_verts(const Vec2 *w) {
    AABB b;
    b.minx = b.maxx = w[0].x;
    b.miny = b.maxy = w[0].y;
    for (int i = 1; i < NV; i++) {
        if (w[i].x < b.minx) b.minx = w[i].x;
        if (w[i].x > b.maxx) b.maxx = w[i].x;
        if (w[i].y < b.miny) b.miny = w[i].y;
        if (w[i].y > b.maxy) b.maxy = w[i].y;
    }
    return b;
}

static AABB aabb_of_tri_pts(Vec2 p0, Vec2 p1, Vec2 p2) {
    AABB b;
    b.minx = b.maxx = p0.x;
    b.miny = b.maxy = p0.y;

    if (p1.x < b.minx) b.minx = p1.x;
    if (p1.x > b.maxx) b.maxx = p1.x;
    if (p1.y < b.miny) b.miny = p1.y;
    if (p1.y > b.maxy) b.maxy = p1.y;

    if (p2.x < b.minx) b.minx = p2.x;
    if (p2.x > b.maxx) b.maxx = p2.x;
    if (p2.y < b.miny) b.miny = p2.y;
    if (p2.y > b.maxy) b.maxy = p2.y;

    return b;
}

static int aabb_overlap(const AABB *a, const AABB *b) {
    if (a->maxx < b->minx || b->maxx < a->minx) return 0;
    if (a->maxy < b->miny || b->maxy < a->miny) return 0;
    return 1;
}

static double outside_penalty_aabb(const AABB *b, double L) {
    double half = 0.5 * L;
    double pen = 0.0;
    if (b->minx < -half) { double d = (-half - b->minx); pen += d * d; }
    if (b->maxx >  half) { double d = (b->maxx - half);  pen += d * d; }
    if (b->miny < -half) { double d = (-half - b->miny); pen += d * d; }
    if (b->maxy >  half) { double d = (b->maxy - half);  pen += d * d; }
    return pen;
}

// SAT helpers
static void proj3(const Vec2 p[3], double ax, double ay, double *mn, double *mx) {
    double v0 = p[0].x * ax + p[0].y * ay;
    double v1 = p[1].x * ax + p[1].y * ay;
    double v2 = p[2].x * ax + p[2].y * ay;
    double lo = v0, hi = v0;
    if (v1 < lo) lo = v1; if (v1 > hi) hi = v1;
    if (v2 < lo) lo = v2; if (v2 > hi) hi = v2;
    *mn = lo; *mx = hi;
}

static int tri_sat_penetration(const Vec2 A[3], const Vec2 B[3], double *depth_out) {
    double min_overlap = 1e300;
    for (int pass = 0; pass < 2; pass++) {
        const Vec2 *T = (pass == 0) ? A : B;
        for (int e = 0; e < 3; e++) {
            Vec2 p0 = T[e];
            Vec2 p1 = T[(e + 1) % 3];
            double ex = p1.x - p0.x;
            double ey = p1.y - p0.y;
            double ax = -ey, ay = ex;
            double len2 = ax*ax + ay*ay;
            if (len2 < 1e-30) continue;

            double amin, amax, bmin, bmax;
            proj3(A, ax, ay, &amin, &amax);
            proj3(B, ax, ay, &bmin, &bmax);
            double o = fmin(amax, bmax) - fmax(amin, bmin);
            if (o <= 0.0) return 0;
            if (o < min_overlap) min_overlap = o;
        }
    }
    *depth_out = (min_overlap > 1e200) ? 0.0 : min_overlap;
    return 1;
}

// ---------------- Fitness evaluation (no incremental; CMA needs many evals) ----------------
// For speed, we use broad-phase and a uniform grid, rebuilt each eval.

typedef struct {
    int N;
    double L;
    double br;
    double cell;
    int nx, ny;
    int *head, *next;
} EvalGrid;

static int grid_index(const EvalGrid *g, int ix, int iy) { return iy*g->nx + ix; }

static void grid_build(EvalGrid *g, const double *cx, const double *cy) {
    int nc = g->nx * g->ny;
    for (int i = 0; i < nc; i++) g->head[i] = -1;
    for (int i = 0; i < g->N; i++) g->next[i] = -1;

    double half = 0.5 * g->L;
    for (int i = 0; i < g->N; i++) {
        double fx = (cx[i] + half) / g->cell;
        double fy = (cy[i] + half) / g->cell;
        int ix = (int)floor(fx);
        int iy = (int)floor(fy);
        if (ix < 0) ix = 0; if (ix >= g->nx) ix = g->nx - 1;
        if (iy < 0) iy = 0; if (iy >= g->ny) iy = g->ny - 1;
        int c = grid_index(g, ix, iy);
        g->next[i] = g->head[c];
        g->head[c] = i;
    }
}

static double overlap_pair_penalty_one(const Vec2 *wi, const AABB *ai, const AABB *tri_ai,
                                      const Vec2 *wj, const AABB *aj, const AABB *tri_aj) {
    if (!aabb_overlap(ai, aj)) return 0.0;
    // triangle pair SAT with triangle-AABB reject
    double pen = 0.0;
    for (int ta = 0; ta < NTRI; ta++) {
        const AABB *aTa = &tri_ai[ta];
        Vec2 Atri[3] = { wi[TRIS[ta].a], wi[TRIS[ta].b], wi[TRIS[ta].c] };
        for (int tb = 0; tb < NTRI; tb++) {
            const AABB *bTb = &tri_aj[tb];
            if (!aabb_overlap(aTa, bTb)) continue;
            Vec2 Btri[3] = { wj[TRIS[tb].a], wj[TRIS[tb].b], wj[TRIS[tb].c] };
            double depth = 0.0;
            if (tri_sat_penetration(Atri, Btri, &depth)) pen += depth*depth;
        }
    }
    return pen;
}

typedef struct {
    double alpha_L;
    double lambda_ov;
    double mu_out;
} Weights;

typedef struct {
    double f;
    double overlap;
    double outside;
} Fitness;

static Fitness eval_fitness(int N, double L,
                            const double *cx, const double *cy, const double *th,
                            Weights w)
{
    // Precompute world verts, AABBs, triangle AABBs
    Vec2 *world = (Vec2*)malloc((size_t)N * (size_t)NV * sizeof(Vec2));
    AABB *aabb  = (AABB*)malloc((size_t)N * sizeof(AABB));
    AABB *tri_aabb = (AABB*)malloc((size_t)N * (size_t)NTRI * sizeof(AABB));
    if (!world || !aabb || !tri_aabb) die("alloc failed in eval");

    for (int i = 0; i < N; i++) {
        Vec2 *wi = &world[(size_t)i * (size_t)NV];
        build_world_verts(wi, cx[i], cy[i], th[i]);
        aabb[i] = aabb_of_verts(wi);

        AABB *tai = &tri_aabb[(size_t)i * (size_t)NTRI];
        for (int t = 0; t < NTRI; t++) {
            Vec2 p0 = wi[TRIS[t].a];
            Vec2 p1 = wi[TRIS[t].b];
            Vec2 p2 = wi[TRIS[t].c];
            tai[t] = aabb_of_tri_pts(p0, p1, p2);
        }
    }

    // Build spatial grid on centers
    double br = base_bounding_radius();
    double cell = 2.0 * br;
    if (cell < 1e-9) cell = 1e-9;
    int nx = (int)ceil(L / cell);
    int ny = (int)ceil(L / cell);
    if (nx < 1) nx = 1;
    if (ny < 1) ny = 1;

    EvalGrid g;
    g.N = N; g.L = L; g.br = br; g.cell = cell; g.nx = nx; g.ny = ny;
    int nc = nx * ny;
    g.head = (int*)malloc((size_t)nc * sizeof(int));
    g.next = (int*)malloc((size_t)N * sizeof(int));
    if (!g.head || !g.next) die("alloc failed in grid");

    grid_build(&g, cx, cy);

    // Compute penalties
    double out_sum = 0.0;
    for (int i = 0; i < N; i++) out_sum += outside_penalty_aabb(&aabb[i], L);

    int R = (int)ceil((2.0 * br) / cell) + 1;
    if (R < 1) R = 1;

    double ov_sum = 0.0;
    double half = 0.5 * L;

    for (int i = 0; i < N; i++) {
        // determine cell for i
        double fx = (cx[i] + half) / cell;
        double fy = (cy[i] + half) / cell;
        int ix = (int)floor(fx);
        int iy = (int)floor(fy);
        if (ix < 0) ix = 0; if (ix >= nx) ix = nx - 1;
        if (iy < 0) iy = 0; if (iy >= ny) iy = ny - 1;

        for (int dy = -R; dy <= R; dy++) {
            int yy = iy + dy;
            if (yy < 0 || yy >= ny) continue;
            for (int dx = -R; dx <= R; dx++) {
                int xx = ix + dx;
                if (xx < 0 || xx >= nx) continue;

                int c = grid_index(&g, xx, yy);
                for (int j = g.head[c]; j != -1; j = g.next[j]) {
                    if (j <= i) continue;

                    // bounding circle reject
                    double ddx = cx[i] - cx[j];
                    double ddy = cy[i] - cy[j];
                    double d2 = ddx*ddx + ddy*ddy;
                    double Rb = 2.0 * br;
                    if (d2 > Rb*Rb) continue;

                    const Vec2 *wi = &world[(size_t)i * (size_t)NV];
                    const Vec2 *wj = &world[(size_t)j * (size_t)NV];
                    const AABB *ai = &aabb[i];
                    const AABB *aj = &aabb[j];
                    const AABB *tai = &tri_aabb[(size_t)i * (size_t)NTRI];
                    const AABB *taj = &tri_aabb[(size_t)j * (size_t)NTRI];

                    ov_sum += overlap_pair_penalty_one(wi, ai, tai, wj, aj, taj);
                }
            }
        }
    }

    Fitness F;
    F.overlap = ov_sum;
    F.outside = out_sum;
    F.f = w.alpha_L * L + w.lambda_ov * ov_sum + w.mu_out * out_sum;

    free(g.head); free(g.next);
    free(world); free(aabb); free(tri_aabb);
    return F;
}

// ---------------- Output: CSV + SVG ----------------

static int write_csv(const char *path, int N, double L,
                     const double *cx, const double *cy, const double *th,
                     Fitness bestF)
{
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    fprintf(f, "# L=%.17g f=%.17g overlap=%.17g outside=%.17g N=%d\n",
            L, bestF.f, bestF.overlap, bestF.outside, N);
    fprintf(f, "i,cx,cy,theta_rad\n");
    for (int i = 0; i < N; i++) fprintf(f, "%d,%.17g,%.17g,%.17g\n", i, cx[i], cy[i], th[i]);
    fclose(f);
    return 1;
}

static int write_svg(const char *path, int N, double L,
                     const double *cx, const double *cy, const double *th,
                     Fitness bestF,
                     int width_px, int height_px, double margin_px)
{
    FILE *f = fopen(path, "w");
    if (!f) return 0;

    double half = 0.5 * L;
    double Wpx = (double)width_px, Hpx = (double)height_px;
    double sx = (Wpx - 2.0*margin_px)/L;
    double sy = (Hpx - 2.0*margin_px)/L;
    double scale = (sx < sy) ? sx : sy;

    double square_px = L * scale;
    double ox = margin_px + 0.5*(Wpx - 2.0*margin_px - square_px);
    double oy = margin_px + 0.5*(Hpx - 2.0*margin_px - square_px);

    fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    fprintf(f, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">\n",
            width_px, height_px, width_px, height_px);
    fprintf(f, "  <rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" fill=\"white\"/>\n", width_px, height_px);
    fprintf(f, "  <text x=\"%.1f\" y=\"%.1f\" font-family=\"monospace\" font-size=\"12\">L=%.6g  f=%.3e  ov=%.3e out=%.3e  N=%d</text>\n",
            10.0, 18.0, L, bestF.f, bestF.overlap, bestF.outside, N);
    fprintf(f, "  <rect x=\"%.6f\" y=\"%.6f\" width=\"%.6f\" height=\"%.6f\" fill=\"none\" stroke=\"#000\" stroke-width=\"2\"/>\n",
            ox, oy, square_px, square_px);

    for (int i = 0; i < N; i++) {
        Vec2 wv[NV];
        build_world_verts(wv, cx[i], cy[i], th[i]);
        fprintf(f, "  <path d=\"");
        for (int k = 0; k < NV; k++) {
            double px = ox + (wv[k].x + half) * scale;
            double py = oy + (half - wv[k].y) * scale;
            fprintf(f, "%c%.6f %.6f ", (k==0?'M':'L'), px, py);
        }
        fprintf(f, "Z\" fill=\"#888\" fill-opacity=\"0.18\" stroke=\"#000\" stroke-width=\"1\"/>\n");
    }

    fprintf(f, "</svg>\n");
    fclose(f);
    return 1;
}

// ---------------- Linear algebra: Cholesky, triangular solves ----------------

static int chol_decomp(int n, const double *A, double *L) {
    // A symmetric positive definite; output lower-triangular L such that A = L L^T
    // A and L are row-major n*n
    for (int i = 0; i < n*n; i++) L[i] = 0.0;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = A[i*n + j];
            for (int k = 0; k < j; k++) sum -= L[i*n + k] * L[j*n + k];
            if (i == j) {
                if (sum <= 0.0) return 0;
                L[i*n + j] = sqrt(sum);
            } else {
                L[i*n + j] = sum / L[j*n + j];
            }
        }
    }
    return 1;
}

static void lower_tri_mv(int n, const double *L, const double *z, double *out) {
    // out = L z
    for (int i = 0; i < n; i++) {
        double s = 0.0;
        for (int j = 0; j <= i; j++) s += L[i*n + j] * z[j];
        out[i] = s;
    }
}

static void mat_identity(int n, double *A) {
    for (int i = 0; i < n*n; i++) A[i] = 0.0;
    for (int i = 0; i < n; i++) A[i*n + i] = 1.0;
}

// ---------------- CMA-ES core ----------------

typedef struct {
    int n;          // dimension
    int lambda;     // population
    int mu;         // parents
    double *w;      // weights (mu)
    double mueff;

    double sigma;   // step size

    double *m;      // mean (n)
    double *C;      // covariance (n*n)
    double *B;      // (unused explicitly; we use Cholesky L of C)
    double *L;      // Cholesky of C (n*n lower)
    double *pc;     // evolution path for C (n)
    double *ps;     // evolution path for sigma (n)

    double cs, ds, cc, c1, cmu;
    double chiN;

    int eigeneval;  // counter for decomposition
} CMAES;

static int cmaes_init(CMAES *c, int n, int lambda, double sigma0,
                      const double *m0)
{
    c->n = n;
    c->lambda = lambda;

    // default mu = lambda/2, log weights
    c->mu = lambda / 2;
    if (c->mu < 1) c->mu = 1;

    c->w = (double*)malloc((size_t)c->mu * sizeof(double));
    c->m = (double*)malloc((size_t)n * sizeof(double));
    c->C = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
    c->L = (double*)malloc((size_t)n * (size_t)n * sizeof(double));
    c->pc = (double*)malloc((size_t)n * sizeof(double));
    c->ps = (double*)malloc((size_t)n * sizeof(double));
    if (!c->w || !c->m || !c->C || !c->L || !c->pc || !c->ps) return 0;

    // weights
    double sumw = 0.0;
    for (int i = 0; i < c->mu; i++) {
        c->w[i] = log((double)(c->mu + 0.5)) - log((double)(i + 1));
        sumw += c->w[i];
    }
    for (int i = 0; i < c->mu; i++) c->w[i] /= sumw;

    double sumw2 = 0.0;
    for (int i = 0; i < c->mu; i++) sumw2 += c->w[i]*c->w[i];
    c->mueff = 1.0 / sumw2;

    // strategy params (standard)
    double nn = (double)n;
    c->cs = (c->mueff + 2.0) / (nn + c->mueff + 5.0);
    c->ds = 1.0 + c->cs + 2.0 * fmax(0.0, sqrt((c->mueff - 1.0)/(nn + 1.0)) - 1.0);
    c->cc = (4.0 + c->mueff/nn) / (nn + 4.0 + 2.0*c->mueff/nn);
    c->c1 = 2.0 / ((nn + 1.3)*(nn + 1.3) + c->mueff);
    c->cmu = fmin(1.0 - c->c1,
                  2.0*(c->mueff - 2.0 + 1.0/c->mueff) / ((nn + 2.0)*(nn + 2.0) + c->mueff));
    c->chiN = sqrt(nn) * (1.0 - 1.0/(4.0*nn) + 1.0/(21.0*nn*nn));

    c->sigma = sigma0;

    for (int i = 0; i < n; i++) c->m[i] = m0[i];
    mat_identity(n, c->C);
    for (int i = 0; i < n; i++) { c->pc[i] = 0.0; c->ps[i] = 0.0; }
    c->eigeneval = 0;

    if (!chol_decomp(n, c->C, c->L)) return 0;
    return 1;
}

static void cmaes_free(CMAES *c) {
    free(c->w); free(c->m); free(c->C); free(c->L); free(c->pc); free(c->ps);
    c->w = c->m = c->C = c->L = c->pc = c->ps = NULL;
}

typedef struct {
    double f;
    double *x; // n
} Candidate;

static int cmp_cand(const void *a, const void *b) {
    const Candidate *A = (const Candidate*)a;
    const Candidate *B = (const Candidate*)b;
    if (A->f < B->f) return -1;
    if (A->f > B->f) return  1;
    return 0;
}

// Note: for simplicity we treat C = L L^T and approximate the "whitening" needed for ps update
// by using inv(L) via forward solve for each y. This keeps the implementation self-contained.

static void forward_solve_L(int n, const double *L, const double *b, double *x) {
    // Solve L x = b where L is lower triangular
    for (int i = 0; i < n; i++) {
        double s = b[i];
        for (int j = 0; j < i; j++) s -= L[i*n + j] * x[j];
        x[i] = s / L[i*n + i];
    }
}

static void cmaes_step(CMAES *c, RNG *rng,
                       double (*fitness_fn)(const double *x, void *ud, Fitness *out_components),
                       void *ud,
                       Candidate *pop,
                       double *best_x, double *best_f, Fitness *best_components,
                       int eval_budget, int *evals_used)
{
    int n = c->n;
    int lambda = c->lambda;

    // Sample population: x_k = m + sigma * L * z, z ~ N(0,I)
    for (int k = 0; k < lambda; k++) {
        double *x = pop[k].x;

        // z
        double *z = (double*)malloc((size_t)n * sizeof(double));
        double *y = (double*)malloc((size_t)n * sizeof(double));
        if (!z || !y) die("alloc failed in sampling");

        for (int i = 0; i < n; i++) z[i] = rng_normal(rng);
        lower_tri_mv(n, c->L, z, y);

        for (int i = 0; i < n; i++) x[i] = c->m[i] + c->sigma * y[i];

        Fitness comp = {0};
        double f = fitness_fn(x, ud, &comp);
        pop[k].f = f;

        if (f < *best_f) {
            *best_f = f;
            for (int i = 0; i < n; i++) best_x[i] = x[i];
            if (best_components) *best_components = comp;
        }

        free(z); free(y);

        (*evals_used)++;
        if (*evals_used >= eval_budget) break;
    }

    // sort by fitness
    qsort(pop, (size_t)lambda, sizeof(Candidate), cmp_cand);

    // compute new mean m = sum_{i=1..mu} w_i x_i
    double *m_old = (double*)malloc((size_t)n * sizeof(double));
    double *m_new = (double*)malloc((size_t)n * sizeof(double));
    if (!m_old || !m_new) die("alloc failed in update");
    for (int i = 0; i < n; i++) m_old[i] = c->m[i];
    for (int i = 0; i < n; i++) m_new[i] = 0.0;

    for (int j = 0; j < c->mu; j++) {
        double wj = c->w[j];
        double *xj = pop[j].x;
        for (int i = 0; i < n; i++) m_new[i] += wj * xj[i];
    }
    for (int i = 0; i < n; i++) c->m[i] = m_new[i];

    // y_w = (m_new - m_old) / sigma
    double *y_w = (double*)malloc((size_t)n * sizeof(double));
    double *z_w = (double*)malloc((size_t)n * sizeof(double));
    if (!y_w || !z_w) die("alloc failed");
    for (int i = 0; i < n; i++) y_w[i] = (m_new[i] - m_old[i]) / c->sigma;

    // z_w = inv(L) * y_w  (approx whitened)
    forward_solve_L(n, c->L, y_w, z_w);

    // update ps
    for (int i = 0; i < n; i++) {
        c->ps[i] = (1.0 - c->cs) * c->ps[i] + sqrt(c->cs*(2.0 - c->cs)*c->mueff) * z_w[i];
    }

    // compute hsig
    double ps_norm = 0.0;
    for (int i = 0; i < n; i++) ps_norm += c->ps[i]*c->ps[i];
    ps_norm = sqrt(ps_norm);
    int hsig = (ps_norm / sqrt(1.0 - pow(1.0 - c->cs, 2.0*(c->eigeneval + 1.0))) / c->chiN) < (1.4 + 2.0/(n+1.0));

    // update pc
    for (int i = 0; i < n; i++) {
        c->pc[i] = (1.0 - c->cc) * c->pc[i] + (hsig ? 1.0 : 0.0) * sqrt(c->cc*(2.0 - c->cc)*c->mueff) * y_w[i];
    }

    // rank-one update term: pc pc^T
    // rank-mu update term: sum w_i * y_i y_i^T, where y_i = (x_i - m_old)/sigma
    double *C_new = (double*)malloc((size_t)n*(size_t)n*sizeof(double));
    if (!C_new) die("alloc failed C_new");
    for (int i = 0; i < n*n; i++) C_new[i] = (1.0 - c->c1 - c->cmu) * c->C[i];

    // (optional) correction when hsig==0
    double c1a = c->c1 * (1.0 - (1.0 - hsig) * c->cc * (2.0 - c->cc));
    // rank-one
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            C_new[i*n + j] += c1a * c->pc[i] * c->pc[j];
        }
    }
    // rank-mu
    for (int k = 0; k < c->mu; k++) {
        double wk = c->w[k];
        double *xk = pop[k].x;
        // yk = (xk - m_old)/sigma
        for (int i = 0; i < n; i++) {
            double yi = (xk[i] - m_old[i]) / c->sigma;
            for (int j = 0; j <= i; j++) {
                double yj = (xk[j] - m_old[j]) / c->sigma;
                C_new[i*n + j] += c->cmu * wk * yi * yj;
            }
        }
    }

    // symmetrize
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) C_new[j*n + i] = C_new[i*n + j];
    }

    // commit
    for (int i = 0; i < n*n; i++) c->C[i] = C_new[i];

    // update sigma
    c->sigma *= exp((c->cs/c->ds) * (ps_norm / c->chiN - 1.0));

    // recompute Cholesky occasionally (here: every generation)
    if (!chol_decomp(n, c->C, c->L)) {
        // numerical fallback: reset covariance if it goes non-PD
        mat_identity(n, c->C);
        if (!chol_decomp(n, c->C, c->L)) die("Cholesky failed even after reset");
        c->sigma *= 0.5;
    }

    c->eigeneval++;

    free(m_old); free(m_new); free(y_w); free(z_w); free(C_new);
}

// ---------------- Packing-specific fitness wrapper ----------------

typedef struct {
    int N;
    int D;
    double L;
    Weights w;
} PackCtx;

static double pack_fitness(const double *x, void *ud, Fitness *out_comp) {
    PackCtx *ctx = (PackCtx*)ud;
    int N = ctx->N;
    double L = ctx->L;

    // unpack + apply mild bounding (important; CMA can wander far)
    double *cx = (double*)malloc((size_t)N*sizeof(double));
    double *cy = (double*)malloc((size_t)N*sizeof(double));
    double *th = (double*)malloc((size_t)N*sizeof(double));
    if (!cx || !cy || !th) die("alloc failed in pack_fitness");

    double half = 0.5 * L;
    for (int i = 0; i < N; i++) {
        double xi = x[3*i + 0];
        double yi = x[3*i + 1];
        double ti = x[3*i + 2];

        // keep search bounded (soft box); CMA is unconstrained
        cx[i] = clamp(xi, -half, half);
        cy[i] = clamp(yi, -half, half);
        th[i] = wrap_angle_0_2pi(ti);
    }

    Fitness F = eval_fitness(N, L, cx, cy, th, ctx->w);

    // additional penalty if clamping occurred (keeps CMA honest)
    double clamp_pen = 0.0;
    for (int i = 0; i < N; i++) {
        double xi = x[3*i + 0];
        double yi = x[3*i + 1];
        double dx = xi - cx[i];
        double dy = yi - cy[i];
        clamp_pen += 1e2 * (dx*dx + dy*dy);
    }
    F.f += clamp_pen;

    if (out_comp) *out_comp = F;

    free(cx); free(cy); free(th);
    return F.f;
}

// ---------------- Main ----------------

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s --N N --L L [--evals E] [--lambda P] [--seed S]\n"
        "       [--sigma0 s] [--lamov v] [--muout u]\n"
        "\n"
        "Example:\n"
        "  %s --N 7 --L 1.50 --evals 100000 --lambda 32 --seed 1\n",
        argv0, argv0
    );
}

int main(int argc, char **argv) {
    int N = -1;
    double L = -1.0;
    int evals = 100000;
    int lambda = 0;
    uint64_t seed = 1;
    double sigma0 = 0.30;

    Weights w;
    w.alpha_L = 0.0;
    w.lambda_ov = 1e4;
    w.mu_out = 1e4;

    for (int i = 1; i < argc; i++) {
        if (streq(argv[i], "--N") && i+1 < argc) { N = atoi(argv[++i]); }
        else if (streq(argv[i], "--L") && i+1 < argc) { L = atof(argv[++i]); }
        else if (streq(argv[i], "--evals") && i+1 < argc) { evals = atoi(argv[++i]); }
        else if (streq(argv[i], "--lambda") && i+1 < argc) { lambda = atoi(argv[++i]); }
        else if (streq(argv[i], "--seed") && i+1 < argc) { seed = (uint64_t)strtoull(argv[++i], NULL, 10); }
        else if (streq(argv[i], "--sigma0") && i+1 < argc) { sigma0 = atof(argv[++i]); }
        else if (streq(argv[i], "--lamov") && i+1 < argc) { w.lambda_ov = atof(argv[++i]); }
        else if (streq(argv[i], "--muout") && i+1 < argc) { w.mu_out = atof(argv[++i]); }
        else { usage(argv[0]); return 1; }
    }

    if (N <= 0 || L <= 0.0) { usage(argv[0]); return 1; }

    ensure_dir("csv");
    ensure_dir("img");

    int D = 3 * N;
    if (lambda <= 0) {
        // common heuristic: 4 + floor(3*log(n))
        lambda = 4 + (int)floor(3.0 * log((double)D));
        if (lambda < 8) lambda = 8;
    }

    RNG rng;
    rng_seed(&rng, seed);

    // Initial mean: random within square, theta uniform
    double *m0 = (double*)malloc((size_t)D * sizeof(double));
    if (!m0) die("alloc m0");

    double half = 0.5 * L;
    for (int i = 0; i < N; i++) {
        m0[3*i + 0] = (2.0*rng_u01(&rng) - 1.0) * half * 0.7;
        m0[3*i + 1] = (2.0*rng_u01(&rng) - 1.0) * half * 0.7;
        m0[3*i + 2] = rng_u01(&rng) * 2.0 * M_PI;
    }

    PackCtx ctx;
    ctx.N = N; ctx.D = D; ctx.L = L; ctx.w = w;

    CMAES cma;
    if (!cmaes_init(&cma, D, lambda, sigma0, m0)) die("cma init failed");

    Candidate *pop = (Candidate*)malloc((size_t)lambda * sizeof(Candidate));
    if (!pop) die("alloc pop");
    for (int k = 0; k < lambda; k++) {
        pop[k].x = (double*)malloc((size_t)D * sizeof(double));
        if (!pop[k].x) die("alloc pop[k].x");
        pop[k].f = 1e300;
    }

    double *best_x = (double*)malloc((size_t)D * sizeof(double));
    if (!best_x) die("alloc best_x");
    double best_f = 1e300;
    Fitness best_comp = {0};

    int evals_used = 0;
    int gen = 0;

    printf("CMA-ES packing demo: N=%d D=%d L=%.6g lambda=%d mu=%d evals=%d seed=%llu\n",
           N, D, L, cma.lambda, cma.mu, evals, (unsigned long long)seed);
    printf("Weights: lamov=%.3e muout=%.3e sigma0=%.3g\n", w.lambda_ov, w.mu_out, sigma0);

    while (evals_used < evals) {
        cmaes_step(&cma, &rng, pack_fitness, &ctx,
                   pop, best_x, &best_f, &best_comp, evals, &evals_used);

        gen++;

        if (gen % 10 == 0 || evals_used >= evals) {
            printf("gen=%d evals=%d best_f=%.3e ov=%.3e out=%.3e sigma=%.3g\n",
                   gen, evals_used, best_f, best_comp.overlap, best_comp.outside, cma.sigma);
        }

        // simple stop if essentially feasible
        if (best_comp.overlap < 1e-12 && best_comp.outside < 1e-12) break;
    }

    // unpack best_x to cx,cy,th
    double *cx = (double*)malloc((size_t)N*sizeof(double));
    double *cy = (double*)malloc((size_t)N*sizeof(double));
    double *th = (double*)malloc((size_t)N*sizeof(double));
    if (!cx || !cy || !th) die("alloc unpack");
    for (int i = 0; i < N; i++) {
        cx[i] = clamp(best_x[3*i + 0], -half, half);
        cy[i] = clamp(best_x[3*i + 1], -half, half);
        th[i] = wrap_angle_0_2pi(best_x[3*i + 2]);
    }

    // recompute exact components for final report (no clamp penalty)
    Fitness finalF = eval_fitness(N, L, cx, cy, th, w);
    printf("FINAL: f=%.6e ov=%.6e out=%.6e\n", finalF.f, finalF.overlap, finalF.outside);

    char csv_path[256], svg_path[256];
    snprintf(csv_path, sizeof(csv_path), "csv/cma_best_N%03d.csv", N);
    snprintf(svg_path, sizeof(svg_path), "img/cma_best_N%03d.svg", N);

    if (!write_csv(csv_path, N, L, cx, cy, th, finalF)) fprintf(stderr, "Failed write %s\n", csv_path);
    if (!write_svg(svg_path, N, L, cx, cy, th, finalF, 1100, 1100, 40.0)) fprintf(stderr, "Failed write %s\n", svg_path);

    printf("Wrote: %s\nWrote: %s\n", csv_path, svg_path);

    // cleanup
    free(cx); free(cy); free(th);
    free(best_x);
    for (int k = 0; k < lambda; k++) free(pop[k].x);
    free(pop);
    cmaes_free(&cma);
    free(m0);

    return 0;
}
