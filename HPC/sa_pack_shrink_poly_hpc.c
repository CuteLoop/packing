// sa_pack_shrink_poly_hpc.c
// ------------------------------------------------------------
// Simulated annealing for NON-CONVEX polygon packing (ChristmasTree) in a square,
// using triangulation + SAT (triangle-triangle) for collisions, and an OUTER LOOP
// to shrink the square side length L.
//
// UPDATE (as requested): “leave running” mode + robust best-L search:
//   - Adds a long-running outer optimizer after the initial bracket/bisect,
//     using adaptive multiplicative shrink steps with backoff (stochastic descent).
//   - Periodic checkpoints (CSV + SVG) even if no improvement, so HPC jobs always
//     produce artifacts.
//   - Optional wall-time limit; otherwise runs until killed.
//
// CLI:
//   ./sa_pack_shrink_poly_hpc N [seed]
//       [--init path] [--demo] [--threads K]
//       [--time_limit SEC]            (default: 0 = run forever)
//       [--checkpoint_every SEC]      (default: 600)
//       [--polish]                    (enable long-running shrink search; default ON)
//       [--no_polish]                 (disable long-running shrink search)
//       [--min_shrink X]              (default: 1e-5)   // min fractional shrink
//       [--max_shrink X]              (default: 2e-3)   // max fractional shrink
//       [--target_success P]          (default: 0.35)   // success rate target for shrink attempts
//       [--trials_polish K]           (default: 5)      // SA trials per polish attempt
//
// Outputs (always under csv/ and img/):
//   csv/best_polys_N###.csv  (best feasible)
//   img/best_N###.svg
//   csv/checkpoint_N###.csv  (periodic checkpoint of current best feasible)
//   img/checkpoint_N###.svg
//
// Compile:
//   gcc -O3 -march=native -fopenmp -std=c11 -Wall -Wextra -pedantic sa_pack_shrink_poly_hpc.c -o sa_pack_shrink_poly_hpc -lm
//
// ------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------- small utils ----------------

static int streq(const char *a, const char *b) { return strcmp(a, b) == 0; }

static int file_exists(const char *path) {
    FILE *f = fopen(path, "r");
    if (f) { fclose(f); return 1; }
    return 0;
}

static void ensure_dir(const char *name) {
    if (mkdir(name, 0755) == 0) return;
    if (errno == EEXIST) return;
    fprintf(stderr, "ERROR: could not create dir '%s' (errno=%d)\n", name, errno);
    exit(1);
}

static double now_seconds(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)time(NULL);
#endif
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "Usage:\n"
        "  %s N [seed] [--init path] [--demo] [--threads K]\n"
        "       [--time_limit SEC] [--checkpoint_every SEC]\n"
        "       [--polish | --no_polish]\n"
        "       [--min_shrink X] [--max_shrink X] [--target_success P]\n"
        "       [--trials_polish K]\n"
        "\n"
        "Outputs:\n"
        "  csv/best_polys_N###.csv\n"
        "  img/best_N###.svg\n"
        "  csv/checkpoint_N###.csv\n"
        "  img/checkpoint_N###.svg\n",
        argv0
    );
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

static double rng_uniform(RNG *rng, double a, double b) {
    return a + (b - a) * rng_u01(rng);
}

static double wrap_angle_0_2pi(double th) {
    double two = 2.0 * M_PI;
    th = fmod(th, two);
    if (th < 0.0) th += two;
    return th;
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

// Base (local) vertices (unscaled), matching Python definition.
static const Vec2 BASE_V[15] = {
    {  0.0,     0.8  },  // 0 tip
    {  0.125,   0.5  },  // 1 top_w/2 , tier1
    {  0.0625,  0.5  },  // 2 top_w/4 , tier1
    {  0.2,     0.25 },  // 3 mid_w/2 , tier2
    {  0.1,     0.25 },  // 4 mid_w/4 , tier2
    {  0.35,    0.0  },  // 5 base_w/2 , base
    {  0.075,   0.0  },  // 6 trunk_w/2, base
    {  0.075,  -0.2  },  // 7 trunk_w/2, trunk_bottom
    { -0.075,  -0.2  },  // 8 -trunk_w/2, trunk_bottom
    { -0.075,   0.0  },  // 9 -trunk_w/2, base
    { -0.35,    0.0  },  // 10 -base_w/2, base
    { -0.1,     0.25 },  // 11 -mid_w/4, tier2
    { -0.2,     0.25 },  // 12 -mid_w/2, tier2
    { -0.0625,  0.5  },  // 13 -top_w/4, tier1
    { -0.125,   0.5  },  // 14 -top_w/2, tier1
};

static double base_bounding_radius(void) {
    double rmax2 = 0.0;
    for (int i = 0; i < NV; i++) {
        double d2 = BASE_V[i].x * BASE_V[i].x + BASE_V[i].y * BASE_V[i].y;
        if (d2 > rmax2) rmax2 = d2;
    }
    return sqrt(rmax2);
}

static double base_polygon_area(void) {
    double a = 0.0;
    for (int i = 0; i < NV; i++) {
        int j = (i + 1) % NV;
        a += BASE_V[i].x * BASE_V[j].y - BASE_V[j].x * BASE_V[i].y;
    }
    a = 0.5 * fabs(a);
    return a;
}

// ---------------- Transform + AABB ----------------

static inline Vec2 rot_trans(Vec2 v, double c, double s, double tx, double ty) {
    Vec2 out;
    out.x = c * v.x - s * v.y + tx;
    out.y = s * v.x + c * v.y + ty;
    return out;
}

static void build_world_verts(const Vec2 *base, Vec2 *world, double cx, double cy, double theta) {
    double c = cos(theta), s = sin(theta);
    for (int i = 0; i < NV; i++) world[i] = rot_trans(base[i], c, s, cx, cy);
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

static inline AABB aabb_of_tri_pts(const Vec2 p0, const Vec2 p1, const Vec2 p2) {
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

static inline int aabb_overlap(const AABB *a, const AABB *b) {
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

// ---------------- SAT for triangle-triangle ----------------

static inline void proj3(const Vec2 p[3], double ax, double ay, double *mn, double *mx) {
    double v0 = p[0].x * ax + p[0].y * ay;
    double v1 = p[1].x * ax + p[1].y * ay;
    double v2 = p[2].x * ax + p[2].y * ay;

    double lo = v0, hi = v0;
    if (v1 < lo) lo = v1;
    if (v1 > hi) hi = v1;
    if (v2 < lo) lo = v2;
    if (v2 > hi) hi = v2;

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

            double ax = -ey;
            double ay =  ex;

            double len2 = ax * ax + ay * ay;
            if (len2 < 1e-30) continue;

            double amin, amax, bmin, bmax;
            proj3(A, ax, ay, &amin, &amax);
            proj3(B, ax, ay, &bmin, &bmax);

            double o = fmin(amax, bmax) - fmax(amin, bmin);
            if (o <= 0.0) return 0;
            if (o < min_overlap) min_overlap = o;
        }
    }

    if (min_overlap > 1e200) min_overlap = 0.0;
    *depth_out = min_overlap;
    return 1;
}

// ---------------- Uniform Grid (spatial hash) ----------------

typedef struct {
    double L;
    double cell;
    int nx, ny;
    double half;

    int *head;     // nx*ny
    int *next;     // N
    int *prev;     // N
    int *cell_id;  // N
    int N;
} Grid;

static inline int clampi(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline int grid_index(const Grid *g, int ix, int iy) {
    return iy * g->nx + ix;
}

static inline void grid_cell_xy(const Grid *g, double x, double y, int *ix, int *iy) {
    double fx = (x + g->half) / g->cell;
    double fy = (y + g->half) / g->cell;
    int cx = (int)floor(fx);
    int cy = (int)floor(fy);
    cx = clampi(cx, 0, g->nx - 1);
    cy = clampi(cy, 0, g->ny - 1);
    *ix = cx; *iy = cy;
}

static void grid_init(Grid *g, int N, double L, double cell) {
    g->N = N;
    g->L = L;
    g->half = 0.5 * L;
    g->cell = cell;

    g->nx = (int)ceil(L / cell);
    g->ny = (int)ceil(L / cell);
    if (g->nx < 1) g->nx = 1;
    if (g->ny < 1) g->ny = 1;

    int nc = g->nx * g->ny;

    g->head = (int*)malloc((size_t)nc * sizeof(int));
    g->next = (int*)malloc((size_t)N * sizeof(int));
    g->prev = (int*)malloc((size_t)N * sizeof(int));
    g->cell_id = (int*)malloc((size_t)N * sizeof(int));

    if (!g->head || !g->next || !g->prev || !g->cell_id) {
        fprintf(stderr, "grid alloc failed\n");
        exit(1);
    }

    for (int c = 0; c < nc; c++) g->head[c] = -1;
    for (int i = 0; i < N; i++) {
        g->next[i] = -1;
        g->prev[i] = -1;
        g->cell_id[i] = -1;
    }
}

static void grid_free(Grid *g) {
    free(g->head); free(g->next); free(g->prev); free(g->cell_id);
    g->head = g->next = g->prev = g->cell_id = NULL;
    g->nx = g->ny = 0;
}

static void grid_insert(Grid *g, int i, double x, double y) {
    int ix, iy;
    grid_cell_xy(g, x, y, &ix, &iy);
    int cid = grid_index(g, ix, iy);

    int h = g->head[cid];
    g->prev[i] = -1;
    g->next[i] = h;
    if (h != -1) g->prev[h] = i;
    g->head[cid] = i;

    g->cell_id[i] = cid;
}

static void grid_remove(Grid *g, int i) {
    int cid = g->cell_id[i];
    if (cid < 0) return;

    int pi = g->prev[i];
    int ni = g->next[i];

    if (pi != -1) g->next[pi] = ni;
    else g->head[cid] = ni;

    if (ni != -1) g->prev[ni] = pi;

    g->prev[i] = g->next[i] = -1;
    g->cell_id[i] = -1;
}

static void grid_update(Grid *g, int i, double x, double y) {
    int ix, iy;
    grid_cell_xy(g, x, y, &ix, &iy);
    int new_cid = grid_index(g, ix, iy);
    int old_cid = g->cell_id[i];

    if (old_cid == new_cid) return;
    if (old_cid != -1) grid_remove(g, i);
    grid_insert(g, i, x, y);
}

static void grid_rebuild(Grid *g, int N, double L, double cell,
                         const double *cx, const double *cy)
{
    grid_free(g);
    grid_init(g, N, L, cell);
    for (int i = 0; i < N; i++) grid_insert(g, i, cx[i], cy[i]);
}

// ---------------- Packing state ----------------

typedef struct {
    int N;
    double L;

    double *cx;
    double *cy;
    double *th;

    Vec2 *world;     // N*NV
    AABB *aabb;      // N
    AABB *tri_aabb;  // N*NTRI (cached triangle AABBs)

    double br;       // base bounding radius

    Grid grid;
    double cell;
} State;

typedef struct {
    double alpha_L;
    double lambda_ov;
    double mu_out;
} Weights;

typedef struct {
    double overlap_total;
    double out_total;
} Totals;

static inline Vec2* W(State *s, int i) { return &s->world[(size_t)i * (size_t)NV]; }
static inline const Vec2* Wc(const State *s, int i) { return &s->world[(size_t)i * (size_t)NV]; }

static inline AABB* TA(State *s, int i) { return &s->tri_aabb[(size_t)i * (size_t)NTRI]; }
static inline const AABB* TAc(const State *s, int i) { return &s->tri_aabb[(size_t)i * (size_t)NTRI]; }

static State state_alloc(int N) {
    State s;
    s.N = N;
    s.L = 1.0;

    s.cx = (double*)calloc((size_t)N, sizeof(double));
    s.cy = (double*)calloc((size_t)N, sizeof(double));
    s.th = (double*)calloc((size_t)N, sizeof(double));

    s.world = (Vec2*)calloc((size_t)N * (size_t)NV, sizeof(Vec2));
    s.aabb  = (AABB*)calloc((size_t)N, sizeof(AABB));
    s.tri_aabb = (AABB*)calloc((size_t)N * (size_t)NTRI, sizeof(AABB));

    if (!s.cx || !s.cy || !s.th || !s.world || !s.aabb || !s.tri_aabb) {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }

    s.br = base_bounding_radius();

    s.cell = 2.0 * s.br;
    if (s.cell < 1e-9) s.cell = 1e-9;

    grid_init(&s.grid, N, s.L, s.cell);
    return s;
}

static void state_free(State *s) {
    free(s->cx); free(s->cy); free(s->th);
    free(s->world); free(s->aabb); free(s->tri_aabb);

    s->cx = s->cy = s->th = NULL;
    s->world = NULL;
    s->aabb = NULL;
    s->tri_aabb = NULL;

    grid_free(&s->grid);
}

static void update_instance(State *s, int i) {
    build_world_verts(BASE_V, W(s, i), s->cx[i], s->cy[i], s->th[i]);
    s->aabb[i] = aabb_of_verts(W(s, i));

    const Vec2 *wi = Wc(s, i);
    AABB *tab = TA(s, i);
    for (int t = 0; t < NTRI; t++) {
        Vec2 p0 = wi[TRIS[t].a];
        Vec2 p1 = wi[TRIS[t].b];
        Vec2 p2 = wi[TRIS[t].c];
        tab[t] = aabb_of_tri_pts(p0, p1, p2);
    }
}

static void update_all(State *s) {
    for (int i = 0; i < s->N; i++) update_instance(s, i);
}

// ---------------- Pair overlap penalty ----------------

static inline int bounding_circle_reject(const State *s, int i, int j) {
    double dx = s->cx[i] - s->cx[j];
    double dy = s->cy[i] - s->cy[j];
    double d2 = dx * dx + dy * dy;
    double R = 2.0 * s->br;
    return (d2 > R * R);
}

static double overlap_pair_penalty(const State *s, int i, int j) {
    if (!aabb_overlap(&s->aabb[i], &s->aabb[j])) return 0.0;
    if (bounding_circle_reject(s, i, j)) return 0.0;

    const Vec2 *wi = Wc(s, i);
    const Vec2 *wj = Wc(s, j);
    const AABB *ai = TAc(s, i);
    const AABB *aj = TAc(s, j);

    double pen = 0.0;

#ifdef _OPENMP
    #pragma omp parallel for reduction(+:pen) schedule(static)
    for (int ta = 0; ta < NTRI; ta++) {
        const AABB *aTa = &ai[ta];
        Vec2 Atri[3] = { wi[TRIS[ta].a], wi[TRIS[ta].b], wi[TRIS[ta].c] };

        for (int tb = 0; tb < NTRI; tb++) {
            if (!aabb_overlap(aTa, &aj[tb])) continue;
            Vec2 Btri[3] = { wj[TRIS[tb].a], wj[TRIS[tb].b], wj[TRIS[tb].c] };
            double depth = 0.0;
            if (tri_sat_penetration(Atri, Btri, &depth)) pen += depth * depth;
        }
    }
#else
    for (int ta = 0; ta < NTRI; ta++) {
        const AABB *aTa = &ai[ta];
        Vec2 Atri[3] = { wi[TRIS[ta].a], wi[TRIS[ta].b], wi[TRIS[ta].c] };

        for (int tb = 0; tb < NTRI; tb++) {
            if (!aabb_overlap(aTa, &aj[tb])) continue;
            Vec2 Btri[3] = { wj[TRIS[tb].a], wj[TRIS[tb].b], wj[TRIS[tb].c] };
            double depth = 0.0;
            if (tri_sat_penetration(Atri, Btri, &depth)) pen += depth * depth;
        }
    }
#endif

    return pen;
}

static inline int grid_R_cells(const State *s) {
    int R = (int)ceil((2.0 * s->br) / s->grid.cell) + 1;
    if (R < 1) R = 1;
    return R;
}

static double overlap_sum_for_k_grid(const State *s, int k) {
    const Grid *g = &s->grid;
    int cid = g->cell_id[k];
    if (cid < 0) {
        double sum = 0.0;
        for (int j = 0; j < s->N; j++) if (j != k) sum += overlap_pair_penalty(s, k, j);
        return sum;
    }

    int kx = cid % g->nx;
    int ky = cid / g->nx;
    int R = grid_R_cells(s);

    double sum = 0.0;
    for (int dy = -R; dy <= R; dy++) {
        int yy = ky + dy;
        if (yy < 0 || yy >= g->ny) continue;

        for (int dx = -R; dx <= R; dx++) {
            int xx = kx + dx;
            if (xx < 0 || xx >= g->nx) continue;

            int c = grid_index(g, xx, yy);
            for (int j = g->head[c]; j != -1; j = g->next[j]) {
                if (j == k) continue;
                sum += overlap_pair_penalty(s, k, j);
            }
        }
    }
    return sum;
}

static double outside_for_k(const State *s, int k) {
    return outside_penalty_aabb(&s->aabb[k], s->L);
}

static Totals compute_totals_full_grid(const State *s) {
    Totals t;
    t.overlap_total = 0.0;
    t.out_total = 0.0;

    for (int i = 0; i < s->N; i++) t.out_total += outside_for_k(s, i);

    const Grid *g = &s->grid;
    int R = grid_R_cells(s);

    for (int i = 0; i < s->N; i++) {
        int cid = g->cell_id[i];
        if (cid < 0) continue;

        int ix = cid % g->nx;
        int iy = cid / g->nx;

        for (int dy = -R; dy <= R; dy++) {
            int yy = iy + dy;
            if (yy < 0 || yy >= g->ny) continue;

            for (int dx = -R; dx <= R; dx++) {
                int xx = ix + dx;
                if (xx < 0 || xx >= g->nx) continue;

                int c = grid_index(g, xx, yy);
                for (int j = g->head[c]; j != -1; j = g->next[j]) {
                    if (j <= i) continue;
                    t.overlap_total += overlap_pair_penalty(s, i, j);
                }
            }
        }
    }

    return t;
}

static double energy_from_totals(const State *s, const Weights *w, const Totals *t) {
    return w->alpha_L * s->L + w->lambda_ov * t->overlap_total + w->mu_out * t->out_total;
}

static double feasibility_metric(const Totals *t) {
    double ov = t->overlap_total;
    double out = t->out_total;
    if (ov < 0.0) ov = 0.0;
    if (out < 0.0) out = 0.0;
    return ov + out;
}

// ---------------- Initialization helpers ----------------

static void rebuild_grid(State *s) {
    grid_rebuild(&s->grid, s->N, s->L, s->cell, s->cx, s->cy);
}

static void random_init(State *s, RNG *rng) {
    double half = 0.5 * s->L;
    for (int i = 0; i < s->N; i++) {
        s->cx[i] = rng_uniform(rng, -half, half);
        s->cy[i] = rng_uniform(rng, -half, half);
        s->th[i] = wrap_angle_0_2pi(rng_uniform(rng, 0.0, 2.0 * M_PI));
    }
    update_all(s);
    rebuild_grid(s);
}

static void scale_positions_for_new_L(State *s, double oldL, double newL, double safety) {
    if (oldL <= 0 || newL <= 0) return;
    double gamma = (newL / oldL) * safety;
    for (int i = 0; i < s->N; i++) {
        s->cx[i] *= gamma;
        s->cy[i] *= gamma;
    }
    update_all(s);
    rebuild_grid(s);
}

static void base_aabb_local(double *minx, double *miny, double *maxx, double *maxy) {
    *minx = *maxx = BASE_V[0].x;
    *miny = *maxy = BASE_V[0].y;
    for (int i = 1; i < NV; i++) {
        if (BASE_V[i].x < *minx) *minx = BASE_V[i].x;
        if (BASE_V[i].x > *maxx) *maxx = BASE_V[i].x;
        if (BASE_V[i].y < *miny) *miny = BASE_V[i].y;
        if (BASE_V[i].y > *maxy) *maxy = BASE_V[i].y;
    }
}

static void grid_init_layout(State *s, int cols, int rows, double gap, double *L_grid_out) {
    double minx, miny, maxx, maxy;
    base_aabb_local(&minx, &miny, &maxx, &maxy);
    double w = maxx - minx;
    double h = maxy - miny;

    double dx = w + gap;
    double dy = h + gap;

    double gridW = cols * dx;
    double gridH = rows * dy;
    double Lg = (gridW > gridH) ? gridW : gridH;

    double half = 0.5 * Lg;

    for (int i = 0; i < s->N; i++) {
        int r = i / cols;
        int c = i % cols;
        double cx = -half + (c + 0.5) * dx;
        double cy =  half - (r + 0.5) * dy;
        s->cx[i] = cx;
        s->cy[i] = cy;
        s->th[i] = 0.0;
    }

    if (L_grid_out) *L_grid_out = Lg;
}

// ---------------- CSV init loader + optional read L from header ----------------

static int load_init_csv(const char *path, int N,
                         double *cx, double *cy, double *th)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    for (int i = 0; i < N; i++) { cx[i] = 0.0; cy[i] = 0.0; th[i] = 0.0; }

    char line[4096];
    int saw_any = 0;

    while (fgets(line, (int)sizeof(line), f)) {
        if (line[0] == '#') continue;
        if (strstr(line, "i,cx,cy,theta_rad")) continue;

        int i;
        double x, y, t;
        if (sscanf(line, " %d , %lf , %lf , %lf", &i, &x, &y, &t) == 4) {
            if (i >= 0 && i < N) {
                cx[i] = x; cy[i] = y; th[i] = t;
                saw_any = 1;
            }
        }
    }

    fclose(f);
    return saw_any;
}

static int read_L_from_csv_header(const char *path, double *L_out) {
    if (!path || !path[0]) return 0;
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[4096];
    int ok = 0;
    while (fgets(line, (int)sizeof(line), f)) {
        if (line[0] != '#') break;
        // Expect: "# L=... best_feas=... N=..."
        double Ltmp = 0.0;
        if (sscanf(line, " # L=%lf", &Ltmp) == 1) {
            if (Ltmp > 0.0) { *L_out = Ltmp; ok = 1; break; }
        }
    }
    fclose(f);
    return ok;
}

// ---------------- Incremental move bookkeeping ----------------

typedef struct {
    int idx;
    double old_cx, old_cy, old_th;
    int old_cell;
    double d_overlap;
    double d_out;
} Move;

static Move propose_move(State *s, Totals *tot, RNG *rng,
                         double step_xy, double step_th,
                         double p_reinsert, double p_rotmix)
{
    Move m;
    int k = (int)(rng_u01(rng) * (double)s->N);
    if (k < 0) k = 0;
    if (k >= s->N) k = s->N - 1;

    m.idx = k;
    m.old_cx = s->cx[k];
    m.old_cy = s->cy[k];
    m.old_th = s->th[k];
    m.old_cell = s->grid.cell_id[k];

    double old_ov  = overlap_sum_for_k_grid(s, k);
    double old_out = outside_for_k(s, k);

    double u = rng_u01(rng);

    if (u < p_reinsert) {
        double half = 0.5 * s->L;
        s->cx[k] = rng_uniform(rng, -half, half);
        s->cy[k] = rng_uniform(rng, -half, half);
        s->th[k] = wrap_angle_0_2pi(rng_uniform(rng, 0.0, 2.0 * M_PI));
    } else {
        double dx = rng_uniform(rng, -step_xy, step_xy);
        double dy = rng_uniform(rng, -step_xy, step_xy);
        s->cx[k] += dx;
        s->cy[k] += dy;

        if (rng_u01(rng) < p_rotmix) {
            double dth = rng_uniform(rng, -step_th, step_th);
            s->th[k] = wrap_angle_0_2pi(s->th[k] + dth);
        }
    }

    update_instance(s, k);
    grid_update(&s->grid, k, s->cx[k], s->cy[k]);

    double new_ov  = overlap_sum_for_k_grid(s, k);
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

    int k = m->idx;

    s->cx[k] = m->old_cx;
    s->cy[k] = m->old_cy;
    s->th[k] = m->old_th;

    update_instance(s, k);
    grid_update(&s->grid, k, s->cx[k], s->cy[k]);
}

// ---------------- SA params (per phase) ----------------

typedef struct {
    int iters;

    double T_start;
    double T_end;

    double step_xy_start;
    double step_th_start;

    int adapt_window;
    double acc_low;
    double acc_high;
    double step_shrink;
    double step_grow;
    double step_xy_min, step_xy_max;
    double step_th_min, step_th_max;

    double lambda_start;
    double mu_start;
    int ramp_every;
    double ramp_factor;
    double lambda_max;
    double mu_max;

    double p_reinsert;
    double p_rotmix;

    int log_every;
} PhaseParams;

static double cooling_from_range(double T_start, double T_end, int iters) {
    if (iters <= 0) return 1.0;
    if (T_start <= 0) T_start = 1e-12;
    if (T_end   <= 0) T_end   = 1e-12;
    return exp(log(T_end / T_start) / (double)iters);
}

static void run_phase(State *s, Totals *tot, Weights *w, RNG *rng,
                      const PhaseParams *pp,
                      double *best_feas_io, double *best_cx, double *best_cy, double *best_th,
                      int verbose)
{
    double T = pp->T_start;
    double cool = cooling_from_range(pp->T_start, pp->T_end, pp->iters);

    double step_xy = pp->step_xy_start;
    double step_th = pp->step_th_start;

    w->lambda_ov = pp->lambda_start;
    w->mu_out    = pp->mu_start;

    double E = energy_from_totals(s, w, tot);

    int accepts_total = 0;
    int accepts_win = 0, moves_win = 0;

    int log_every = pp->log_every;
    if (log_every < 1) log_every = 1;

    for (int t = 0; t < pp->iters; t++) {
        if (pp->ramp_every > 0 && t > 0 && (t % pp->ramp_every) == 0) {
            w->lambda_ov *= pp->ramp_factor;
            w->mu_out    *= pp->ramp_factor;
            if (w->lambda_ov > pp->lambda_max) w->lambda_ov = pp->lambda_max;
            if (w->mu_out    > pp->mu_max)     w->mu_out    = pp->mu_max;
            E = energy_from_totals(s, w, tot);
        }

        Move m = propose_move(s, tot, rng, step_xy, step_th, pp->p_reinsert, pp->p_rotmix);
        double Enew = energy_from_totals(s, w, tot);
        double dE = Enew - E;

        int accept = 0;
        if (dE <= 0.0) accept = 1;
        else {
            double u = rng_u01(rng);
            double pacc = exp(-dE / T);
            if (u < pacc) accept = 1;
        }

        if (accept) {
            E = Enew;
            accepts_total++;
            accepts_win++;

            double feas = feasibility_metric(tot);
            if (feas < *best_feas_io) {
                *best_feas_io = feas;
                for (int i = 0; i < s->N; i++) {
                    best_cx[i] = s->cx[i];
                    best_cy[i] = s->cy[i];
                    best_th[i] = s->th[i];
                }
            }
        } else {
            undo_move(s, tot, &m);
        }

        moves_win++;

        T *= cool;
        if (T < 1e-12) T = 1e-12;

        if (pp->adapt_window > 0 && moves_win >= pp->adapt_window) {
            double acc = (double)accepts_win / (double)moves_win;
            if (acc < pp->acc_low) { step_xy *= pp->step_shrink; step_th *= pp->step_shrink; }
            else if (acc > pp->acc_high) { step_xy *= pp->step_grow; step_th *= pp->step_grow; }

            if (step_xy < pp->step_xy_min) step_xy = pp->step_xy_min;
            if (step_xy > pp->step_xy_max) step_xy = pp->step_xy_max;
            if (step_th < pp->step_th_min) step_th = pp->step_th_min;
            if (step_th > pp->step_th_max) step_th = pp->step_th_max;

            accepts_win = 0;
            moves_win = 0;
        }

        if (verbose && (t % log_every) == 0) {
            double acc_rate = (double)accepts_total / (double)(t + 1);
            printf("    iter=%d/%d T=%.2e step_xy=%.2e step_th=%.2e E=%.3e ov=%.2e out=%.2e feas=%.2e acc=%.3f lam=%.2e mu=%.2e\n",
                   t, pp->iters, T, step_xy, step_th, E,
                   tot->overlap_total, tot->out_total, feasibility_metric(tot),
                   acc_rate, w->lambda_ov, w->mu_out);
        }
    }
}

static double run_trial(State *s, RNG *rng,
                        const PhaseParams *A, const PhaseParams *B,
                        double *best_cx, double *best_cy, double *best_th,
                        int verbose)
{
    Totals tot = compute_totals_full_grid(s);

    Weights w;
    w.alpha_L = 0.0;

    double best_feas = feasibility_metric(&tot);
    for (int i = 0; i < s->N; i++) {
        best_cx[i] = s->cx[i];
        best_cy[i] = s->cy[i];
        best_th[i] = s->th[i];
    }

    if (verbose) {
        printf("    start: ov=%.2e out=%.2e feas=%.2e\n", tot.overlap_total, tot.out_total, best_feas);
        printf("    Phase A (explore)\n");
    }
    run_phase(s, &tot, &w, rng, A, &best_feas, best_cx, best_cy, best_th, verbose);

    if (verbose) printf("    Phase B (enforce)\n");
    run_phase(s, &tot, &w, rng, B, &best_feas, best_cx, best_cy, best_th, verbose);

    for (int i = 0; i < s->N; i++) {
        s->cx[i] = best_cx[i];
        s->cy[i] = best_cy[i];
        s->th[i] = best_th[i];
    }
    update_all(s);
    rebuild_grid(s);

    Totals best_tot = compute_totals_full_grid(s);
    double check = feasibility_metric(&best_tot);
    if (check < best_feas) best_feas = check;

    if (verbose) {
        printf("    trial best: ov=%.2e out=%.2e feas=%.2e\n", best_tot.overlap_total, best_tot.out_total, best_feas);
    }
    return best_feas;
}

static double try_pack_at_current_L(State *s, RNG *rng,
                                   const PhaseParams *A, const PhaseParams *B,
                                   int trials,
                                   double *out_best_cx, double *out_best_cy, double *out_best_th,
                                   int verbose_trials)
{
    int N = s->N;

    double *trial_best_cx = (double*)malloc((size_t)N * sizeof(double));
    double *trial_best_cy = (double*)malloc((size_t)N * sizeof(double));
    double *trial_best_th = (double*)malloc((size_t)N * sizeof(double));
    if (!trial_best_cx || !trial_best_cy || !trial_best_th) {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }

    double best_feas = 1e300;

    for (int tr = 0; tr < trials; tr++) {
        uint64_t seed = (uint64_t)time(NULL) ^ (uint64_t)(0x9E3779B97F4A7C15ULL * (uint64_t)(tr + 1));
        rng_seed(rng, seed);

        if (tr > 0) random_init(s, rng);

        if (verbose_trials) {
            printf("  - SA trial %d/%d (seed=%llu)\n", tr + 1, trials, (unsigned long long)seed);
        }

        double feas = run_trial(s, rng, A, B, trial_best_cx, trial_best_cy, trial_best_th, verbose_trials);

        if (feas < best_feas) {
            best_feas = feas;
            for (int i = 0; i < N; i++) {
                out_best_cx[i] = trial_best_cx[i];
                out_best_cy[i] = trial_best_cy[i];
                out_best_th[i] = trial_best_th[i];
            }
        }

        for (int i = 0; i < N; i++) {
            s->cx[i] = out_best_cx[i];
            s->cy[i] = out_best_cy[i];
            s->th[i] = out_best_th[i];
        }
        update_all(s);
        rebuild_grid(s);
    }

    free(trial_best_cx);
    free(trial_best_cy);
    free(trial_best_th);
    return best_feas;
}

// ---------------- Export: CSV + SVG ----------------

static int write_polys_csv(const char *path, const State *s, double best_feas) {
    FILE *f = fopen(path, "w");
    if (!f) return 0;

    fprintf(f, "# L=%.17g best_feas=%.17g N=%d\n", s->L, best_feas, s->N);
    fprintf(f, "i,cx,cy,theta_rad\n");
    for (int i = 0; i < s->N; i++) {
        fprintf(f, "%d,%.17g,%.17g,%.17g\n", i, s->cx[i], s->cy[i], s->th[i]);
    }
    fclose(f);
    return 1;
}

static int write_best_svg(const char *path, const State *s, double best_feas,
                          int width_px, int height_px, double margin_px)
{
    FILE *f = fopen(path, "w");
    if (!f) return 0;

    const double L = s->L;
    const double half = 0.5 * L;
    const double Wpx = (double)width_px;
    const double Hpx = (double)height_px;

    const double sx = (Wpx - 2.0 * margin_px) / L;
    const double sy = (Hpx - 2.0 * margin_px) / L;
    const double scale = (sx < sy) ? sx : sy;

    const double square_px = L * scale;
    const double ox = margin_px + 0.5 * (Wpx - 2.0 * margin_px - square_px);
    const double oy = margin_px + 0.5 * (Hpx - 2.0 * margin_px - square_px);

    fprintf(f, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    fprintf(f, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">\n",
            width_px, height_px, width_px, height_px);

    fprintf(f, "  <rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" fill=\"white\"/>\n", width_px, height_px);
    fprintf(f, "  <text x=\"%.1f\" y=\"%.1f\" font-family=\"monospace\" font-size=\"12\">L=%.12g feas=%.3e N=%d</text>\n",
            10.0, 18.0, s->L, best_feas, s->N);

    fprintf(f, "  <rect x=\"%.6f\" y=\"%.6f\" width=\"%.6f\" height=\"%.6f\" fill=\"none\" stroke=\"#000\" stroke-width=\"2\"/>\n",
            ox, oy, square_px, square_px);

    for (int i = 0; i < s->N; i++) {
        const Vec2 *w = Wc(s, i);
        fprintf(f, "  <path d=\"");
        for (int k = 0; k < NV; k++) {
            double px = ox + (w[k].x + half) * scale;
            double py = oy + (half - w[k].y) * scale;
            fprintf(f, "%c%.6f %.6f ", (k == 0 ? 'M' : 'L'), px, py);
        }
        fprintf(f, "Z\" fill=\"#888\" fill-opacity=\"0.18\" stroke=\"#000\" stroke-width=\"1\"/>\n");
    }

    fprintf(f, "</svg>\n");
    fclose(f);
    return 1;
}

// ---------------- Feasibility helpers ----------------

static inline int is_feasible(double feas, double tol) { return (feas <= tol); }

static inline int guaranteed_infeasible_by_area(double L, double area_lb) {
    return (L <= area_lb);
}

// ---------------- N=1 shortcut ----------------

// ---------------- N=1 shortcut (tilted AABB) ----------------

static void solve_N1_bounding_box(State *s) {
    const double theta = M_PI / 4.0;           // 45 degrees
    const double c = cos(theta), ss = sin(theta);

    // Compute AABB of the *rotated* base polygon (rotation about origin)
    double minx = 1e300, miny = 1e300, maxx = -1e300, maxy = -1e300;
    for (int i = 0; i < NV; i++) {
        const double x = BASE_V[i].x;
        const double y = BASE_V[i].y;
        const double rx = c * x - ss * y;
        const double ry = ss * x + c  * y;
        if (rx < minx) minx = rx;
        if (rx > maxx) maxx = rx;
        if (ry < miny) miny = ry;
        if (ry > maxy) maxy = ry;
    }

    const double w = maxx - minx;
    const double h = maxy - miny;

    // Minimal square side length containing that AABB
    s->L = (w > h) ? w : h;

    // Center the polygon's AABB in the square so it is guaranteed inside
    const double cx0 = 0.5 * (minx + maxx);
    const double cy0 = 0.5 * (miny + maxy);

    s->cx[0] = -cx0;
    s->cy[0] = -cy0;
    s->th[0] = theta;

    update_all(s);
    rebuild_grid(s);
}

// ---------------- Config ----------------

typedef struct {
    int N;
    uint64_t seed;
    const char *init_path;
    int demo;
    int threads;

    double time_limit_sec;          // 0 => forever
    double checkpoint_every_sec;    // periodic checkpoint
    int polish;                     // long-running shrink search

    double min_shrink;
    double max_shrink;
    double target_success;
    int trials_polish;
} Config;

static Config parse_args(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); exit(1); }

    Config cfg;
    cfg.N = atoi(argv[1]);
    if (cfg.N <= 0) { usage(argv[0]); exit(1); }

    cfg.seed = (uint64_t)time(NULL);
    cfg.init_path = NULL;
    cfg.demo = 0;
    cfg.threads = -1;

    cfg.time_limit_sec = 0.0;
    cfg.checkpoint_every_sec = 600.0;
    cfg.polish = 1;

    cfg.min_shrink = 1e-5;
    cfg.max_shrink = 2e-3;
    cfg.target_success = 0.35;
    cfg.trials_polish = 5;

    if (argc >= 3 && argv[2][0] != '-') {
        cfg.seed = (uint64_t)strtoull(argv[2], NULL, 10);
        if (cfg.seed == 0) cfg.seed = (uint64_t)time(NULL);
    }

    for (int a = 2; a < argc; a++) {
        if (streq(argv[a], "--demo")) cfg.demo = 1;
        else if (streq(argv[a], "--init") && a + 1 < argc) { cfg.init_path = argv[a + 1]; a++; }
        else if (streq(argv[a], "--threads") && a + 1 < argc) { cfg.threads = atoi(argv[a + 1]); a++; }
        else if (streq(argv[a], "--time_limit") && a + 1 < argc) { cfg.time_limit_sec = atof(argv[a + 1]); a++; }
        else if (streq(argv[a], "--checkpoint_every") && a + 1 < argc) { cfg.checkpoint_every_sec = atof(argv[a + 1]); a++; }
        else if (streq(argv[a], "--polish")) cfg.polish = 1;
        else if (streq(argv[a], "--no_polish")) cfg.polish = 0;
        else if (streq(argv[a], "--min_shrink") && a + 1 < argc) { cfg.min_shrink = atof(argv[a + 1]); a++; }
        else if (streq(argv[a], "--max_shrink") && a + 1 < argc) { cfg.max_shrink = atof(argv[a + 1]); a++; }
        else if (streq(argv[a], "--target_success") && a + 1 < argc) { cfg.target_success = atof(argv[a + 1]); a++; }
        else if (streq(argv[a], "--trials_polish") && a + 1 < argc) { cfg.trials_polish = atoi(argv[a + 1]); a++; }
        else if (argv[a][0] == '-') {
            fprintf(stderr, "Unknown/invalid arg: %s\n", argv[a]);
            usage(argv[0]);
            exit(1);
        }
    }

    if (cfg.checkpoint_every_sec < 10.0) cfg.checkpoint_every_sec = 10.0;
    if (cfg.min_shrink <= 0.0) cfg.min_shrink = 1e-6;
    if (cfg.max_shrink <= cfg.min_shrink) cfg.max_shrink = cfg.min_shrink * 10.0;
    if (cfg.target_success < 0.05) cfg.target_success = 0.05;
    if (cfg.target_success > 0.95) cfg.target_success = 0.95;
    if (cfg.trials_polish < 1) cfg.trials_polish = 1;

    return cfg;
}

// ---------------- Main ----------------

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

#ifdef _OPENMP
    if (cfg.threads > 0) omp_set_num_threads(cfg.threads);
#endif

    ensure_dir("csv");
    ensure_dir("img");

    RNG rng;
    rng_seed(&rng, cfg.seed);

    State s = state_alloc(cfg.N);

    // ---- N=1 shortcut ----
    if (cfg.N == 1) {
        solve_N1_bounding_box(&s);
        Totals t = compute_totals_full_grid(&s);
        double feas = feasibility_metric(&t);

        char csv_path[256], svg_path[256];
        snprintf(csv_path, sizeof(csv_path), "csv/best_polys_N%03d.csv", cfg.N);
        snprintf(svg_path, sizeof(svg_path), "img/best_N%03d.svg", cfg.N);

        printf("N=1 bounding-box solution: L=%.12g feas=%.3e\n", s.L, feas);
        write_polys_csv(csv_path, &s, feas);
        write_best_svg(svg_path, &s, feas, 1100, 1100, 40.0);

        state_free(&s);
        return 0;
    }

    // ---- Area lower bound ----
    const double poly_area = base_polygon_area();
    const double area_lb = sqrt((double)cfg.N * poly_area);
    const double area_lb_infeas = area_lb * (1.0 - 1e-6);

    printf("Area bound: area(poly)=%.12g  sqrt(N*area)=%.12g  (provably infeasible L<=%.12g)\n",
           poly_area, area_lb, area_lb_infeas);

    // ---- Initial L and layout ----
    int cols = (int)ceil(sqrt((double)cfg.N));
    int rows = (int)ceil((double)cfg.N / (double)cols);

    double L_grid = 0.0;
    grid_init_layout(&s, cols, rows, 0.0, &L_grid);

    s.L = 1.15 * L_grid;

    // If init is provided: load it and (optionally) adopt the L in its header
    if (cfg.init_path && cfg.init_path[0] && file_exists(cfg.init_path)) {
        double Linit = 0.0;
        if (read_L_from_csv_header(cfg.init_path, &Linit) && Linit > 0.0) {
            s.L = Linit;
            printf("Init header L=%.12g from %s\n", s.L, cfg.init_path);
        } else {
            printf("Init header L not found; keeping L=%.12g\n", s.L);
        }

        if (load_init_csv(cfg.init_path, s.N, s.cx, s.cy, s.th)) {
            for (int i = 0; i < s.N; i++) s.th[i] = wrap_angle_0_2pi(s.th[i]);
            update_all(&s);
            rebuild_grid(&s);
            printf("Loaded init positions from: %s\n", cfg.init_path);
        } else {
            printf("WARNING: could not parse init; falling back to grid init.\n");
            update_all(&s);
            rebuild_grid(&s);
        }
    } else {
        // mild randomization around grid layout to break symmetry
        double half = 0.5 * s.L;
        for (int i = 0; i < s.N; i++) {
            s.cx[i] += rng_uniform(&rng, -0.02 * half, 0.02 * half);
            s.cy[i] += rng_uniform(&rng, -0.02 * half, 0.02 * half);
            s.th[i]  = wrap_angle_0_2pi(rng_uniform(&rng, 0.0, 2.0 * M_PI));
        }
        update_all(&s);
        rebuild_grid(&s);
    }

    printf("Grid init: cols=%d rows=%d L_grid=%.6g -> start L=%.6g\n", cols, rows, L_grid, s.L);

    // ---- Two-phase schedule ----
    PhaseParams A;
    A.iters = 70000;
    A.T_start = 0.30;
    A.T_end   = 3e-4;

    A.step_xy_start = 1.0;
    A.step_th_start = 0.35;

    A.adapt_window = 1500;
    A.acc_low  = 0.25;
    A.acc_high = 0.55;
    A.step_shrink = 0.88;
    A.step_grow   = 1.12;
    A.step_xy_min = 1e-4;
    A.step_xy_max = 2.0;
    A.step_th_min = 1e-4;
    A.step_th_max = 1.0;

    A.lambda_start = 1e2;
    A.mu_start     = 1e2;
    A.ramp_every   = 0;
    A.ramp_factor  = 1.0;
    A.lambda_max   = 1e8;
    A.mu_max       = 1e8;

    A.p_reinsert = 0.02;
    A.p_rotmix   = 0.30;
    A.log_every  = A.iters / 6;

    PhaseParams B;
    B.iters = 35000;
    B.T_start = 0.06;
    B.T_end   = 1e-6;

    B.step_xy_start = 0.45;
    B.step_th_start = 0.18;

    B.adapt_window = 1500;
    B.acc_low  = 0.10;
    B.acc_high = 0.30;
    B.step_shrink = 0.90;
    B.step_grow   = 1.08;
    B.step_xy_min = 1e-5;
    B.step_xy_max = 1.0;
    B.step_th_min = 1e-5;
    B.step_th_max = 0.6;

    B.lambda_start = 1e3;
    B.mu_start     = 1e3;
    B.ramp_every   = 1200;
    B.ramp_factor  = 1.8;
    B.lambda_max   = 1e8;
    B.mu_max       = 1e8;

    B.p_reinsert = 0.005;
    B.p_rotmix   = 0.30;
    B.log_every  = B.iters / 6;

    // ---- Demo mode overrides ----
    int trials_bracket = 10;
    int trials_bisect  = 7;

    if (cfg.demo) {
        A.iters = 12000;
        B.iters = 6000;
        trials_bracket = 3;
        trials_bisect  = 2;
        A.log_every = A.iters / 4;
        B.log_every = B.iters / 4;
        printf("DEMO mode: A.iters=%d B.iters=%d trials_bracket=%d trials_bisect=%d\n",
               A.iters, B.iters, trials_bracket, trials_bisect);
    }

    // ---- Outer loop controls ----
    const double feas_tol = 1e-10;
    const int bracket_max_steps = 35;
    const int bisect_steps = 26;
    const double shrink_factor = 0.97;
    const double grow_factor   = 1.05;
    const double warm_safety   = 0.98;

    int N = cfg.N;
    double *best_cx = (double*)malloc((size_t)N * sizeof(double));
    double *best_cy = (double*)malloc((size_t)N * sizeof(double));
    double *best_th = (double*)malloc((size_t)N * sizeof(double));
    double *best2_cx = (double*)malloc((size_t)N * sizeof(double));
    double *best2_cy = (double*)malloc((size_t)N * sizeof(double));
    double *best2_th = (double*)malloc((size_t)N * sizeof(double));
    double *best_high_cx = (double*)malloc((size_t)N * sizeof(double));
    double *best_high_cy = (double*)malloc((size_t)N * sizeof(double));
    double *best_high_th = (double*)malloc((size_t)N * sizeof(double));

    if (!best_cx || !best_cy || !best_th || !best2_cx || !best2_cy || !best2_th ||
        !best_high_cx || !best_high_cy || !best_high_th)
    {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    printf("=== INITIAL PACK at L=%.6g ===\n", s.L);
    double feas0 = try_pack_at_current_L(&s, &rng, &A, &B, trials_bracket,
                                         best_cx, best_cy, best_th, 0);

    for (int i = 0; i < N; i++) { s.cx[i] = best_cx[i]; s.cy[i] = best_cy[i]; s.th[i] = best_th[i]; }
    update_all(&s);
    rebuild_grid(&s);

    int feas0_flag = is_feasible(feas0, feas_tol) && !guaranteed_infeasible_by_area(s.L, area_lb);

    printf("Initial result: L=%.12g feas=%.3e (%s)\n",
           s.L, feas0, feas0_flag ? "FEASIBLE" : "INFEASIBLE");

    double L_low = 0.0, L_high = 0.0;
    double L_curr = s.L;
    double best_feas_high = 1e300;

    // ---------------- Bracketing ----------------
    if (feas0_flag) {
        L_high = L_curr;
        best_feas_high = feas0;
        for (int i = 0; i < N; i++) { best_high_cx[i] = s.cx[i]; best_high_cy[i] = s.cy[i]; best_high_th[i] = s.th[i]; }

        int found_infeas = 0;
        for (int k = 0; k < bracket_max_steps; k++) {
            double L_new = L_curr * shrink_factor;

            if (L_new <= area_lb_infeas) {
                L_low = area_lb_infeas;
                found_infeas = 1;
                printf("\n=== BRACKET SHRINK: hit provable infeasible area bound. Set L_low=%.12g ===\n", L_low);
                break;
            }

            printf("\n=== BRACKET SHRINK: L=%.12g -> %.12g ===\n", L_curr, L_new);

            s.L = L_new;
            scale_positions_for_new_L(&s, L_curr, L_new, warm_safety);

            double feas = try_pack_at_current_L(&s, &rng, &A, &B, trials_bracket,
                                                best_cx, best_cy, best_th, 0);
            for (int i = 0; i < N; i++) { s.cx[i] = best_cx[i]; s.cy[i] = best_cy[i]; s.th[i] = best_th[i]; }
            update_all(&s);
            rebuild_grid(&s);

            int feasible_flag = is_feasible(feas, feas_tol) && !guaranteed_infeasible_by_area(s.L, area_lb);

            printf("Bracket check: L=%.12g feas=%.3e (%s)\n",
                   s.L, feas, feasible_flag ? "FEASIBLE" : "INFEASIBLE");

            if (feasible_flag) {
                L_high = s.L;
                best_feas_high = feas;
                for (int i = 0; i < N; i++) { best_high_cx[i] = s.cx[i]; best_high_cy[i] = s.cy[i]; best_high_th[i] = s.th[i]; }
                L_curr = s.L;
            } else {
                L_low = s.L;
                if (L_low > area_lb_infeas) L_low = area_lb_infeas;
                found_infeas = 1;
                break;
            }
        }

        if (!found_infeas) {
            L_low = area_lb_infeas;
            printf("\nWARNING: did not find infeasible by SA shrinking; using provably infeasible L_low=%.12g\n", L_low);
        }

    } else {
        L_low = area_lb_infeas;
        L_curr = s.L;

        int found_feas = 0;
        for (int k = 0; k < bracket_max_steps; k++) {
            double L_new = L_curr * grow_factor;
            if (L_new < area_lb * 1.01) L_new = area_lb * 1.01;

            printf("\n=== BRACKET GROW: L=%.12g -> %.12g ===\n", L_curr, L_new);

            s.L = L_new;
            scale_positions_for_new_L(&s, L_curr, L_new, warm_safety);

            double feas = try_pack_at_current_L(&s, &rng, &A, &B, trials_bracket,
                                                best_cx, best_cy, best_th, 0);
            for (int i = 0; i < N; i++) { s.cx[i] = best_cx[i]; s.cy[i] = best_cy[i]; s.th[i] = best_th[i]; }
            update_all(&s);
            rebuild_grid(&s);

            int feasible_flag = is_feasible(feas, feas_tol) && !guaranteed_infeasible_by_area(s.L, area_lb);

            printf("Bracket check: L=%.12g feas=%.3e (%s)\n",
                   s.L, feas, feasible_flag ? "FEASIBLE" : "INFEASIBLE");

            if (feasible_flag) {
                L_high = s.L;
                best_feas_high = feas;
                for (int i = 0; i < N; i++) { best_high_cx[i] = s.cx[i]; best_high_cy[i] = s.cy[i]; best_high_th[i] = s.th[i]; }
                found_feas = 1;
                break;
            } else {
                L_curr = s.L;
            }
        }
        if (!found_feas) {
            fprintf(stderr, "ERROR: could not find feasible L by growing.\n");
            state_free(&s);
            return 1;
        }
    }

    if (L_low >= L_high) {
        double new_high = fmax(L_high, area_lb * 1.01);
        printf("\nWARNING: L_low>=L_high after bracketing. Adjusting L_high to %.12g and retrying feasibility.\n", new_high);

        double old = s.L;
        s.L = new_high;
        scale_positions_for_new_L(&s, old, s.L, warm_safety);

        double feas = try_pack_at_current_L(&s, &rng, &A, &B, trials_bracket, best_cx, best_cy, best_th, 0);
        for (int i = 0; i < N; i++) { s.cx[i] = best_cx[i]; s.cy[i] = best_cy[i]; s.th[i] = best_th[i]; }
        update_all(&s);
        rebuild_grid(&s);

        int feasible_flag = is_feasible(feas, feas_tol) && !guaranteed_infeasible_by_area(s.L, area_lb);
        if (!feasible_flag) {
            fprintf(stderr, "ERROR: cannot establish a valid feasible upper bracket.\n");
            state_free(&s);
            return 1;
        }

        L_high = s.L;
        best_feas_high = feas;
        for (int i = 0; i < N; i++) { best_high_cx[i] = s.cx[i]; best_high_cy[i] = s.cy[i]; best_high_th[i] = s.th[i]; }

        L_low = area_lb_infeas;
    }

    printf("\n=== BRACKET FOUND ===\n");
    printf("L_low  (provably infeasible) = %.12g\n", L_low);
    printf("L_high (feasible)            = %.12g  (feas=%.3e)\n", L_high, best_feas_high);

    // ---------------- Bisection ----------------
    double L_prev_feas = L_high;

    s.L = L_high;
    for (int i = 0; i < N; i++) { s.cx[i] = best_high_cx[i]; s.cy[i] = best_high_cy[i]; s.th[i] = best_high_th[i]; }
    update_all(&s);
    rebuild_grid(&s);

    for (int it = 0; it < bisect_steps; it++) {
        double L_mid = 0.5 * (L_low + L_high);

        if (guaranteed_infeasible_by_area(L_mid, area_lb)) {
            L_low = L_mid;
            printf("\n=== BISECT %d/%d: mid=%.12g is provably infeasible by area; update L_low ===\n",
                   it + 1, bisect_steps, L_mid);
            continue;
        }

        printf("\n=== BISECT %d/%d: [%.12g, %.12g] mid=%.12g ===\n",
               it + 1, bisect_steps, L_low, L_high, L_mid);

        s.L = L_mid;
        scale_positions_for_new_L(&s, L_prev_feas, L_mid, warm_safety);

        double feas = try_pack_at_current_L(&s, &rng, &A, &B, trials_bisect,
                                            best2_cx, best2_cy, best2_th, 0);

        for (int i = 0; i < N; i++) { s.cx[i] = best2_cx[i]; s.cy[i] = best2_cy[i]; s.th[i] = best2_th[i]; }
        update_all(&s);
        rebuild_grid(&s);

        int feasible_flag = is_feasible(feas, feas_tol) && !guaranteed_infeasible_by_area(s.L, area_lb);

        printf("Mid result: L=%.12g feas=%.3e (%s)\n",
               s.L, feas, feasible_flag ? "FEASIBLE" : "INFEASIBLE");

        if (feasible_flag) {
            L_high = L_mid;
            L_prev_feas = L_mid;
            best_feas_high = feas;
            for (int i = 0; i < N; i++) { best_high_cx[i] = s.cx[i]; best_high_cy[i] = s.cy[i]; best_high_th[i] = s.th[i]; }
        } else {
            L_low = L_mid;

            s.L = L_high;
            for (int i = 0; i < N; i++) { s.cx[i] = best_high_cx[i]; s.cy[i] = best_high_cy[i]; s.th[i] = best_high_th[i]; }
            update_all(&s);
            rebuild_grid(&s);
        }
    }

    // Set final best from bisection
    s.L = L_high;
    for (int i = 0; i < N; i++) { s.cx[i] = best_high_cx[i]; s.cy[i] = best_high_cy[i]; s.th[i] = best_high_th[i]; }
    update_all(&s);
    rebuild_grid(&s);

    Totals final_tot = compute_totals_full_grid(&s);
    double final_feas = feasibility_metric(&final_tot);

    printf("\n=== AFTER BISECT (BEST FEASIBLE) ===\n");
    printf("L=%.12g ov=%.6e out=%.6e feas=%.6e (tol=%.1e)\n",
           s.L, final_tot.overlap_total, final_tot.out_total, final_feas, feas_tol);

    // Always write current best after bisection
    {
        char csv_path[256], svg_path[256];
        snprintf(csv_path, sizeof(csv_path), "csv/best_polys_N%03d.csv", cfg.N);
        snprintf(svg_path, sizeof(svg_path), "img/best_N%03d.svg", cfg.N);
        write_polys_csv(csv_path, &s, final_feas);
        write_best_svg(svg_path, &s, final_feas, 1100, 1100, 40.0);
        printf("Wrote best configuration to %s and %s\n", csv_path, svg_path);
    }

    // ---------------- Long-running shrink search (“leave running”) ----------------
    // Goal: keep trying to reduce L below current best, forever (or until time_limit).
    // Adaptive shrink step 'eps' adjusts based on success rate.
    double start_time = now_seconds();
    double last_ckpt_time = start_time;

    // store best feasible config
    double bestL = s.L;
    double bestFeas = final_feas;

    // A consistent “best so far” snapshot buffers
    double *best_pack_cx = (double*)malloc((size_t)N * sizeof(double));
    double *best_pack_cy = (double*)malloc((size_t)N * sizeof(double));
    double *best_pack_th = (double*)malloc((size_t)N * sizeof(double));
    if (!best_pack_cx || !best_pack_cy || !best_pack_th) { fprintf(stderr, "alloc failed\n"); return 1; }
    for (int i = 0; i < N; i++) { best_pack_cx[i] = s.cx[i]; best_pack_cy[i] = s.cy[i]; best_pack_th[i] = s.th[i]; }

    // local buffers for polish attempts
    double *tmp_cx = (double*)malloc((size_t)N * sizeof(double));
    double *tmp_cy = (double*)malloc((size_t)N * sizeof(double));
    double *tmp_th = (double*)malloc((size_t)N * sizeof(double));
    if (!tmp_cx || !tmp_cy || !tmp_th) { fprintf(stderr, "alloc failed\n"); return 1; }

    if (!cfg.polish) {
        printf("\nPolish disabled (--no_polish). Exiting after bisection.\n");
        free(best_cx); free(best_cy); free(best_th);
        free(best2_cx); free(best2_cy); free(best2_th);
        free(best_high_cx); free(best_high_cy); free(best_high_th);
        free(best_pack_cx); free(best_pack_cy); free(best_pack_th);
        free(tmp_cx); free(tmp_cy); free(tmp_th);
        state_free(&s);
        return 0;
    }

    printf("\n=== POLISH / LONG-RUN SHRINK SEARCH ===\n");
    printf("Running until killed%s.\n", (cfg.time_limit_sec > 0.0 ? " or time_limit" : ""));
    printf("Checkpoint every %.1f sec to csv/checkpoint_N%03d.csv and img/checkpoint_N%03d.svg\n",
           cfg.checkpoint_every_sec, cfg.N, cfg.N);

    double eps = fmin(fmax(5e-4, cfg.min_shrink), cfg.max_shrink); // initial fractional shrink
    int attempt = 0;
    int success_in_window = 0;
    int total_in_window = 0;
    const int WINDOW = 20;

    while (1) {
        double tnow = now_seconds();
        if (cfg.time_limit_sec > 0.0 && (tnow - start_time) >= cfg.time_limit_sec) {
            printf("\n[stop] time_limit reached (%.1f sec)\n", cfg.time_limit_sec);
            break;
        }

        // periodic checkpoint (best so far)
        if ((tnow - last_ckpt_time) >= cfg.checkpoint_every_sec) {
            // restore best state into s for writing
            s.L = bestL;
            for (int i = 0; i < N; i++) { s.cx[i] = best_pack_cx[i]; s.cy[i] = best_pack_cy[i]; s.th[i] = best_pack_th[i]; }
            update_all(&s);
            rebuild_grid(&s);

            char ccsv[256], csvg[256];
            snprintf(ccsv, sizeof(ccsv), "csv/checkpoint_N%03d.csv", cfg.N);
            snprintf(csvg, sizeof(csvg), "img/checkpoint_N%03d.svg", cfg.N);
            write_polys_csv(ccsv, &s, bestFeas);
            write_best_svg(csvg, &s, bestFeas, 1100, 1100, 40.0);

            printf("[ckpt] t=%.0fs bestL=%.12g bestFeas=%.3e eps=%.3g\n",
                   (tnow - start_time), bestL, bestFeas, eps);

            last_ckpt_time = tnow;
        }

        // do not go below area bound
        double L_try = bestL * (1.0 - eps);
        if (L_try <= area_lb * 1.000001) {
            // very close to necessary bound; keep searching with smaller steps
            eps *= 0.5;
            if (eps < cfg.min_shrink) eps = cfg.min_shrink;
            // If we are truly at the bound, this can run forever; that is intended.
            continue;
        }

        attempt++;

        // start from best and scale down into candidate L_try
        s.L = L_try;
        for (int i = 0; i < N; i++) { s.cx[i] = best_pack_cx[i]; s.cy[i] = best_pack_cy[i]; s.th[i] = best_pack_th[i]; }
        update_all(&s);
        rebuild_grid(&s);
        scale_positions_for_new_L(&s, bestL, L_try, warm_safety);

        // run SA at L_try
        double feas = try_pack_at_current_L(&s, &rng, &A, &B, cfg.trials_polish,
                                            tmp_cx, tmp_cy, tmp_th, 0);

        int feasible_flag = is_feasible(feas, feas_tol) && !guaranteed_infeasible_by_area(L_try, area_lb);

        total_in_window++;
        if (feasible_flag) success_in_window++;

        if (feasible_flag) {
            // accept improvement
            bestL = L_try;
            bestFeas = feas;
            for (int i = 0; i < N; i++) { best_pack_cx[i] = tmp_cx[i]; best_pack_cy[i] = tmp_cy[i]; best_pack_th[i] = tmp_th[i]; }

            // also persist “best” files immediately on improvement
            s.L = bestL;
            for (int i = 0; i < N; i++) { s.cx[i] = best_pack_cx[i]; s.cy[i] = best_pack_cy[i]; s.th[i] = best_pack_th[i]; }
            update_all(&s);
            rebuild_grid(&s);

            char bestcsv[256], bestsvg[256];
            snprintf(bestcsv, sizeof(bestcsv), "csv/best_polys_N%03d.csv", cfg.N);
            snprintf(bestsvg, sizeof(bestsvg), "img/best_N%03d.svg", cfg.N);
            write_polys_csv(bestcsv, &s, bestFeas);
            write_best_svg(bestsvg, &s, bestFeas, 1100, 1100, 40.0);

            printf("[improve] attempt=%d bestL=%.12g feas=%.3e eps=%.3g\n",
                   attempt, bestL, bestFeas, eps);
        }

        // adapt eps based on success rate in window
        if (total_in_window >= WINDOW) {
            double rate = (double)success_in_window / (double)total_in_window;

            // If too easy, increase eps (try larger jumps); if too hard, decrease eps.
            if (rate > cfg.target_success + 0.15) eps *= 1.25;
            else if (rate < cfg.target_success - 0.15) eps *= 0.80;
            else if (rate < cfg.target_success) eps *= 0.93;
            else eps *= 1.05;

            if (eps < cfg.min_shrink) eps = cfg.min_shrink;
            if (eps > cfg.max_shrink) eps = cfg.max_shrink;

            success_in_window = 0;
            total_in_window = 0;
        }
    }

    // Final write (best)
    s.L = bestL;
    for (int i = 0; i < N; i++) { s.cx[i] = best_pack_cx[i]; s.cy[i] = best_pack_cy[i]; s.th[i] = best_pack_th[i]; }
    update_all(&s);
    rebuild_grid(&s);

    Totals bt = compute_totals_full_grid(&s);
    double bfeas = feasibility_metric(&bt);

    printf("\n=== FINAL BEST (AFTER LONG RUN) ===\n");
    printf("bestL=%.12g ov=%.6e out=%.6e feas=%.6e\n", bestL, bt.overlap_total, bt.out_total, bfeas);

    {
        char bestcsv[256], bestsvg[256];
        snprintf(bestcsv, sizeof(bestcsv), "csv/best_polys_N%03d.csv", cfg.N);
        snprintf(bestsvg, sizeof(bestsvg), "img/best_N%03d.svg", cfg.N);
        write_polys_csv(bestcsv, &s, bfeas);
        write_best_svg(bestsvg, &s, bfeas, 1100, 1100, 40.0);
        printf("Wrote best configuration to %s and %s\n", bestcsv, bestsvg);
    }

    free(best_cx); free(best_cy); free(best_th);
    free(best2_cx); free(best2_cy); free(best2_th);
    free(best_high_cx); free(best_high_cy); free(best_high_th);

    free(best_pack_cx); free(best_pack_cy); free(best_pack_th);
    free(tmp_cx); free(tmp_cy); free(tmp_th);

    state_free(&s);
    return 0;
}
