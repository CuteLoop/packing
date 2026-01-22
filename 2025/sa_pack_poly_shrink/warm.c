// warm_starts.c
// ------------------------------------------------------------
// Generate several polygon warm-start configurations for the Kaggle
// Santa 2025 "ChristmasTree" polygon (non-convex), save CSVs,
// and render side-by-side SVG comparisons:
//
//   compare.svg         : raw warm starts
//   compare_center.svg  : after polygon-aware center compactification
//   compare_corner.svg  : after polygon-aware corner compactification
//
// Compactification is polygon-aware (no disk proxy):
//   - Fixed triangulation TRIS[] (NTRI=13)
//   - Triangle-triangle SAT overlap with MTV
//   - Iterative attractor + collision resolution
//
// Outputs:
//   outdir/n1/, outdir/n2/, ... outdir/nK/
//
// Build:
//   gcc -O2 -std=c11 -Wall -Wextra -pedantic warm_starts.c -o warm_starts -lm
//
// Run:
//   ./warm_starts --outdir warm_starts --max_n 10 --gap 0.02 --seed 42
//
// Updates in this version (warm starts beyond LP/ILP ideas):
//   - Sheared (brickwall) lattice warm start
//   - Diamond/rhombus lattice warm start (good for 180°-pair embeddings)
//   - Herringbone-like micro-motif warm start (0/90 with offsets)
//   - Sunflower (Fermat) spiral warm start (golden-angle)
//   - Core+spiral hybrid warm start (periodic-ish core + spiral boundary)
//
// Notes:
//   - These are deterministic generators intended to create strong initial
//     structures for subsequent compactification and SA/CMA-ES polishing.
//   - Angles remain fixed during compactification (by design).
// ------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// =============================================================
// Basic geometry types
// =============================================================
typedef struct { double x, y; } Pt;

typedef struct { int a,b,c; } Tri;

// Your fixed triangulation (vertex indices) for the tree polygon:
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

// =============================================================
// Tree polygon (unscaled) in its local coordinates.
// Matches the Kaggle polygon in your Python snippet, but UNITS are unscaled.
// Vertex order must match your triangulation indices above.
// =============================================================
static const Pt TREE_POLY[] = {
    { 0.0,   0.8 },       // 0 tip
    { 0.125, 0.5 },       // 1 top_w/2 at tier_1_y
    { 0.0625,0.5 },       // 2 top_w/4 at tier_1_y
    { 0.2,   0.25 },      // 3 mid_w/2 at tier_2_y
    { 0.1,   0.25 },      // 4 mid_w/4 at tier_2_y
    { 0.35,  0.0 },       // 5 base_w/2 at base_y
    { 0.075, 0.0 },       // 6 trunk_w/2 at base_y
    { 0.075,-0.2 },       // 7 trunk_w/2 at trunk_bottom_y
    { -0.075,-0.2 },      // 8 -trunk_w/2 at trunk_bottom_y
    { -0.075,0.0 },       // 9 -trunk_w/2 at base_y
    { -0.35, 0.0 },       // 10 -base_w/2 at base_y
    { -0.1,  0.25 },      // 11 -mid_w/4 at tier_2_y
    { -0.2,  0.25 },      // 12 -mid_w/2 at tier_2_y
    { -0.0625,0.5 },      // 13 -top_w/4 at tier_1_y
    { -0.125,0.5 },       // 14 -top_w/2 at tier_1_y
};
static const int TREE_M = (int)(sizeof(TREE_POLY)/sizeof(TREE_POLY[0]));

// =============================================================
// RNG (xorshift64* + splitmix seeding)
// =============================================================
typedef struct { uint64_t s; } RNG;

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static RNG rng_init(uint64_t seed) {
    uint64_t x = seed ? seed : 0x123456789abcdefULL;
    RNG r;
    r.s = splitmix64(&x);
    if (r.s == 0) r.s = 0xdeadbeefcafebabeULL;
    return r;
}

static uint64_t rng_u64(RNG *r) {
    // xorshift64*
    uint64_t x = r->s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->s = x;
    return x * 2685821657736338717ULL;
}

static double rng_u01(RNG *r) {
    // [0,1)
    return (rng_u64(r) >> 11) * (1.0/9007199254740992.0);
}

// =============================================================
// Warm-start state (centers + angles). No scaling.
// =============================================================
typedef struct {
    double *cx;
    double *cy;
    double *ang; // degrees
} WS;

static WS ws_alloc(int n) {
    WS w;
    w.cx = (double*)calloc((size_t)n, sizeof(double));
    w.cy = (double*)calloc((size_t)n, sizeof(double));
    w.ang = (double*)calloc((size_t)n, sizeof(double));
    if (!w.cx || !w.cy || !w.ang) {
        fprintf(stderr, "OOM ws_alloc\n");
        exit(1);
    }
    return w;
}

static WS ws_clone(const WS *src, int n) {
    WS w = ws_alloc(n);
    memcpy(w.cx, src->cx, (size_t)n*sizeof(double));
    memcpy(w.cy, src->cy, (size_t)n*sizeof(double));
    memcpy(w.ang, src->ang, (size_t)n*sizeof(double));
    return w;
}

static void ws_free(WS *w) {
    free(w->cx); free(w->cy); free(w->ang);
    w->cx = w->cy = w->ang = NULL;
}

static void ws_write_csv(const char *path, const WS *w, int n) {
    FILE *f = fopen(path, "w");
    if (!f) { perror("fopen csv"); exit(1); }
    fprintf(f, "i,x,y,deg\n");
    for (int i = 0; i < n; i++) {
        fprintf(f, "%d,%.12f,%.12f,%.12f\n", i, w->cx[i], w->cy[i], w->ang[i]);
    }
    fclose(f);
}

// =============================================================
// Filesystem helpers
// =============================================================
static int ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        if (S_ISDIR(st.st_mode)) return 1;
        fprintf(stderr, "ERROR: '%s' exists but is not a directory.\n", path);
        fprintf(stderr, "Delete/rename it, or pass a different --outdir.\n");
        return 0;
    }
    if (mkdir(path, 0775) != 0) {
        fprintf(stderr, "mkdir('%s') failed: %s\n", path, strerror(errno));
        return 0;
    }
    return 1;
}

// =============================================================
// Transform polygon by rotation (deg) and translation (cx,cy).
// =============================================================
static void poly_transform(const Pt *poly, int m, double cx, double cy, double deg, Pt *out) {
    double th = deg * (M_PI/180.0);
    double c = cos(th), s = sin(th);
    for (int i = 0; i < m; i++) {
        double x = poly[i].x;
        double y = poly[i].y;
        out[i].x = cx + (c*x - s*y);
        out[i].y = cy + (s*x + c*y);
    }
}

// =============================================================
// AABB helpers
// =============================================================
typedef struct { double minx, miny, maxx, maxy; } AABB;

static inline AABB tri_aabb(Pt a, Pt b, Pt c) {
    AABB B;
    B.minx = fmin(a.x, fmin(b.x, c.x));
    B.miny = fmin(a.y, fmin(b.y, c.y));
    B.maxx = fmax(a.x, fmax(b.x, c.x));
    B.maxy = fmax(a.y, fmax(b.y, c.y));
    return B;
}
static inline int aabb_overlap(AABB A, AABB B) {
    return !(A.maxx < B.minx || B.maxx < A.minx || A.maxy < B.miny || B.maxy < A.miny);
}

// =============================================================
// SAT triangle-triangle MTV
// =============================================================
static inline void axis_norm(double *x, double *y) {
    double n = sqrt((*x)*(*x) + (*y)*(*y));
    if (n < 1e-30) { *x = 1.0; *y = 0.0; return; }
    *x /= n; *y /= n;
}

static inline void proj_tri(Pt a, Pt b, Pt c, double ax, double ay, double *mn, double *mx) {
    double p1 = a.x*ax + a.y*ay;
    double p2 = b.x*ax + b.y*ay;
    double p3 = c.x*ax + c.y*ay;
    *mn = fmin(p1, fmin(p2, p3));
    *mx = fmax(p1, fmax(p2, p3));
}

static inline double overlap_1d(double a0, double a1, double b0, double b1) {
    double lo = fmax(a0, b0);
    double hi = fmin(a1, b1);
    return hi - lo; // <=0 => separated
}

// Returns 1 if overlapping and sets (mtvx,mtvy) to push A out of B.
static int sat_tri_tri_mtv(Pt A0, Pt A1, Pt A2, Pt B0, Pt B1, Pt B2,
                           double *mtvx, double *mtvy) {
    Pt As[3] = {A0,A1,A2};
    Pt Bs[3] = {B0,B1,B2};

    // Direction A->B to orient MTV consistently
    double Acx = (A0.x + A1.x + A2.x)/3.0, Acy = (A0.y + A1.y + A2.y)/3.0;
    double Bcx = (B0.x + B1.x + B2.x)/3.0, Bcy = (B0.y + B1.y + B2.y)/3.0;
    double dirx = Bcx - Acx, diry = Bcy - Acy;

    double best_ov = 1e300;
    double best_ax = 1.0, best_ay = 0.0;

    for (int which = 0; which < 2; which++) {
        Pt *P = (which==0) ? As : Bs;
        for (int e = 0; e < 3; e++) {
            Pt p0 = P[e], p1 = P[(e+1)%3];
            double ex = p1.x - p0.x, ey = p1.y - p0.y;
            double ax = -ey, ay = ex;
            axis_norm(&ax, &ay);

            double amin,amax,bmin,bmax;
            proj_tri(A0,A1,A2, ax,ay, &amin,&amax);
            proj_tri(B0,B1,B2, ax,ay, &bmin,&bmax);

            double ov = overlap_1d(amin,amax,bmin,bmax);
            if (ov <= 0.0) return 0;

            if (ov < best_ov) {
                best_ov = ov;
                best_ax = ax; best_ay = ay;

                // orient axis so it points from A toward B
                double d = dirx*best_ax + diry*best_ay;
                if (d < 0) { best_ax = -best_ax; best_ay = -best_ay; }
            }
        }
    }

    // Push A away from B
    *mtvx = -best_ax * best_ov;
    *mtvy = -best_ay * best_ov;
    return 1;
}

// Polygon overlap MTV using TRIS (accumulate from overlapping tri pairs)
static int poly_mtv_using_TRIS(const Pt *A, const Pt *B, double *dx, double *dy) {
    double sx = 0.0, sy = 0.0;
    int hit = 0;

    for (int i = 0; i < NTRI; i++) {
        Tri ta = TRIS[i];
        Pt A0 = A[ta.a], A1 = A[ta.b], A2 = A[ta.c];
        AABB aabA = tri_aabb(A0,A1,A2);

        for (int j = 0; j < NTRI; j++) {
            Tri tb = TRIS[j];
            Pt B0 = B[tb.a], B1 = B[tb.b], B2 = B[tb.c];
            AABB aabB = tri_aabb(B0,B1,B2);

            if (!aabb_overlap(aabA, aabB)) continue;

            double mx,my;
            if (sat_tri_tri_mtv(A0,A1,A2, B0,B1,B2, &mx,&my)) {
                sx += mx; sy += my;
                hit = 1;
            }
        }
    }

    if (!hit) return 0;

    // Cap MTV magnitude to avoid oscillations when multiple tri pairs overlap
    double mag = sqrt(sx*sx + sy*sy);
    if (mag > 1e-30) {
        const double cap = 0.25; // tune
        if (mag > cap) { sx *= cap/mag; sy *= cap/mag; }
    }

    *dx = sx; *dy = sy;
    return 1;
}

// =============================================================
// Bounding square for a WS (based on polygon bounds)
// =============================================================
static void ws_bounds(const WS *w, int n, double *minx, double *miny, double *maxx, double *maxy) {
    *minx =  1e300; *miny =  1e300;
    *maxx = -1e300; *maxy = -1e300;
    Pt verts[TREE_M];

    for (int i = 0; i < n; i++) {
        poly_transform(TREE_POLY, TREE_M, w->cx[i], w->cy[i], w->ang[i], verts);
        for (int k = 0; k < TREE_M; k++) {
            *minx = fmin(*minx, verts[k].x);
            *miny = fmin(*miny, verts[k].y);
            *maxx = fmax(*maxx, verts[k].x);
            *maxy = fmax(*maxy, verts[k].y);
        }
    }
}

static void ws_bsquare(const WS *w, int n, double *x0, double *y0, double *side) {
    double minx,miny,maxx,maxy;
    ws_bounds(w,n,&minx,&miny,&maxx,&maxy);
    double wdx = maxx - minx;
    double wdy = maxy - miny;
    *side = fmax(wdx, wdy);
    double cx = 0.5*(minx + maxx);
    double cy = 0.5*(miny + maxy);
    *x0 = cx - 0.5*(*side);
    *y0 = cy - 0.5*(*side);
}

static void ws_center_mass(const WS *w, int n, double *mx, double *my) {
    double sx = 0.0, sy = 0.0;
    for (int i = 0; i < n; i++) { sx += w->cx[i]; sy += w->cy[i]; }
    *mx = (n>0) ? sx/(double)n : 0.0;
    *my = (n>0) ? sy/(double)n : 0.0;
}

// =============================================================
// Polygon-aware compactification routines
//   1) Center attractor: target = (0,0)
//   2) Corner attractor: target = lower-left corner of current bounding square
// Both are: attractor step + collision resolution passes using MTV.
// Angles remain fixed.
// =============================================================
static WS compactify_attractor(const WS *src, int n, uint64_t seed,
                               int use_corner_target) {
    WS w = ws_clone(src, n);
    RNG rng = rng_init(seed);

    // Parameters (tune as needed)
    const int OUTER_ITERS = 260;
    const int RESOLVE_PASSES = 22;
    const double k_attract = 0.035;
    const double dt0 = 0.30;
    const double jitter = 1e-7;
    const double mtv_split = 0.5;

    Pt *V = (Pt*)malloc((size_t)n * (size_t)TREE_M * sizeof(Pt));
    if (!V) { fprintf(stderr, "OOM compactify verts\n"); exit(1); }

    for (int it = 0; it < OUTER_ITERS; it++) {
        double dt = dt0 * (1.0 - 0.70 * ((double)it / (double)OUTER_ITERS));

        // choose target
        double tx = 0.0, ty = 0.0;
        if (use_corner_target) {
            // shove into lower-left corner of current bounding square
            double x0,y0,side;
            ws_bsquare(&w, n, &x0,&y0,&side);
            tx = x0;
            ty = y0;
        }

        // 1) attractor move
        for (int i = 0; i < n; i++) {
            double dx = (tx - w.cx[i]);
            double dy = (ty - w.cy[i]);

            w.cx[i] += dt * (k_attract * dx);
            w.cy[i] += dt * (k_attract * dy);

            w.cx[i] += (rng_u01(&rng) - 0.5) * jitter;
            w.cy[i] += (rng_u01(&rng) - 0.5) * jitter;
        }

        // 2) collision resolution
        for (int pass = 0; pass < RESOLVE_PASSES; pass++) {
            for (int i = 0; i < n; i++) {
                poly_transform(TREE_POLY, TREE_M, w.cx[i], w.cy[i], w.ang[i], &V[i*TREE_M]);
            }

            int any = 0;
            for (int i = 0; i < n; i++) {
                for (int j = i+1; j < n; j++) {
                    double mx,my;
                    if (poly_mtv_using_TRIS(&V[i*TREE_M], &V[j*TREE_M], &mx,&my)) {
                        any = 1;
                        w.cx[i] += mtv_split * mx;
                        w.cy[i] += mtv_split * my;
                        w.cx[j] -= mtv_split * mx;
                        w.cy[j] -= mtv_split * my;
                    }
                }
            }
            if (!any) break;
        }
    }

    free(V);
    return w;
}

static WS compactify_center(const WS *src, int n, uint64_t seed) {
    return compactify_attractor(src, n, seed, 0);
}
static WS compactify_corner(const WS *src, int n, uint64_t seed) {
    return compactify_attractor(src, n, seed, 1);
}

// =============================================================
// Warm start strategies (unscaled)
// =============================================================

static WS ws_centered_single(int n) {
    WS w = ws_alloc(n);
    for (int i = 0; i < n; i++) { w.cx[i]=0; w.cy[i]=0; w.ang[i]=0; }
    return w;
}

// near-square grid using cols/rows
static void grid_dims(int n, int *rows, int *cols) {
    int c = (int)ceil(sqrt((double)n));
    int r = (int)ceil((double)n / (double)c);
    *rows = r; *cols = c;
}

// Conservative spacings derived from your polygon rough AABB:
// x span ~0.70, y span ~1.00
static void base_spacings(double gap, double *dx, double *dy) {
    *dx = 0.70 + gap;
    *dy = 1.00 + gap;
}

// Strategy 1: axis-aligned grid, angle=0 for all
static WS ws_grid(int n, double gap) {
    WS w = ws_alloc(n);
    int rows, cols; grid_dims(n, &rows, &cols);

    double dx,dy; base_spacings(gap,&dx,&dy);

    double x0 = -0.5*(cols-1)*dx;
    double y0 = -0.5*(rows-1)*dy;

    for (int i = 0; i < n; i++) {
        int rr = i / cols;
        int cc = i % cols;
        w.cx[i] = x0 + cc*dx;
        w.cy[i] = y0 + rr*dy;
        w.ang[i] = 0.0;
    }
    return w;
}

// Strategy 2: alternating grid, angles along ±45 degrees (kept as-is)
static WS ws_grid_alt45(int n, double gap) {
    WS w = ws_alloc(n);
    int rows, cols; grid_dims(n, &rows, &cols);

    double dx = 0.80 + gap;
    double dy = 1.10 + gap;

    double x0 = -0.5*(cols-1)*dx;
    double y0 = -0.5*(rows-1)*dy;

    for (int i = 0; i < n; i++) {
        int rr = i / cols;
        int cc = i % cols;
        w.cx[i] = x0 + cc*dx;
        w.cy[i] = y0 + rr*dy;

        int parity = (rr + cc) & 1;
        w.ang[i] = parity ? 45.0 : -45.0;
    }
    return w;
}

// Strategy 3: normal alternating grid, angles = 0 and 180.
static WS ws_grid_alt180(int n, double gap) {
    WS w = ws_alloc(n);
    int rows, cols; grid_dims(n, &rows, &cols);

    double dx,dy; base_spacings(gap,&dx,&dy);

    double x0 = -0.5*(cols-1)*dx;
    double y0 = -0.5*(rows-1)*dy;

    for (int i = 0; i < n; i++) {
        int rr = i / cols;
        int cc = i % cols;
        w.cx[i] = x0 + cc*dx;
        w.cy[i] = y0 + rr*dy;

        int parity = (rr + cc) & 1;
        w.ang[i] = parity ? 180.0 : 0.0;
    }
    return w;
}

// Strategy 4: gridlike alternating, angles = 45 and 225 (45+180).
static WS ws_alternating(int n, double gap) {
    WS w = ws_alloc(n);
    int rows, cols; grid_dims(n, &rows, &cols);

    double dx = 0.80 + gap;
    double dy = 1.10 + gap;

    double x0 = -0.5*(cols-1)*dx;
    double y0 = -0.5*(rows-1)*dy;

    for (int i = 0; i < n; i++) {
        int rr = i / cols;
        int cc = i % cols;
        w.cx[i] = x0 + cc*dx;
        w.cy[i] = y0 + rr*dy;

        int parity = (rr + cc) & 1;
        w.ang[i] = parity ? 225.0 : 45.0;
    }
    return w;
}

// -------------------------------------------------------------
// NEW Strategy A: sheared (brickwall) lattice
// v1=(dx,0), v2=(shear,dy) with optional half-shift every other row.
// Useful when "grid all same direction" produces very compact patches.
// -------------------------------------------------------------
static WS ws_sheared_brick(int n, double gap, uint64_t seed) {
    (void)seed;
    WS w = ws_alloc(n);
    int rows, cols; grid_dims(n, &rows, &cols);

    double dx,dy; base_spacings(gap,&dx,&dy);

    // Shear magnitude: a modest fraction of dx.
    // This creates a parallelogram lattice (two-vector tiling family).
    double shear = 0.45 * dx;

    double x0 = -0.5*(cols-1)*dx;
    double y0 = -0.5*(rows-1)*dy;

    for (int i = 0; i < n; i++) {
        int rr = i / cols;
        int cc = i % cols;

        // Lattice embedding:
        // position = cc*v1 + rr*v2, plus "brick" half-shift on odd rows.
        double x = x0 + cc*dx + rr*shear;
        double y = y0 + rr*dy;

        if (rr & 1) x += 0.5*dx;

        w.cx[i] = x;
        w.cy[i] = y;

        // Keep orientations simple: alternate 0/180 to encourage dimer-like pairing.
        w.ang[i] = ((rr + cc) & 1) ? 180.0 : 0.0;
    }
    return w;
}

// -------------------------------------------------------------
// NEW Strategy B: diamond / rhombus lattice (two-vector family)
// v1 and v2 symmetric about x-axis (a "diamond grid").
// Intended for embeddings where pairs look 180° symmetric.
// -------------------------------------------------------------
static WS ws_diamond_lattice(int n, double gap, uint64_t seed) {
    (void)seed;
    WS w = ws_alloc(n);
    int rows, cols; grid_dims(n, &rows, &cols);

    // Choose a diamond basis length based on conservative footprint.
    // Smaller than dy to promote denser initial placement; compactify will resolve.
    double dx0,dy0; base_spacings(gap,&dx0,&dy0);
    double r = 0.60 + 0.5*gap;     // base edge length of rhombus steps (tunable)
    double phi = 55.0 * (M_PI/180.0); // diamond angle (tunable)

    Pt v1 = { r*cos(phi),  r*sin(phi) };
    Pt v2 = { r*cos(phi), -r*sin(phi) };

    // Center indices around origin
    double i0 = -0.5*(cols-1);
    double j0 = -0.5*(rows-1);

    for (int k = 0; k < n; k++) {
        int rr = k / cols;
        int cc = k % cols;
        double i = i0 + (double)cc;
        double j = j0 + (double)rr;

        double x = i*v1.x + j*v2.x;
        double y = i*v1.y + j*v2.y;

        w.cx[k] = x;
        w.cy[k] = y;

        // Orientation: alternating 45/225 tends to create interlocking "diamond dimers".
        w.ang[k] = ((rr + cc) & 1) ? 225.0 : 45.0;
    }

    (void)dx0; (void)dy0; // keep for future tuning without warnings
    return w;
}

// -------------------------------------------------------------
// NEW Strategy C: herringbone-like micro-motif
// Two-row motif: horizontal-ish rows (0°) alternating with vertical-ish (90°),
// with half-cell shifts to emulate a weave/herringbone macro-pattern.
// -------------------------------------------------------------
static WS ws_herringbone(int n, double gap, uint64_t seed) {
    (void)seed;
    WS w = ws_alloc(n);
    int rows, cols; grid_dims(n, &rows, &cols);

    // Use slightly larger spacing because 90° rotations change AABB.
    double dx = 0.80 + gap;
    double dy = 0.95 + gap;

    double x0 = -0.5*(cols-1)*dx;
    double y0 = -0.5*(rows-1)*dy;

    for (int i = 0; i < n; i++) {
        int rr = i / cols;
        int cc = i % cols;

        double x = x0 + cc*dx;
        double y = y0 + rr*dy;

        // shift every other row and also every other column lightly to break alignment
        if (rr & 1) x += 0.5*dx;
        if (cc & 1) y += 0.08*dy;

        w.cx[i] = x;
        w.cy[i] = y;

        // orientation bands
        w.ang[i] = (rr & 1) ? 90.0 : 0.0;
    }
    return w;
}

// Strategy 5: circle ring, angles radial-ish
static WS ws_circle_ring(int n, double gap, uint64_t seed) {
    (void)seed;
    WS w = ws_alloc(n);
    if (n == 1) { w.cx[0]=0; w.cy[0]=0; w.ang[0]=0; return w; }

    double R = 0.65*(double)n/(2.0*M_PI) + 0.65 + gap; // loose but fine for warm start
    for (int i = 0; i < n; i++) {
        double a = (2.0*M_PI*i)/(double)n;
        w.cx[i] = R*cos(a);
        w.cy[i] = R*sin(a);
        w.ang[i] = a*(180.0/M_PI); // roughly radial
    }
    return w;
}

// Strategy 6: spiral, angles follow spiral direction
static WS ws_spiral(int n, double gap, uint64_t seed) {
    (void)seed;
    WS w = ws_alloc(n);
    double a_step = 0.85; // radians
    for (int i = 0; i < n; i++) {
        double a = i*a_step;
        double r = (0.35 + gap) * sqrt((double)i);
        w.cx[i] = r*cos(a);
        w.cy[i] = r*sin(a);
        w.ang[i] = fmod(a*(180.0/M_PI), 360.0);
    }
    return w;
}

// -------------------------------------------------------------
// NEW Strategy D: sunflower / Fermat spiral (golden-angle)
// r_i = c*sqrt(i), theta_i = i*alpha, alpha = golden angle.
// Orientation tangential tends to reduce early tip collisions.
// -------------------------------------------------------------
static WS ws_sunflower(int n, double gap, uint64_t seed) {
    (void)seed;
    WS w = ws_alloc(n);

    const double golden = M_PI*(3.0 - sqrt(5.0)); // ~2.399963...
    const double c = 0.42 + 0.5*gap;              // radial scale (tune)
    for (int i = 0; i < n; i++) {
        double th = (double)i * golden;
        double r  = c * sqrt((double)i);
        w.cx[i] = r*cos(th);
        w.cy[i] = r*sin(th);

        // tangential orientation
        double deg = (th + M_PI/2.0) * (180.0/M_PI);
        w.ang[i] = fmod(deg, 360.0);
    }
    return w;
}

// Strategy 7: random in box, fixed angles 0/90 alternating
static WS ws_random_box(int n, double gap, uint64_t seed) {
    WS w = ws_alloc(n);
    RNG rng = rng_init(seed);

    // generous box
    double B = (double)n * (0.25 + gap) + 0.5;
    for (int i = 0; i < n; i++) {
        w.cx[i] = (rng_u01(&rng)*2.0 - 1.0) * B;
        w.cy[i] = (rng_u01(&rng)*2.0 - 1.0) * B;
        w.ang[i] = (i & 1) ? 90.0 : 0.0;
    }
    return w;
}

// -------------------------------------------------------------
// NEW Strategy E: core (periodic-ish) + spiral boundary
// Build a sheared-brick core for ~60% of items, then place remainder
// on a sunflower spiral outside the core radius.
// -------------------------------------------------------------
static WS ws_core_plus_spiral(int n, double gap, uint64_t seed) {
    WS w = ws_alloc(n);
    if (n <= 2) {
        // trivial fallback
        WS t = ws_sunflower(n, gap, seed);
        memcpy(w.cx, t.cx, (size_t)n*sizeof(double));
        memcpy(w.cy, t.cy, (size_t)n*sizeof(double));
        memcpy(w.ang, t.ang, (size_t)n*sizeof(double));
        ws_free(&t);
        return w;
    }

    int n_core = (int)floor(0.60 * (double)n);
    if (n_core < 1) n_core = 1;
    if (n_core > n-1) n_core = n-1;

    WS core = ws_sheared_brick(n_core, gap, seed + 11ULL);

    // Copy core into w
    for (int i = 0; i < n_core; i++) {
        w.cx[i]  = core.cx[i];
        w.cy[i]  = core.cy[i];
        w.ang[i] = core.ang[i];
    }

    // Estimate a radius to start spiral outside the core.
    double mx,my; ws_center_mass(&core, n_core, &mx,&my);
    double rmax = 0.0;
    for (int i = 0; i < n_core; i++) {
        double dx = core.cx[i] - mx;
        double dy = core.cy[i] - my;
        double r = sqrt(dx*dx + dy*dy);
        if (r > rmax) rmax = r;
    }
    double r0 = rmax + (0.8 + 2.0*gap);

    // Place remaining points on a sunflower spiral, but shifted outward.
    const double golden = M_PI*(3.0 - sqrt(5.0));
    const double c = 0.38 + 0.5*gap;

    for (int i = n_core; i < n; i++) {
        int t = i - n_core;
        double th = (double)(t) * golden;
        double r  = r0 + c * sqrt((double)t);
        w.cx[i] = mx + r*cos(th);
        w.cy[i] = my + r*sin(th);

        // tangential orientation
        double deg = (th + M_PI/2.0) * (180.0/M_PI);
        w.ang[i] = fmod(deg, 360.0);
    }

    ws_free(&core);
    return w;
}

// =============================================================
// SVG rendering
// =============================================================
static void compute_global_bounds(const WS *arr, int S, int n,
                                  double *minx, double *miny, double *maxx, double *maxy) {
    *minx =  1e300; *miny =  1e300;
    *maxx = -1e300; *maxy = -1e300;

    Pt verts[TREE_M];
    for (int s = 0; s < S; s++) {
        for (int i = 0; i < n; i++) {
            poly_transform(TREE_POLY, TREE_M, arr[s].cx[i], arr[s].cy[i], arr[s].ang[i], verts);
            for (int k = 0; k < TREE_M; k++) {
                *minx = fmin(*minx, verts[k].x);
                *miny = fmin(*miny, verts[k].y);
                *maxx = fmax(*maxx, verts[k].x);
                *maxy = fmax(*maxy, verts[k].y);
            }
        }
    }

    // add margin
    double dx = (*maxx - *minx);
    double dy = (*maxy - *miny);
    double pad = 0.08 * fmax(dx, dy) + 0.15;
    *minx -= pad; *miny -= pad;
    *maxx += pad; *maxy += pad;
}

static void svg_draw_poly(FILE *f, const Pt *poly, int m) {
    fprintf(f, "<path d=\"M ");
    for (int i = 0; i < m; i++) {
        fprintf(f, "%.9f %.9f ", poly[i].x, poly[i].y);
    }
    fprintf(f, "Z\" fill=\"none\" stroke=\"#222\" stroke-width=\"0.01\" />\n");
}

static void svg_draw_bsquare(FILE *f, const WS *w, int n) {
    double x0,y0,side;
    ws_bsquare(w,n,&x0,&y0,&side);
    fprintf(f,
        "<rect x=\"%.9f\" y=\"%.9f\" width=\"%.9f\" height=\"%.9f\" "
        "fill=\"none\" stroke=\"#000\" stroke-width=\"0.01\" />\n",
        x0, y0, side, side
    );
}

static void write_compare_svg(const char *path,
                              const char **names, const WS *arr, int S, int n,
                              const char *title) {
    // Layout: 3 columns, enough rows
    int cols = 3;
    int rows = (int)ceil((double)S / (double)cols);

    const int PW = 420;
    const int PH = 420;
    const int MARGIN = 40;

    int W = cols*PW + (cols+1)*MARGIN;
    int H = rows*PH + (rows+1)*MARGIN + 60;

    double gminx,gminy,gmaxx,gmaxy;
    compute_global_bounds(arr, S, n, &gminx,&gminy,&gmaxx,&gmaxy);
    double gdx = gmaxx - gminx;
    double gdy = gmaxy - gminy;

    FILE *f = fopen(path, "w");
    if (!f) { perror("fopen svg"); exit(1); }

    fprintf(f, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\">\n", W,H,W,H);
    fprintf(f, "<rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" fill=\"white\" />\n", W,H);
    fprintf(f, "<text x=\"%d\" y=\"30\" font-family=\"monospace\" font-size=\"16\">%s (n=%d)</text>\n", MARGIN, title, n);

    for (int s = 0; s < S; s++) {
        int rr = s / cols;
        int cc = s % cols;
        int px = MARGIN + cc*(PW + MARGIN);
        int py = 60 + MARGIN + rr*(PH + MARGIN);

        double sx = (double)PW / gdx;
        double sy = (double)PH / gdy;
        double sc = fmin(sx, sy);

        double tx = (double)px;
        double ty = (double)(py + PH); // because we flip y

        fprintf(f, "<g transform=\"translate(%.3f %.3f) scale(%.9f %.9f) translate(%.9f %.9f)\">\n",
                tx, ty, sc, -sc, -gminx, -gminy);

        Pt verts[TREE_M];
        for (int i = 0; i < n; i++) {
            poly_transform(TREE_POLY, TREE_M, arr[s].cx[i], arr[s].cy[i], arr[s].ang[i], verts);
            svg_draw_poly(f, verts, TREE_M);
        }
        svg_draw_bsquare(f, &arr[s], n);

        fprintf(f, "</g>\n");

        fprintf(f, "<text x=\"%d\" y=\"%d\" font-family=\"monospace\" font-size=\"12\">%s</text>\n",
                px, py-8, names[s]);
    }

    fprintf(f, "</svg>\n");
    fclose(f);
}

// =============================================================
// Strategies registry
// =============================================================
typedef WS (*ws_fn)(int n, double gap, uint64_t seed);

static WS wrap_centered(int n, double gap, uint64_t seed) { (void)gap; (void)seed; return ws_centered_single(n); }
static WS wrap_grid(int n, double gap, uint64_t seed) { (void)seed; return ws_grid(n, gap); }
static WS wrap_grid_alt180(int n, double gap, uint64_t seed) { (void)seed; return ws_grid_alt180(n, gap); }
static WS wrap_grid_alt45(int n, double gap, uint64_t seed) { (void)seed; return ws_grid_alt45(n, gap); }
static WS wrap_alternating(int n, double gap, uint64_t seed) { (void)seed; return ws_alternating(n, gap); }
static WS wrap_sheared_brick(int n, double gap, uint64_t seed) { return ws_sheared_brick(n, gap, seed); }
static WS wrap_diamond(int n, double gap, uint64_t seed) { return ws_diamond_lattice(n, gap, seed); }
static WS wrap_herringbone(int n, double gap, uint64_t seed) { return ws_herringbone(n, gap, seed); }
static WS wrap_circle(int n, double gap, uint64_t seed) { (void)seed; return ws_circle_ring(n, gap, 0); }
static WS wrap_spiral(int n, double gap, uint64_t seed) { (void)seed; return ws_spiral(n, gap, 0); }
static WS wrap_sunflower(int n, double gap, uint64_t seed) { return ws_sunflower(n, gap, seed); }
static WS wrap_core_spiral(int n, double gap, uint64_t seed) { return ws_core_plus_spiral(n, gap, seed); }
static WS wrap_random(int n, double gap, uint64_t seed) { return ws_random_box(n, gap, seed); }

// =============================================================
// CLI / main
// =============================================================
typedef struct {
    const char *outdir;
    int max_n;
    double gap;
    uint64_t seed;
} Config;

static Config parse_args(int argc, char **argv) {
    Config c;
    c.outdir = "warm_starts";
    c.max_n = 10;
    c.gap = 0.02;
    c.seed = 42;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--outdir") && i+1 < argc) { c.outdir = argv[++i]; }
        else if (!strcmp(argv[i], "--max_n") && i+1 < argc) { c.max_n = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--gap") && i+1 < argc) { c.gap = atof(argv[++i]); }
        else if (!strcmp(argv[i], "--seed") && i+1 < argc) { c.seed = (uint64_t)strtoull(argv[++i], NULL, 10); }
        else {
            fprintf(stderr, "Unknown/invalid arg: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [--outdir DIR] [--max_n 10] [--gap 0.02] [--seed 42]\n", argv[0]);
            exit(1);
        }
    }
    if (c.max_n < 1) c.max_n = 1;
    if (c.max_n > 2000) c.max_n = 2000;
    if (c.gap < 0) c.gap = 0;
    return c;
}

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    if (!ensure_dir(cfg.outdir)) return 1;

    // Strategies (add/remove as desired)
    const char *names[] = {
        "centered",
        "grid",
        "grid_alt180",
        "grid_alt45",
        "alternating",
        "sheared_brick",
        "diamond_lattice",
        "herringbone",
        "circle_ring",
        "spiral",
        "sunflower",
        "core_plus_spiral",
        "random_box"
    };
    ws_fn fns[] = {
        wrap_centered,
        wrap_grid,
        wrap_grid_alt180,
        wrap_grid_alt45,
        wrap_alternating,
        wrap_sheared_brick,
        wrap_diamond,
        wrap_herringbone,
        wrap_circle,
        wrap_spiral,
        wrap_sunflower,
        wrap_core_spiral,
        wrap_random
    };
    const int S = (int)(sizeof(fns)/sizeof(fns[0]));

    for (int n = 1; n <= cfg.max_n; n++) {
        char ndir[4096];
        int rc = snprintf(ndir, sizeof(ndir), "%s/n%d", cfg.outdir, n);
        if (rc < 0 || rc >= (int)sizeof(ndir)) { fprintf(stderr, "ndir too long\n"); return 1; }
        if (!ensure_dir(ndir)) return 1;

        WS *raw = (WS*)calloc((size_t)S, sizeof(WS));
        WS *cen = (WS*)calloc((size_t)S, sizeof(WS));
        WS *cor = (WS*)calloc((size_t)S, sizeof(WS));
        if (!raw || !cen || !cor) { fprintf(stderr, "OOM arrays\n"); return 1; }

        for (int s = 0; s < S; s++) {
            raw[s] = fns[s](n, cfg.gap, cfg.seed + 1000ULL*(uint64_t)n + 17ULL*(uint64_t)s);

            cen[s] = compactify_center(&raw[s], n, cfg.seed + 90000ULL + 1000ULL*(uint64_t)n + 23ULL*(uint64_t)s);
            cor[s] = compactify_corner(&raw[s], n, cfg.seed + 190000ULL + 1000ULL*(uint64_t)n + 29ULL*(uint64_t)s);

            char csvp[4096];
            rc = snprintf(csvp, sizeof(csvp), "%s/%s.csv", ndir, names[s]);
            if (rc < 0 || rc >= (int)sizeof(csvp)) { fprintf(stderr, "csv path too long\n"); return 1; }
            ws_write_csv(csvp, &raw[s], n);

            rc = snprintf(csvp, sizeof(csvp), "%s/%s_center.csv", ndir, names[s]);
            if (rc < 0 || rc >= (int)sizeof(csvp)) { fprintf(stderr, "csv path too long\n"); return 1; }
            ws_write_csv(csvp, &cen[s], n);

            rc = snprintf(csvp, sizeof(csvp), "%s/%s_corner.csv", ndir, names[s]);
            if (rc < 0 || rc >= (int)sizeof(csvp)) { fprintf(stderr, "csv path too long\n"); return 1; }
            ws_write_csv(csvp, &cor[s], n);
        }

        // SVG comparisons
        char svgp[4096];

        rc = snprintf(svgp, sizeof(svgp), "%s/compare.svg", ndir);
        if (rc < 0 || rc >= (int)sizeof(svgp)) { fprintf(stderr, "svg path too long\n"); return 1; }
        write_compare_svg(svgp, names, raw, S, n, "Warm starts (raw)");

        rc = snprintf(svgp, sizeof(svgp), "%s/compare_center.svg", ndir);
        if (rc < 0 || rc >= (int)sizeof(svgp)) { fprintf(stderr, "svg path too long\n"); return 1; }
        write_compare_svg(svgp, names, cen, S, n, "After compactification (center attractor)");

        rc = snprintf(svgp, sizeof(svgp), "%s/compare_corner.svg", ndir);
        if (rc < 0 || rc >= (int)sizeof(svgp)) { fprintf(stderr, "svg path too long\n"); return 1; }
        write_compare_svg(svgp, names, cor, S, n, "After compactification (corner shove)");

        for (int s = 0; s < S; s++) {
            ws_free(&raw[s]);
            ws_free(&cen[s]);
            ws_free(&cor[s]);
        }
        free(raw); free(cen); free(cor);

        fprintf(stdout, "[ok] n=%d wrote CSVs + SVGs into %s/\n", n, ndir);
    }

    fprintf(stdout, "\nDone.\n");
    fprintf(stdout, "Open e.g.: %s/n10/compare.svg\n", cfg.outdir);
    fprintf(stdout, "          %s/n10/compare_center.svg\n", cfg.outdir);
    fprintf(stdout, "          %s/n10/compare_corner.svg\n", cfg.outdir);
    return 0;
}
