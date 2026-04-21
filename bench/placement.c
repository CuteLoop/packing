/*
 * placement.c — Three placement regimes for the collision benchmark.
 *
 *  SPARSE  : grid layout, spacing = 3 × diameter.  Almost no AABB overlaps.
 *            Tests broadphase rejection efficiency.
 *
 *  DENSE   : random placement with minimum spacing ≈ 1.05 × diameter.
 *            Closest to the regime SA spends most of its time in.
 *
 *  JAMMED  : grid layout, spacing = 0.85 × diameter.  Forced overlaps.
 *            Tests worst-case narrowphase throughput.
 */

#include "bench.h"
#include <math.h>
#include <stdio.h>

/* ── helpers ─────────────────────────────────────────────────── */

/* Check whether a new pose overlaps any existing poses given min_dist */
static int overlaps_any(const Pose *poses, int placed,
                        double xn, double yn, double min_dist)
{
    double d2 = min_dist * min_dist;
    for (int i = 0; i < placed; i++) {
        double dx = xn - poses[i].x;
        double dy = yn - poses[i].y;
        if (dx*dx + dy*dy < d2) return 1;
    }
    return 0;
}

/* ── PLACE_SPARSE ────────────────────────────────────────────── */
/*
 * Axis-aligned grid, cell spacing = 3 × diameter.
 * Random rotation added to each instance.
 * Container L = cols × spacing, padded by one diameter.
 */
static void place_sparse(Pose *poses, int N, const PolyDef *p,
                         RNG *rng, double *L_out)
{
    double spacing = 3.0 * (2.0 * p->br);
    int cols = (int)ceil(sqrt((double)N));
    int rows = (N + cols - 1) / cols;

    double L = (cols < rows ? rows : cols) * spacing + p->br;
    if (L_out) *L_out = L;

    double ox = p->br * 0.5;
    for (int i = 0; i < N; i++) {
        int c = i % cols;
        int r = i / cols;
        poses[i].x     = ox + c * spacing + (rng_f64(rng) - 0.5) * 0.2 * spacing;
        poses[i].y     = ox + r * spacing + (rng_f64(rng) - 0.5) * 0.2 * spacing;
        poses[i].theta = rng_f64(rng) * 2.0 * M_PI;
    }
}

/* ── PLACE_DENSE ─────────────────────────────────────────────── */
/*
 * Random placement with rejection sampling: new polygon accepted only if
 * its centre is at least min_dist = 1.05 × 2 × diameter from all placed
 * polygon centres.  Container L chosen to give a feasible target density.
 *
 * For robustness, if max_tries is exceeded we fall back to grid placement
 * with 1.2 × diameter spacing (avoids infinite loops for large N).
 */
static void place_dense(Pose *poses, int N, const PolyDef *p,
                        RNG *rng, double *L_out)
{
    double min_dist = 1.05 * 2.0 * p->br;
    double L = 1.4 * sqrt((double)N) * 2.0 * p->br;
    if (L_out) *L_out = L;

    double margin = p->br;
    int placed = 0;
    int max_tries = 20000;

    while (placed < N) {
        int ok = 0;
        for (int t = 0; t < max_tries; t++) {
            double x = margin + rng_f64(rng) * (L - 2.0*margin);
            double y = margin + rng_f64(rng) * (L - 2.0*margin);
            if (!overlaps_any(poses, placed, x, y, min_dist)) {
                poses[placed].x     = x;
                poses[placed].y     = y;
                poses[placed].theta = rng_f64(rng) * 2.0 * M_PI;
                placed++;
                ok = 1;
                break;
            }
        }
        if (!ok) {
            /* Fallback: grid finish for remaining polygons */
            fprintf(stderr, "[placement] dense fallback to grid at N=%d/%d\n",
                    placed, N);
            double spacing = 1.2 * 2.0 * p->br;
            int cols = (int)ceil(sqrt((double)(N - placed))) + 1;
            for (int i = placed; i < N; i++) {
                int c = (i - placed) % cols;
                int r = (i - placed) / cols;
                poses[i].x     = margin + c * spacing;
                poses[i].y     = margin + r * spacing;
                poses[i].theta = rng_f64(rng) * 2.0 * M_PI;
            }
            break;
        }
    }
}

/* ── PLACE_JAMMED ────────────────────────────────────────────── */
/*
 * Grid placement with spacing = 0.85 × diameter (forced overlap).
 * Random small rotation per instance.
 */
static void place_jammed(Pose *poses, int N, const PolyDef *p,
                         RNG *rng, double *L_out)
{
    double spacing = 0.85 * 2.0 * p->br;
    int cols = (int)ceil(sqrt((double)N));
    int rows = (N + cols - 1) / cols;

    double L = (cols < rows ? rows : cols) * spacing + p->br;
    if (L_out) *L_out = L;

    double ox = p->br * 0.5;
    for (int i = 0; i < N; i++) {
        int c = i % cols;
        int r = i / cols;
        poses[i].x     = ox + c * spacing;
        poses[i].y     = ox + r * spacing;
        poses[i].theta = (rng_f64(rng) - 0.5) * 0.3; /* small random angle */
    }
}

/* ── public dispatch ─────────────────────────────────────────── */
void placement_generate(Pose *poses, int N, const PolyDef *p,
                        RNG *rng, PlaceMode mode, double *L_out)
{
    switch (mode) {
        case PLACE_SPARSE: place_sparse(poses, N, p, rng, L_out); break;
        case PLACE_DENSE:  place_dense (poses, N, p, rng, L_out); break;
        case PLACE_JAMMED: place_jammed(poses, N, p, rng, L_out); break;
    }
}
