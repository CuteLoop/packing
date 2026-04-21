/*
 * methods.c — Three narrowphase methods, all matching the packing solver.
 *
 * ── Shared SAT kernel: tri_sat_depth() ──────────────────────────────────
 *
 * Inputs: two triangles already in world frame (3 Vec2 each).
 * Returns: minimum penetration depth if intersecting, 0.0 if separated.
 *
 * Matches physics.c / tri_sat_penetration_idx() exactly:
 *   - Axes are UNNORMALIZED: (-dy, dx) for each edge.
 *   - Degenerate edges (|axis|² < 1e-30) are skipped.
 *   - Gap condition: maxA < minB  (touching = overlapping).
 *
 * ── Energy accumulation ──────────────────────────────────────────────────
 *
 * For a polygon instance pair (i, k):
 *   penalty = Σ_{j,l}  depth(tri_j^i, tri_l^k)²   (all 13×13 pairs)
 *   NO early exit — all pairs summed even if first pair already intersects.
 *
 * This matches the solver's overlap_pair_penalty(), which accumulates
 * depth² over all triangle pairs without early termination.
 *
 * ── Methods ──────────────────────────────────────────────────────────────
 *
 * A  Recomputes world transforms for BOTH polygons inside every pair loop.
 *    No caching.  O(N²·T) transforms where T = n_tris.
 *
 * B  Computes world transforms for the outer polygon i once per outer-loop
 *    iteration (not once per pair).  Inner polygon k still recomputed.
 *    O(N·T + N²·T/2) transforms — partial caching.
 *
 * C  Pre-pass caches world verts for ALL N polygons before the pair loop.
 *    O(N·T) transforms total.  Adds two exact filters per triangle pair:
 *      (1) Per-triangle AABB overlap check.
 *      (2) Hint-axis early-out: centroid-centroid unit vector.
 *    Both filters are exact for triangles (convex): if either shows a gap,
 *    the pair definitely does not intersect.  Numerical results are
 *    identical to A and B.
 *
 * ── Broadphase (all methods) ─────────────────────────────────────────────
 *
 * Two instance-level filters applied before the triangle loop:
 *   (1) Instance AABB overlap (fast, 4 comparisons).
 *   (2) Bounding circle check: dist(ci, ck) < 2*br  (matches solver).
 */

#include "bench.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

/* ── timing ──────────────────────────────────────────────────── */
static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* ── Core SAT kernel ─────────────────────────────────────────── */
/*
 * Both triangles in world frame: vA[3], vB[3].
 * Returns minimum penetration depth (>0) if intersecting, 0 if separated.
 *
 * Axes tested: 3 from A + 3 from B = up to 6.
 * Axis = (-dy, dx) for edge direction (dx, dy).  Unnormalized.
 * Degenerate edges skipped (axis·axis < 1e-30).
 * Gap condition: maxA < minB  (strict — touching = overlap).
 */
static double tri_sat_depth(const Vec2 vA[3], const Vec2 vB[3]) {
    double min_pen = 1e18;

    for (int pass = 0; pass < 2; pass++) {
        const Vec2 *src = (pass == 0) ? vA : vB;
        for (int i = 0; i < 3; i++) {
            int j = (i + 1) % 3;
            double dx = src[j].x - src[i].x;
            double dy = src[j].y - src[i].y;
            double ax = -dy, ay = dx;
            if (ax*ax + ay*ay < 1e-30) continue; /* degenerate edge */

            double minA =  1e18, maxA = -1e18;
            double minB =  1e18, maxB = -1e18;
            for (int k = 0; k < 3; k++) {
                double dA = ax*vA[k].x + ay*vA[k].y;
                double dB = ax*vB[k].x + ay*vB[k].y;
                if (dA < minA) minA = dA; if (dA > maxA) maxA = dA;
                if (dB < minB) minB = dB; if (dB > maxB) maxB = dB;
            }
            /* Strict less-than: touching counts as overlap */
            if (maxA < minB || maxB < minA) return 0.0; /* gap found */

            double pen = (maxA < maxB ? maxA : maxB)
                       - (minA > minB ? minA : minB);
            if (pen < min_pen) min_pen = pen;
        }
    }
    return (min_pen > 0.0) ? min_pen : 0.0;
}

/* ── Instance-level pre-filters ──────────────────────────────── */

/* Compute instance AABB on the fly from triangle world verts */
static AABB inst_aabb_from_tris(
    const PolyDef *poly, double cx, double cy,
    double cos_t, double sin_t)
{
    AABB box = { 1e18, -1e18, 1e18, -1e18 };
    for (int t = 0; t < poly->n_tris; t++) {
        for (int k = 0; k < 3; k++) {
            Vec2 w = v2add(v2rot(poly->tris[t].v[k], cos_t, sin_t),
                           (Vec2){cx, cy});
            if (w.x < box.xmin) box.xmin = w.x;
            if (w.x > box.xmax) box.xmax = w.x;
            if (w.y < box.ymin) box.ymin = w.y;
            if (w.y > box.ymax) box.ymax = w.y;
        }
    }
    return box;
}

/* Bounding circle check: squared distance < (2*br)² */
static inline int circle_overlap(double xi, double yi,
                                  double xk, double yk, double br) {
    double dx = xi - xk, dy = yi - yk;
    double lim = 2.0 * br;
    return (dx*dx + dy*dy) < (lim * lim);
}

/* ═══════════════════════════════════════════════════════════════
 * Method A — Exact match to packing solver
 *
 * Both polygon transforms computed inside the pair loop (no cache).
 * For each instance pair (i,k):
 *   1. Instance AABB filter.
 *   2. Bounding circle filter.
 *   3. All 13×13 = 169 triangle pair SAT calls, no early exit.
 *   4. penalty += depth²  for each intersecting triangle pair.
 * ═══════════════════════════════════════════════════════════════ */
BenchStats bench_method_a(
    const PolyDef *poly, const Pose *poses, int N, int reps)
{
    BenchStats s = {0};
    Vec2 wvA[MAX_TRIS][3], wvB[MAX_TRIS][3];

    double t0 = now_s();

    for (int rep = 0; rep < reps; rep++) {
        for (int i = 0; i < N; i++) {
            double ci = cos(poses[i].theta), si = sin(poses[i].theta);
            AABB baA = inst_aabb_from_tris(poly,
                           poses[i].x, poses[i].y, ci, si);

            for (int k = i+1; k < N; k++) {
                s.pairs_total++;

                /* (1) Instance AABB filter */
                double ck = cos(poses[k].theta), sk = sin(poses[k].theta);
                AABB baB = inst_aabb_from_tris(poly,
                               poses[k].x, poses[k].y, ck, sk);
                if (!aabb_overlap(baA, baB)) continue;

                /* (2) Bounding circle filter (matches solver's br check) */
                if (!circle_overlap(poses[i].x, poses[i].y,
                                    poses[k].x, poses[k].y, poly->br))
                    continue;

                s.broadphase_pass++;

                /* Transform all triangles of i (recomputed per pair) */
                for (int t = 0; t < poly->n_tris; t++) {
                    for (int v = 0; v < 3; v++)
                        wvA[t][v] = v2add(v2rot(poly->tris[t].v[v], ci, si),
                                          (Vec2){poses[i].x, poses[i].y});
                }
                /* Transform all triangles of k (recomputed per pair) */
                for (int t = 0; t < poly->n_tris; t++) {
                    for (int v = 0; v < 3; v++)
                        wvB[t][v] = v2add(v2rot(poly->tris[t].v[v], ck, sk),
                                          (Vec2){poses[k].x, poses[k].y});
                }

                /* (3) All triangle pairs, depth² accumulation, no early exit */
                double penalty = 0.0;
                for (int j = 0; j < poly->n_tris; j++) {
                    for (int l = 0; l < poly->n_tris; l++) {
                        s.narrow_calls++;
                        double d = tri_sat_depth(wvA[j], wvB[l]);
                        penalty += d * d;
                    }
                }
                if (penalty > 0.0) s.collisions++;
            }
        }
    }

    s.wall_time_s  = now_s() - t0;
    long long total = (long long)N*(N-1)/2 * reps;
    s.mpairs_per_s = (s.wall_time_s > 1e-9)
        ? (double)total / s.wall_time_s * 1e-6 : 0;
    return s;
}

/* ═══════════════════════════════════════════════════════════════
 * Method B — Partial-cache (outer polygon only)
 *
 * Same SAT formula and energy accumulation as A.
 * Outer polygon i transforms computed once per i-iteration, not
 * once per pair.  Inner polygon k still recomputed per k.
 * This halves the transform cost for the outer polygon.
 * ═══════════════════════════════════════════════════════════════ */
BenchStats bench_method_b(
    const PolyDef *poly, const Pose *poses, int N, int reps)
{
    BenchStats s = {0};
    Vec2 wvA[MAX_TRIS][3], wvB[MAX_TRIS][3];

    double t0 = now_s();

    for (int rep = 0; rep < reps; rep++) {
        for (int i = 0; i < N; i++) {
            double ci = cos(poses[i].theta), si = sin(poses[i].theta);
            AABB baA = inst_aabb_from_tris(poly,
                           poses[i].x, poses[i].y, ci, si);

            /* Transform outer polygon i ONCE for all its k-partners */
            for (int t = 0; t < poly->n_tris; t++) {
                for (int v = 0; v < 3; v++)
                    wvA[t][v] = v2add(v2rot(poly->tris[t].v[v], ci, si),
                                      (Vec2){poses[i].x, poses[i].y});
            }

            for (int k = i+1; k < N; k++) {
                s.pairs_total++;

                double ck = cos(poses[k].theta), sk = sin(poses[k].theta);
                AABB baB = inst_aabb_from_tris(poly,
                               poses[k].x, poses[k].y, ck, sk);
                if (!aabb_overlap(baA, baB)) continue;

                if (!circle_overlap(poses[i].x, poses[i].y,
                                    poses[k].x, poses[k].y, poly->br))
                    continue;

                s.broadphase_pass++;

                /* Inner polygon k still recomputed per k */
                for (int t = 0; t < poly->n_tris; t++) {
                    for (int v = 0; v < 3; v++)
                        wvB[t][v] = v2add(v2rot(poly->tris[t].v[v], ck, sk),
                                          (Vec2){poses[k].x, poses[k].y});
                }

                double penalty = 0.0;
                for (int j = 0; j < poly->n_tris; j++) {
                    for (int l = 0; l < poly->n_tris; l++) {
                        s.narrow_calls++;
                        double d = tri_sat_depth(wvA[j], wvB[l]);
                        penalty += d * d;
                    }
                }
                if (penalty > 0.0) s.collisions++;
            }
        }
    }

    s.wall_time_s  = now_s() - t0;
    long long total = (long long)N*(N-1)/2 * reps;
    s.mpairs_per_s = (s.wall_time_s > 1e-9)
        ? (double)total / s.wall_time_s * 1e-6 : 0;
    return s;
}

/* ── Method C workspace ──────────────────────────────────────── */
/* Per-instance cached data: world verts, per-triangle AABB, instance AABB */
typedef struct {
    Vec2  wv[MAX_TRIS][3];     /* world verts for all triangles */
    AABB  tri_aabb[MAX_TRIS];  /* per-triangle world AABB       */
    AABB  inst_aabb;           /* union of all tri AABBs        */
} InstCache;

/* Compute world AABB of one world-frame triangle */
static AABB world_tri_aabb(const Vec2 wv[3]) {
    double xmin = wv[0].x, xmax = wv[0].x;
    double ymin = wv[0].y, ymax = wv[0].y;
    for (int v = 1; v < 3; v++) {
        if (wv[v].x < xmin) xmin = wv[v].x;
        if (wv[v].x > xmax) xmax = wv[v].x;
        if (wv[v].y < ymin) ymin = wv[v].y;
        if (wv[v].y > ymax) ymax = wv[v].y;
    }
    return (AABB){xmin, xmax, ymin, ymax};
}

/* ═══════════════════════════════════════════════════════════════
 * Method C — Full pre-pass cache + triangle filters
 *
 * Pre-pass: one sin/cos per instance, cache all triangle world verts
 *           and per-triangle AABBs.  O(N·T) transforms total.
 *
 * Pair loop:
 *   1. Instance AABB filter  (cached).
 *   2. Bounding circle filter.
 *   3. Per triangle pair (j, l):
 *      a. Per-triangle AABB filter  — exact, O(4) comparisons.
 *      b. Hint-axis filter          — exact, cheap early-out.
 *         Axis = unit(centroid_j - centroid_l).
 *         If hint axis shows gap → pair definitely separated.
 *      c. Full SAT (if not filtered) → depth² accumulation.
 *
 * All filters are exact for triangles (triangles are convex).
 * Numerical results identical to Methods A and B.
 * ═══════════════════════════════════════════════════════════════ */
BenchStats bench_method_c(
    const PolyDef *poly, const Pose *poses, int N, int reps,
    CachedStats *cs_out)
{
    BenchStats  s  = {0};
    CachedStats cs = {0};

    InstCache *cache = (InstCache *)calloc(N, sizeof(InstCache));
    if (!cache) { fprintf(stderr, "OOM in method C\n"); exit(1); }

    double t0 = now_s();

    for (int rep = 0; rep < reps; rep++) {

        /* ── Pre-pass: populate cache ──────────────────────────── */
        for (int i = 0; i < N; i++) {
            double cx = poses[i].x, cy = poses[i].y;
            double c  = cos(poses[i].theta), ss = sin(poses[i].theta);

            AABB inst = { 1e18, -1e18, 1e18, -1e18 };

            for (int t = 0; t < poly->n_tris; t++) {
                for (int v = 0; v < 3; v++) {
                    cache[i].wv[t][v] = v2add(
                        v2rot(poly->tris[t].v[v], c, ss),
                        (Vec2){cx, cy});
                }
                AABB tb = world_tri_aabb(cache[i].wv[t]);
                cache[i].tri_aabb[t] = tb;
                if (tb.xmin < inst.xmin) inst.xmin = tb.xmin;
                if (tb.xmax > inst.xmax) inst.xmax = tb.xmax;
                if (tb.ymin < inst.ymin) inst.ymin = tb.ymin;
                if (tb.ymax > inst.ymax) inst.ymax = tb.ymax;
            }
            cache[i].inst_aabb = inst;
        }

        /* ── Pair loop ─────────────────────────────────────────── */
        for (int i = 0; i < N; i++) {
            for (int k = i+1; k < N; k++) {
                s.pairs_total++;

                /* (1) Instance AABB filter (from cache) */
                if (!aabb_overlap(cache[i].inst_aabb, cache[k].inst_aabb))
                    continue;

                /* (2) Bounding circle filter */
                if (!circle_overlap(poses[i].x, poses[i].y,
                                    poses[k].x, poses[k].y, poly->br))
                    continue;

                s.broadphase_pass++;

                double penalty = 0.0;

                for (int j = 0; j < poly->n_tris; j++) {
                    for (int l = 0; l < poly->n_tris; l++) {
                        cs.part_pairs++;

                        /* (3a) Per-triangle AABB filter */
                        if (!aabb_overlap(cache[i].tri_aabb[j],
                                          cache[k].tri_aabb[l]))
                            continue;
                        cs.part_aabb_pass++;

                        /* (3b) Hint-axis filter
                         *  Axis = unit vector from centroid_l to centroid_j.
                         *  If projections don't overlap → exact separation.  */
                        cs.hint_tests++;
                        {
                            /* World centroids from cached verts */
                            double cjx = (cache[i].wv[j][0].x +
                                          cache[i].wv[j][1].x +
                                          cache[i].wv[j][2].x) / 3.0;
                            double cjy = (cache[i].wv[j][0].y +
                                          cache[i].wv[j][1].y +
                                          cache[i].wv[j][2].y) / 3.0;
                            double clx = (cache[k].wv[l][0].x +
                                          cache[k].wv[l][1].x +
                                          cache[k].wv[l][2].x) / 3.0;
                            double cly = (cache[k].wv[l][0].y +
                                          cache[k].wv[l][1].y +
                                          cache[k].wv[l][2].y) / 3.0;

                            double hx = cjx - clx, hy = cjy - cly;
                            double hl = sqrt(hx*hx + hy*hy);
                            if (hl > 1e-12) {
                                hx /= hl; hy /= hl; /* normalise */
                                double minJ= 1e18,maxJ=-1e18;
                                double minL= 1e18,maxL=-1e18;
                                for (int v=0;v<3;v++){
                                    double dj = hx*cache[i].wv[j][v].x
                                              + hy*cache[i].wv[j][v].y;
                                    double dl = hx*cache[k].wv[l][v].x
                                              + hy*cache[k].wv[l][v].y;
                                    if(dj<minJ)minJ=dj; if(dj>maxJ)maxJ=dj;
                                    if(dl<minL)minL=dl; if(dl>maxL)maxL=dl;
                                }
                                /* strict: maxJ < minL = gap */
                                if (maxJ < minL || maxL < minJ) {
                                    cs.hint_rejects++;
                                    continue;
                                }
                            }
                        }

                        /* (3c) Full SAT */
                        s.narrow_calls++;
                        cs.sat_full_calls++;
                        cs.sat_axes_tested += 6; /* always 6 for triangles */

                        double d = tri_sat_depth(cache[i].wv[j],
                                                 cache[k].wv[l]);
                        if (d > 0.0) cs.sat_hits++;
                        penalty += d * d;
                    }
                }

                if (penalty > 0.0) s.collisions++;
            }
        }
    }

    s.wall_time_s  = now_s() - t0;
    long long total = (long long)N*(N-1)/2 * reps;
    s.mpairs_per_s = (s.wall_time_s > 1e-9)
        ? (double)total / s.wall_time_s * 1e-6 : 0;

    if (cs_out) *cs_out = cs;
    free(cache);
    return s;
}
