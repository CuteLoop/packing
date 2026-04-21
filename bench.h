#pragma once
/*
 * bench.h — Collision benchmark types and declarations
 *
 * All three methods test the same 15-vertex polygon (NV=15, NTRI=13)
 * matching the geometry in common.h of the packing solver.
 *
 *  Method A : Exact match to solver  — triangle-triangle SAT with
 *             unnormalized axes, depth² accumulation over ALL pairs,
 *             no early exit.  Recomputes transforms for every pair.
 *
 *  Method B : Partial-cache          — same SAT formula and triangle
 *             geometry as A.  Outer polygon transforms computed once
 *             per outer-loop iteration instead of once per pair.
 *
 *  Method C : Full-cache + filters   — pre-pass caches world verts
 *             for ALL N polygons before the pair loop.  Adds a
 *             per-triangle AABB filter and a cheap hint-axis early-out.
 *             Numerically identical to A and B (all filters are exact).
 *
 * AABB boundary convention (matches main solver):
 *   touching polygons (maxA == minB) are considered OVERLAPPING.
 *   Gap condition: maxA < minB  (strict less-than only).
 */

/* POSIX timers + M_PI */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <stdio.h>

/* ── Geometry limits ──────────────────────────────────────────── */
#define MAX_VERTS_PER_PART  4    /* triangles: 3 verts, quads: 4     */
#define MAX_PARTS           16   /* triangles used as parts for B/C  */
#define MAX_TRIS            16   /* actual solver: NV=15, NTRI=13    */
#define MAX_N               512

/* ── Basic types ─────────────────────────────────────────────── */
typedef struct { double x, y; } Vec2;
typedef struct { double xmin, xmax, ymin, ymax; } AABB;

/* One convex part in local frame */
typedef struct {
    int   nv;                          /* vertex count              */
    Vec2  v[MAX_VERTS_PER_PART];       /* CCW local-frame vertices  */
    Vec2  n[MAX_VERTS_PER_PART];       /* outward edge normals      */
    Vec2  centroid;
    AABB  local_aabb;
} ConvexPart;

/* Complete polygon (local frame).
 * tris[] and parts[] both store triangles (nv=3).
 * Methods B and C use parts[], which equals tris[] for this polygon
 * because no convex decomposition exists in the packing solver.    */
typedef struct {
    int        n_parts;
    ConvexPart parts[MAX_PARTS];
    int        n_tris;
    ConvexPart tris[MAX_TRIS];
    double     br;     /* bounding circle radius: max |v - centroid| */
    double     area;
} PolyDef;

/* Rigid-body pose */
typedef struct { double x, y, theta; } Pose;

/* ── Method C workspace (per-thread cache) ────────────────────── */
typedef struct {
    Vec2 wv[MAX_PARTS][MAX_VERTS_PER_PART]; /* world verts  */
    Vec2 wn[MAX_PARTS][MAX_VERTS_PER_PART]; /* world norms  */
    AABB part_aabb[MAX_PARTS];
    AABB inst_aabb;
    Vec2 centroid_world;
} CachedInst;

/* ── Benchmark stats ─────────────────────────────────────────── */
typedef struct {
    long long pairs_total;
    long long broadphase_pass;
    long long narrow_calls;
    long long collisions;
    double    wall_time_s;
    double    mpairs_per_s;
} BenchStats;

/* Extended stats collected only for Method C */
typedef struct {
    long long part_pairs;
    long long part_aabb_pass;
    long long hint_tests;
    long long hint_rejects;
    long long sat_full_calls;
    long long sat_hits;
    long long sat_axes_tested;
} CachedStats;

/* Placement regime */
typedef enum { PLACE_SPARSE, PLACE_DENSE, PLACE_JAMMED } PlaceMode;

/* ── xorshift64 RNG ──────────────────────────────────────────── */
typedef struct { uint64_t s; } RNG;
static inline uint64_t rng_next(RNG *r) {
    r->s ^= r->s << 13;
    r->s ^= r->s >> 7;
    r->s ^= r->s << 17;
    return r->s;
}
static inline double rng_f64(RNG *r) {
    return (double)(rng_next(r) >> 11) * (1.0 / (double)(1ULL << 53));
}

/* ── Function declarations ────────────────────────────────────── */

/* geom */
static inline Vec2   v2add(Vec2 a, Vec2 b)   { return (Vec2){a.x+b.x, a.y+b.y}; }
static inline Vec2   v2sub(Vec2 a, Vec2 b)   { return (Vec2){a.x-b.x, a.y-b.y}; }
static inline double v2dot(Vec2 a, Vec2 b)   { return a.x*b.x + a.y*b.y; }
static inline double v2len(Vec2 a)           { return sqrt(a.x*a.x + a.y*a.y); }
static inline Vec2   v2norm(Vec2 a)          {
    double l = v2len(a);
    return l > 1e-12 ? (Vec2){a.x/l, a.y/l} : (Vec2){0,0};
}
static inline Vec2   v2rot(Vec2 v, double c, double s) {
    return (Vec2){ c*v.x - s*v.y, s*v.x + c*v.y };
}
/*
 * aabb_overlap: touching (==) counts as overlapping.
 * Gap condition is strict: maxA < minB.
 * Matches main solver: "if (maxx < minx)" for no-overlap.
 */
static inline int aabb_overlap(AABB a, AABB b) {
    return !(a.xmax < b.xmin || b.xmax < a.xmin ||
             a.ymax < b.ymin || b.ymax < a.ymin);
}

/* poly.c */
void poly_build_tree(PolyDef *p);

/* placement.c */
void placement_generate(Pose *poses, int N, const PolyDef *p,
                        RNG *rng, PlaceMode mode, double *L_out);

/* methods.c */
BenchStats bench_method_a(const PolyDef *p, const Pose *poses, int N, int reps);
BenchStats bench_method_b(const PolyDef *p, const Pose *poses, int N, int reps);
BenchStats bench_method_c(const PolyDef *p, const Pose *poses, int N, int reps,
                          CachedStats *cs_out);

/* io */
void csv_write_header(FILE *f);
void csv_write_row(FILE *f, int N, const char *regime, const char *method,
                  const BenchStats *s, const CachedStats *cs);
