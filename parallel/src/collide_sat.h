// collide_sat.h - SAT narrowphase API (baseline + cached)
#ifndef COLLIDE_SAT_H
#define COLLIDE_SAT_H

#include "geom_vec2.h"
#include "convex_decomp.h"
#include "instance_cache.h"
typedef struct {
    long long part_pair_total;
    long long part_aabb_pass;
    long long hint_axis_tests;
    long long hint_axis_rejects;
    long long sat_full_calls;
    long long sat_full_hits;
    long long sat_axes_tested;
} NarrowStats;

// Baseline SAT
int convex_sat_intersect(const Vec2 *A, int nA, const Vec2 *B, int nB);

// Cached-axis SAT (world axes provided)
int convex_sat_intersect_cached_axes(const Vec2 *A, int nA, const Vec2 *axisA_world,
                                    const Vec2 *B, int nB, const Vec2 *axisB_world);

// Counted variant (accumulates number of axes tested into axes_tested_accum if non-NULL)
int convex_sat_intersect_cached_axes_counted(
    const Vec2 *A, int nA, const Vec2 *axisA_world,
    const Vec2 *B, int nB, const Vec2 *axisB_world,
    long long *axes_tested_accum);

// Hint-axis early reject counted
int sat_hint_separated_counted(
    const Vec2 *WA, int nA,
    const Vec2 *WB, int nB,
    Vec2 axis, long long *hint_tests_accum);

// Cached narrowphase over InstanceCache; updates `N` if non-NULL
int collide_cached_convex(const ConvexDecomp *D, const InstanceCache *C, int i, int j, NarrowStats *N);

#endif // COLLIDE_SAT_H
