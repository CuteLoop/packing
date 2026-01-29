#include <assert.h>
#include <float.h>
#include <math.h>
#include "collide_sat.h"

// Project polygon vertices onto axis; return min and max via pointers
static void project_poly_axis(const Vec2 *pts, int n, Vec2 axis, double *out_min, double *out_max) {
    double min = DBL_MAX, max = -DBL_MAX;
    for (int i = 0; i < n; ++i) {
        double p = pts[i].x * axis.x + pts[i].y * axis.y;
        if (p < min) min = p;
        if (p > max) max = p;
    }
    *out_min = min; *out_max = max;
}

// Check separation on single axis
static int axis_separates(const Vec2 *A, int nA, const Vec2 *B, int nB, Vec2 axis) {
    double a_min, a_max, b_min, b_max;
    project_poly_axis(A, nA, axis, &a_min, &a_max);
    project_poly_axis(B, nB, axis, &b_min, &b_max);
    return (a_max < b_min) || (b_max < a_min);
}

int convex_sat_intersect(const Vec2 *A, int nA, const Vec2 *B, int nB) {
    // Axes: normals of edges from A and B
    for (int i = 0; i < nA; ++i) {
        Vec2 e = {A[(i+1)%nA].x - A[i].x, A[(i+1)%nA].y - A[i].y};
        Vec2 axis = {-e.y, e.x};
        if (axis_separates(A, nA, B, nB, axis)) return 0;
    }
    for (int i = 0; i < nB; ++i) {
        Vec2 e = {B[(i+1)%nB].x - B[i].x, B[(i+1)%nB].y - B[i].y};
        Vec2 axis = {-e.y, e.x};
        if (axis_separates(A, nA, B, nB, axis)) return 0;
    }
    return 1;
}

int convex_sat_intersect_cached_axes_counted(
    const Vec2 *A, int nA, const Vec2 *axisA_world,
    const Vec2 *B, int nB, const Vec2 *axisB_world,
    long long *axes_tested_accum)
{
    long long local = 0;
    // Test A's axes
    for (int i = 0; i < nA; ++i) {
        ++local;
        if (axis_separates(A, nA, B, nB, axisA_world[i])) {
            if (axes_tested_accum) *axes_tested_accum += local;
            return 0;
        }
    }
    // Test B's axes
    for (int i = 0; i < nB; ++i) {
        ++local;
        if (axis_separates(A, nA, B, nB, axisB_world[i])) {
            if (axes_tested_accum) *axes_tested_accum += local;
            return 0;
        }
    }
    if (axes_tested_accum) *axes_tested_accum += local;
    return 1;
}

int convex_sat_intersect_cached_axes(const Vec2 *A, int nA, const Vec2 *axisA_world,
                                    const Vec2 *B, int nB, const Vec2 *axisB_world)
{
    return convex_sat_intersect_cached_axes_counted(A, nA, axisA_world, B, nB, axisB_world, NULL);
}

int sat_hint_separated_counted(
    const Vec2 *WA, int nA,
    const Vec2 *WB, int nB,
    Vec2 axis, long long *hint_tests_accum)
{
    if (hint_tests_accum) (*hint_tests_accum)++;
    double a_min, a_max, b_min, b_max;
    project_poly_axis(WA, nA, axis, &a_min, &a_max);
    project_poly_axis(WB, nB, axis, &b_min, &b_max);
    return (a_max < b_min) || (b_max < a_min);
}

int collide_cached_convex(const ConvexDecomp *D, const InstanceCache *C, int i, int j, NarrowStats *N){
    int nParts = C->nParts;

    for(int pa=0; pa<nParts; pa++){
        const ConvexPart *A = &D->parts[pa];
        const Vec2 *WA  = &C->worldVerts[C->vertOffset[i*nParts + pa]];
        const Vec2 *AxW = &C->worldAxis[C->axisOffset[i*nParts + pa]];
        AABB aBox = C->partAabb[i*nParts + pa];
        Vec2 aCtr = C->partCenter[i*nParts + pa];

        for(int pb=0; pb<nParts; pb++){
            const ConvexPart *B = &D->parts[pb];
            const Vec2 *WB  = &C->worldVerts[C->vertOffset[j*nParts + pb]];
            const Vec2 *BxW = &C->worldAxis[C->axisOffset[j*nParts + pb]];
            AABB bBox = C->partAabb[j*nParts + pb];
            Vec2 bCtr = C->partCenter[j*nParts + pb];

            if(N) N->part_pair_total++;

            if(!aabb_overlap(aBox, bBox)) continue;
            if(N) N->part_aabb_pass++;

            Vec2 d = sub(bCtr, aCtr);
            Vec2 ax0 = norm_or_zero(d);
            if(ax0.x != 0.0 || ax0.y != 0.0){
                if(sat_hint_separated_counted(WA, A->n, WB, B->n, ax0, N? &N->hint_axis_tests : NULL)){
                    if(N) N->hint_axis_rejects++;
                    continue;
                }
                Vec2 ax1 = perp(ax0);
                if(sat_hint_separated_counted(WA, A->n, WB, B->n, ax1, N? &N->hint_axis_tests : NULL)){
                    if(N) N->hint_axis_rejects++;
                    continue;
                }
            }

            if(N) N->sat_full_calls++;
            if(convex_sat_intersect_cached_axes_counted(WA, A->n, AxW, WB, B->n, BxW, N? &N->sat_axes_tested : NULL)){
                if(N) N->sat_full_hits++;
                return 1;
            }
        }
    }
    return 0;
}
