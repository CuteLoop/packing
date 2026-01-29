#include "energy.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define WEIGHT_COLLISION 1.0
#define WEIGHT_BOUNDARY  10.0

// Helper: Mixed Collision (Explicit Pose vs Cached Instance)
// Transform A (explicit pose) on the fly and test against cached B.
static int collide_mixed_explicit_vs_cached(const ConvexDecomp *D, Pose poseA, 
                                            const InstanceCache *C, int idxB)
{
    int nParts = D->nParts;
    // conservative max verts per part
    const int MAXV = 64;
    Vec2 WA[MAXV];
    Vec2 axisA[MAXV];

    double c = cos(poseA.ang);
    double s = sin(poseA.ang);

    for(int pa=0; pa<nParts; pa++){
        const ConvexPart *PartA = &D->parts[pa];
        if(PartA->n > MAXV) return 1; // fallback: consider collision if too large

        // Transform A's verts/axes
        for(int k=0;k<PartA->n;k++){
            Vec2 v = PartA->v[k];
            WA[k] = v2(c*v.x - s*v.y + poseA.t.x, s*v.x + c*v.y + poseA.t.y);
            Vec2 ax = PartA->axis[k];
            axisA[k] = v2(c*ax.x - s*ax.y, s*ax.x + c*ax.y);
        }

        for(int pb=0; pb<nParts; pb++){
            int offB = idxB * nParts + pb;
            const ConvexPart *PartB = &D->parts[pb];
            const Vec2 *WB = &C->worldVerts[C->vertOffset[offB]];
            const Vec2 *axisB = &C->worldAxis[C->axisOffset[offB]];

            // AABB quick reject
            AABB boxA = aabb_empty();
            for(int v=0; v<PartA->n; v++) aabb_add_point(&boxA, WA[v]);
            if(!aabb_overlap(boxA, C->partAabb[offB])) continue;

            // SAT test: A's axes
            int sep = 0;
            for(int k=0;k<PartA->n;k++){
                double aMin = dot(WA[0], axisA[k]), aMax = aMin;
                for(int v=1; v<PartA->n; v++){ double val = dot(WA[v], axisA[k]); if(val<aMin) aMin=val; if(val>aMax) aMax=val; }
                double bMin = dot(WB[0], axisA[k]), bMax = bMin;
                for(int v=1; v<PartB->n; v++){ double val = dot(WB[v], axisA[k]); if(val<bMin) bMin=val; if(val>bMax) bMax=val; }
                if(aMax < bMin || bMax < aMin){ sep = 1; break; }
            }
            if(sep) continue;

            // SAT test: B's axes
            for(int k=0;k<PartB->n;k++){
                Vec2 ax = axisB[k];
                double aMin = dot(WA[0], ax), aMax = aMin;
                for(int v=1; v<PartA->n; v++){ double val = dot(WA[v], ax); if(val<aMin) aMin=val; if(val>aMax) aMax=val; }
                double bMin = dot(WB[0], ax), bMax = bMin;
                for(int v=1; v<PartB->n; v++){ double val = dot(WB[v], ax); if(val<bMin) bMin=val; if(val>bMax) bMax=val; }
                if(aMax < bMin || bMax < aMin){ sep = 1; break; }
            }
            if(sep) continue;

            return 1; // collision
        }
    }
    return 0;
}

double outside_penalty_aabb(AABB box, Container container) {
    double penalty = 0.0;
    double hw = container.width * 0.5;
    double hh = container.height * 0.5;

    if (box.min.x < -hw) penalty += (-hw - box.min.x);
    if (box.max.x >  hw) penalty += (box.max.x - hw);
    if (box.min.y < -hh) penalty += (-hh - box.min.y);
    if (box.max.y >  hh) penalty += (box.max.y - hh);

    return penalty;
}

double wall_penalty(const ConvexDecomp *D, Pose p, double box_size)
{
    double limit = 0.5 * box_size;
    double penalty = 0.0;

    for(int i=0; i<D->nParts; i++) {
        const ConvexPart *part = &D->parts[i];
        for(int v=0; v<part->n; v++) {
            Vec2 world_v = apply_pose(part->v[v], p);

            if (world_v.x > limit)  penalty += (world_v.x - limit) * (world_v.x - limit);
            if (world_v.x < -limit) penalty += (-limit - world_v.x) * (-limit - world_v.x);
            if (world_v.y > limit)  penalty += (world_v.y - limit) * (world_v.y - limit);
            if (world_v.y < -limit) penalty += (-limit - world_v.y) * (-limit - world_v.y);
        }
    }

    return penalty * 1000.0;
}

double energy_full(const ConvexDecomp *D, const InstanceCache *C, 
                   GridHash *grid, Container container)
{
    int n = C->nInstances;
    double boundary = 0.0;
    long long collisions = 0;
    int *cands = (int*)malloc(sizeof(int)*(size_t)128);
    for(int i=0;i<n;i++){
        boundary += outside_penalty_aabb(C->aabb[i], container);
        int nc = grid_query_candidates(grid, C->aabb[i], i, cands, 128);
        for(int k=0;k<nc;k++){
            int j = cands[k];
            if(j <= i) continue;
            if(collide_cached_convex(D, C, i, j, NULL)) collisions++;
        }
    }
    free(cands);

    double energy = (double)collisions * WEIGHT_COLLISION + boundary * WEIGHT_BOUNDARY;
    return energy;
}

double delta_energy_move_one(const ConvexDecomp *D, const InstanceCache *C, 
                             GridHash *grid, Container container,
                             int idx, Pose oldPose, Pose newPose)
{
    // Old energy for idx: boundary + collisions (using cached state)
    double old_boundary = outside_penalty_aabb(C->aabb[idx], container);

    int buffer_sz = 1024;
    int *cands = (int*)malloc(sizeof(int)*(size_t)buffer_sz);
    int nc = grid_query_candidates(grid, C->aabb[idx], idx, cands, buffer_sz);
    int old_coll = 0;
    for(int k=0;k<nc;k++){
        int j = cands[k];
        if(collide_cached_convex(D, C, idx, j, NULL)) old_coll++;
    }

    // New energy for idx: compute new AABB and test against neighbors
    AABB newA = aabb_empty();
    double cc = cos(newPose.ang), ss = sin(newPose.ang);
    for(int p=0;p<D->nParts;p++){
        const ConvexPart *cp = &D->parts[p];
        for(int v=0; v<cp->n; v++){
            Vec2 w = v2(cc*cp->v[v].x - ss*cp->v[v].y + newPose.t.x, ss*cp->v[v].x + cc*cp->v[v].y + newPose.t.y);
            aabb_add_point(&newA, w);
        }
    }
    double new_boundary = outside_penalty_aabb(newA, container);

    int nc2 = grid_query_candidates(grid, newA, idx, cands, buffer_sz);
    int new_coll = 0;
    for(int k=0;k<nc2;k++){
        int j = cands[k];
        if(collide_mixed_explicit_vs_cached(D, newPose, C, j)) new_coll++;
    }

    free(cands);

    double oldE = (double)old_coll * WEIGHT_COLLISION + old_boundary * WEIGHT_BOUNDARY;
    double newE = (double)new_coll * WEIGHT_COLLISION + new_boundary * WEIGHT_BOUNDARY;
    return newE - oldE;
}

double get_pair_energy(const ConvexDecomp *D, Pose A, Pose B)
{
    Pose poses[2] = { A, B };
    InstanceCache cache = build_instance_cache(D, poses, 2);
    int coll = collide_cached_convex(D, &cache, 0, 1, NULL);
    free_instance_cache(&cache);
    return (double)coll * WEIGHT_COLLISION;
}

double total_energy(const ConvexDecomp *D, Pose *poses, int nInstances, double box_size)
{
    double E = 0.0;

    for(int i=0; i<nInstances; i++) {
        for(int j=i+1; j<nInstances; j++) {
            E += get_pair_energy(D, poses[i], poses[j]);
        }
    }

    for(int i=0; i<nInstances; i++) {
        E += wall_penalty(D, poses[i], box_size);
    }

    return E;
}

double delta_energy(const ConvexDecomp *D, Pose *poses, int nInstances, int moved_idx, Pose new_pose, double box_size)
{
    double E_old = 0.0;
    double E_new = 0.0;

    Pose old_pose = poses[moved_idx];

    E_old += wall_penalty(D, old_pose, box_size);
    E_new += wall_penalty(D, new_pose, box_size);

    for(int i=0; i<nInstances; i++) {
        if(i == moved_idx) continue;
        E_old += get_pair_energy(D, old_pose, poses[i]);
        E_new += get_pair_energy(D, new_pose, poses[i]);
    }

    return E_new - E_old;
}

