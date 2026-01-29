// instance_cache.h - Instance cache used by the cached SAT narrowphase
#ifndef INSTANCE_CACHE_H
#define INSTANCE_CACHE_H

#include "geom_vec2.h"
#include "aabb.h"
#include "convex_decomp.h"

typedef struct {
    Vec2 *worldVerts;   // flat buffer of world vertices for every (instance, part)
    Vec2 *worldAxis;    // flat buffer of world axes for every (instance, part)
    int  *vertOffset;   // offset into worldVerts: [instance*nParts + part]
    int  *axisOffset;   // offset into worldAxis:  [instance*nParts + part]

    AABB *aabb;         // per instance overall AABB (from all worldVerts)

    AABB *partAabb;     // size nInstances*nParts
    Vec2 *partCenter;   // size nInstances*nParts

    double *cang, *sang;
    int nInstances;
    int nParts;
} InstanceCache;

// Build/Free API used by the benchmark driver
InstanceCache build_instance_cache(const ConvexDecomp *D, const Pose *poses, int nInstances);
void free_instance_cache(InstanceCache *C);

// Update a single instance in the cache (optional)
void cache_update_one(const ConvexDecomp *D, InstanceCache *C, int i, Pose newPose);

#endif // INSTANCE_CACHE_H
