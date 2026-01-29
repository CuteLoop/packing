// geometry_bake.h - flatten convex decomposition for constant-memory style layouts
#ifndef GEOMETRY_BAKE_H
#define GEOMETRY_BAKE_H

#include "convex_decomp.h"
#include "aabb.h"
#include <stddef.h>

typedef struct {
    int nParts;
    int totalVerts;
    // contiguous buffers (SoA-style)
    Vec2 *verts;     // [totalVerts]
    Vec2 *axes;      // [totalVerts] (one axis per vertex/edge)
    AABB *partAabb;  // [nParts]
    int *partStart;  // [nParts]
    int *partCount;  // [nParts]
} BakedGeometry;

// Bake convex parts into flattened arrays (caller frees with baked_geometry_free).
int baked_geometry_build(const ConvexDecomp *D, BakedGeometry *out);
void baked_geometry_free(BakedGeometry *g);

#endif // GEOMETRY_BAKE_H
