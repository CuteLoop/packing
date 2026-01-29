// convex_decomp.h - convex parts and merging triangles
#ifndef CONVEX_DECOMP_H
#define CONVEX_DECOMP_H

#include "shape_tree.h"

typedef struct {
    int n;
    Vec2 *v;      // local vertices (CCW)
    Vec2 *axis;   // local-space precomputed unit edge normals, length n
} ConvexPart;

typedef struct {
    int nParts;
    ConvexPart *parts; // convex polygons + cached axes
} ConvexDecomp;

// Merge triangles into convex parts. Caller owns returned ConvexDecomp.parts and per-part vertex arrays.
ConvexDecomp convex_decomp_merge_tris(const Poly *p, const Triangulation *T);

// Free helper
static inline void free_convex_decomp(ConvexDecomp *D){
    if(!D) return;
    for(int i=0;i<D->nParts;i++){ free(D->parts[i].v); free(D->parts[i].axis); }
    free(D->parts); D->parts=NULL; D->nParts=0;
}

#endif // CONVEX_DECOMP_H
