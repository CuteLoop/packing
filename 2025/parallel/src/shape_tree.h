// shape_tree.h - tree polygon factory and basic shape types
#ifndef SHAPE_TREE_H
#define SHAPE_TREE_H

#include "geom_vec2.h"

typedef struct {
    int n;
    Vec2 *v; // CCW assumed
} Poly;

typedef struct { int a,b,c; } Tri;

typedef struct {
    int nTris;
    Tri *tris;
} Triangulation;

// Build an example 'tree' polygon in local coordinates. Caller owns returned Poly.v array.
Poly make_tree_poly_local(void);

double poly_area(const Poly *p);

// Free helper for Poly if needed.
static inline void free_poly(Poly *p){ if(p && p->v) free(p->v); p->v=NULL; p->n=0; }

#endif // SHAPE_TREE_H
