// triangulate_earclip.h - ear clipping triangulation API
#ifndef TRIANGULATE_EARC_LIP_H
#define TRIANGULATE_EARC_LIP_H

#include "shape_tree.h"

// Triangulate polygon p (assumes CCW order). Returns Triangulation with owned tris array.
Triangulation triangulate_earclip(const Poly *p);

// Free triangulation
static inline void free_triangulation(Triangulation *T){ if(T && T->tris) free(T->tris); T->tris=NULL; T->nTris=0; }

#endif // TRIANGULATE_EARC_LIP_H
