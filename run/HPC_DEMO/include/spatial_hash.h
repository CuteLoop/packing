#ifndef SPATIAL_HASH_H
#define SPATIAL_HASH_H

#include "common.h"

// Grid type is defined in common.h (shared State/Grid declaration).
void grid_init(Grid *g, int N, double L, double cell);
void grid_free(Grid *g);
void grid_insert(Grid *g, int i, double x, double y);
void grid_remove(Grid *g, int i);
void grid_update(Grid *g, int i, double x, double y);
void grid_rebuild(Grid *g, int N, double L, double cell, const double *cx, const double *cy);

int grid_index(const Grid *g, int ix, int iy);
int grid_R_cells(const State *s);
void grid_cell_xy(const Grid *g, double x, double y, int *ix, int *iy);

// Map an AABB to cell index range [ix0,iy0]..[ix1,iy1]
void aabb_to_cell_range(const Grid *g, const AABB *b, int *ix0, int *iy0, int *ix1, int *iy1);

// Iterate neighbor candidates around item k and call `visit(j, ctx)` for each
// where `visit` should return non-zero to continue, zero to stop early.
typedef int (*cell_visit_fn)(int j, void *ctx);
void grid_query_neighbors(const Grid *g, const State *s, int k, cell_visit_fn visit, void *ctx);

#endif // SPATIAL_HASH_H
