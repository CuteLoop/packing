#ifndef GRID_HASH_H
#define GRID_HASH_H

#include "aabb.h"

// Opaque handle to the grid structure
typedef struct GridHash GridHash;

// Initialize the grid.
// cellSize: logical side length of one grid cell (tune to avg particle size)
// nInstancesHint: expected number of items (sets hash table size)
GridHash* grid_init(double cellSize, int nInstancesHint);

// Free all grid memory
void grid_free(GridHash *g);

// Clear and rebuild the entire grid from a list of AABBs
void grid_build_all(GridHash *g, const AABB *aabbs, int n);

// Incremental update: Remove 'id' from cells covering oldAabb, add to newAabb.
// Pass id=-1 or invalid AABB to skip remove/add steps respectively.
void grid_update_one(GridHash *g, int id, AABB oldAabb, AABB newAabb);

// Query overlaps: Finds all IDs in cells overlapping 'queryBox'.
// Returns the number of candidates found.
// Writes unique, SORTED IDs into 'outResults' (up to maxResults).
// 'excludeId' is typically the object itself (skipped in output).
int grid_query_candidates(GridHash *g, AABB queryBox, int excludeId,
                          int *outResults, int maxResults);

#endif
