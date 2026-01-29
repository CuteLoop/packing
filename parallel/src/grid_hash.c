#include "grid_hash.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// -----------------------------------------------------------------------------
// Constants / Types
// -----------------------------------------------------------------------------

#define INITIAL_CELL_CAPACITY 4

typedef struct {
    int *ids;
    int count;
    int capacity;
} Cell;

struct GridHash {
    Cell *cells;
    int tableSize;
    double cellSize;
    double invCellSize;
    
    // Scratchpad for query deduplication (avoids small allocs)
    int *scratch;
    int scratchCap;
};

// Large primes for spatial hashing
static const int P1 = 73856093;
static const int P2 = 19349663;

// -----------------------------------------------------------------------------
// Internal Helpers
// -----------------------------------------------------------------------------

static inline int hash_coord(int x, int y, int tableSize) {
    // Handle negative coords correctly for hashing
    unsigned int h = (unsigned int)((x * P1) ^ (y * P2));
    return (int)(h % (unsigned int)tableSize);
}

// Convert AABB to integer grid range [minX, maxX] x [minY, maxY]
static inline void get_grid_range(const GridHash *g, AABB box, 
                                  int *minX, int *minY, int *maxX, int *maxY) 
{
    // Floor ensures we handle negative coords correctly
    *minX = (int)floor(box.min.x * g->invCellSize);
    *minY = (int)floor(box.min.y * g->invCellSize);
    *maxX = (int)floor(box.max.x * g->invCellSize);
    *maxY = (int)floor(box.max.y * g->invCellSize);
}

static void cell_add(Cell *c, int id) {
    if (c->count == c->capacity) {
        c->capacity = c->capacity ? c->capacity * 2 : INITIAL_CELL_CAPACITY;
        c->ids = (int*)realloc(c->ids, sizeof(int) * (size_t)c->capacity);
    }
    c->ids[c->count++] = id;
}

static void cell_remove(Cell *c, int id) {
    // Swap-remove is fast O(1), order doesn't matter inside cell
    for (int i = 0; i < c->count; i++) {
        if (c->ids[i] == id) {
            c->ids[i] = c->ids[--c->count];
            return;
        }
    }
}

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------

GridHash* grid_init(double cellSize, int nInstancesHint) {
    GridHash *g = (GridHash*)malloc(sizeof(GridHash));
    g->cellSize = cellSize;
    g->invCellSize = 1.0 / cellSize;
    
    // Heuristic: table size ~ 2x number of instances to reduce collisions
    // Ensure it's not too small
    g->tableSize = nInstancesHint * 2;
    if (g->tableSize < 128) g->tableSize = 128;

    g->cells = (Cell*)calloc((size_t)g->tableSize, sizeof(Cell));
    
    // Pre-allocate scratch buffer (resize later if needed)
    g->scratchCap = 256; 
    g->scratch = (int*)malloc(sizeof(int) * (size_t)g->scratchCap);

    return g;
}

void grid_free(GridHash *g) {
    if (!g) return;
    for (int i = 0; i < g->tableSize; i++) {
        free(g->cells[i].ids);
    }
    free(g->cells);
    free(g->scratch);
    free(g);
}

void grid_build_all(GridHash *g, const AABB *aabbs, int n) {
    // Clear existing
    for (int i = 0; i < g->tableSize; i++) {
        g->cells[i].count = 0;
    }

    // Insert all
    for (int i = 0; i < n; i++) {
        grid_update_one(g, i, aabb_empty(), aabbs[i]);
    }
}

void grid_update_one(GridHash *g, int id, AABB oldAabb, AABB newAabb) {
    int minX, minY, maxX, maxY;

    // Remove from old cells (if AABB not empty)
    if (oldAabb.max.x >= oldAabb.min.x) { // check valid
        get_grid_range(g, oldAabb, &minX, &minY, &maxX, &maxY);
        for (int x = minX; x <= maxX; x++) {
            for (int y = minY; y <= maxY; y++) {
                int h = hash_coord(x, y, g->tableSize);
                cell_remove(&g->cells[h], id);
            }
        }
    }

    // Add to new cells
    if (newAabb.max.x >= newAabb.min.x) {
        get_grid_range(g, newAabb, &minX, &minY, &maxX, &maxY);
        for (int x = minX; x <= maxX; x++) {
            for (int y = minY; y <= maxY; y++) {
                int h = hash_coord(x, y, g->tableSize);
                cell_add(&g->cells[h], id);
            }
        }
    }
}

// Compare function for qsort
static int cmp_ints(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

int grid_query_candidates(GridHash *g, AABB queryBox, int excludeId, 
                          int *outResults, int maxResults) 
{
    int minX, minY, maxX, maxY;
    get_grid_range(g, queryBox, &minX, &minY, &maxX, &maxY);

    int count = 0;
    int capacity = g->scratchCap;
    int *buffer = g->scratch; // pointer alias (could change if realloc)

    // 1. Gather all IDs from overlapping cells
    for (int x = minX; x <= maxX; x++) {
        for (int y = minY; y <= maxY; y++) {
            int h = hash_coord(x, y, g->tableSize);
            Cell *c = &g->cells[h];
            
            // Grow scratch if needed
            while (count + c->count > capacity) {
                capacity *= 2;
                buffer = (int*)realloc(g->scratch, sizeof(int) * (size_t)capacity);
                g->scratch = buffer;
                g->scratchCap = capacity;
            }

            for (int i = 0; i < c->count; i++) {
                int cand = c->ids[i];
                if (cand != excludeId) {
                    buffer[count++] = cand;
                }
            }
        }
    }

    if (count == 0) return 0;

    // 2. Sort
    qsort(buffer, (size_t)count, sizeof(int), cmp_ints);

    // 3. Unique + Copy to output
    int outCount = 0;
    if (count > 0) {
        // Always take the first one
        if (outCount < maxResults) outResults[outCount++] = buffer[0];
        
        for (int i = 1; i < count; i++) {
            // If duplicate, skip
            if (buffer[i] == buffer[i-1]) continue;
            
            // If buffer full, break
            if (outCount >= maxResults) break;
            
            outResults[outCount++] = buffer[i];
        }
    }

    return outCount;
}
