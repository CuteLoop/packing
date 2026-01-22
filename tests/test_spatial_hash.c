#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../run/HPC_DEMO/include/spatial_hash.h"
#include "../run/HPC_DEMO/include/common.h"

static void fail(const char *msg) { fprintf(stderr, "FAIL: %s\n", msg); exit(1); }

static int visit_count_fn(int j, void *ctx) {
    int *cnt = (int*)ctx;
    (*cnt)++;
    return 1; // continue
}

int main(void) {
    int N = 4;
    double L = 4.0;
    double cell = 1.0;

    Grid g;
    grid_init(&g, N, L, cell);

    double cx[4] = { -1.0, -0.9, 1.0, 1.1 };
    double cy[4] = { 0.0,  0.1,  0.0, 0.1 };

    // insert
    for (int i = 0; i < N; ++i) grid_insert(&g, i, cx[i], cy[i]);

    // basic cell_id sanity
    for (int i = 0; i < N; ++i) {
        if (g.cell_id[i] < 0) fail("cell_id not set after insert");
    }

    // build a minimal State for query
    State s;
    s.N = N;
    s.br = 0.5; // small bounding radius
    s.grid = g;

    int cnt0 = 0;
    grid_query_neighbors(&g, &s, 0, visit_count_fn, &cnt0);
    if (cnt0 < 1) fail("expected at least one neighbor for item 0");

    // move item 0 far away
    grid_update(&g, 0, 10.0, 10.0);
    if (g.cell_id[0] == -1) fail("cell_id lost after update");

    int cnt_after = 0;
    grid_query_neighbors(&g, &s, 0, visit_count_fn, &cnt_after);
    // since we moved far away (outside grid bounds), brute-force path may be used
    if (cnt_after != N-1) fail("expected brute-force neighbor count after moving outside cell");

    grid_free(&g);
    printf("test_spatial_hash: OK\n");
    return 0;
}
