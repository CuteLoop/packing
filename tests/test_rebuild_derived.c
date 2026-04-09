#include "../include/replica.h"
#include "../include/geometry.h"
#include "../include/spatial_hash.h"
#include "../include/physics.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    ReplicaState rs = {0};
    Workspace ws;
    Weights w = { .alpha_L = 0.0, .lambda_ov = 1.0, .mu_out = 1.0 };
    const double eps_feas = 1e-6;

    int N = 2;
    double L = 2.0;
    rs.cx[0] = -0.2; rs.cy[0] = 0.1; rs.th[0] = 0.0;
    rs.cx[1] =  0.2; rs.cy[1] = -0.1; rs.th[1] = 0.1;

    double cell = base_bounding_radius() * 2.0;
    workspace_init(&ws, N, L, cell);

    rebuild_derived(&rs, &ws, N, L, cell);
    evaluate_full(&rs, &ws, N, L, &w, eps_feas);

    assert(ws.grid.cell_id[0] >= 0);
    assert(ws.grid.cell_id[1] >= 0);
    assert(isfinite(rs.feas));
    assert(rs.totals.overlap_total >= 0.0);
    assert(rs.totals.out_total >= 0.0);

    // Compare against legacy State path.
    State legacy = {0};
    legacy.N = N;
    legacy.L = L;
    legacy.cx = malloc((size_t)N * sizeof(double));
    legacy.cy = malloc((size_t)N * sizeof(double));
    legacy.th = malloc((size_t)N * sizeof(double));
    legacy.world = malloc((size_t)N * NV * sizeof(Vec2));
    legacy.aabb = malloc((size_t)N * sizeof(AABB));
    legacy.tri_aabb = malloc((size_t)N * NTRI * sizeof(AABB));
    assert(legacy.cx && legacy.cy && legacy.th && legacy.world && legacy.aabb && legacy.tri_aabb);
    legacy.br = base_bounding_radius();

    for (int i = 0; i < N; i++) {
        legacy.cx[i] = rs.cx[i];
        legacy.cy[i] = rs.cy[i];
        legacy.th[i] = rs.th[i];
        update_instance(&legacy, i);
    }
    grid_rebuild(&legacy.grid, N, L, cell, legacy.cx, legacy.cy);

    Totals legacy_totals = compute_totals_full_grid(&legacy);
    double legacy_feas = feasibility_metric(&legacy_totals);
    double legacy_energy = energy_from_totals(&legacy, &w, &legacy_totals);

    assert(fabs(legacy_feas - rs.feas) < 1e-10);
    assert(fabs(legacy_energy - rs.energy) < 1e-10);

    grid_free(&legacy.grid);
    free(legacy.cx);
    free(legacy.cy);
    free(legacy.th);
    free(legacy.world);
    free(legacy.aabb);
    free(legacy.tri_aabb);

    workspace_free(&ws);

    printf("test_rebuild_derived ok\n");
    return 0;
}
