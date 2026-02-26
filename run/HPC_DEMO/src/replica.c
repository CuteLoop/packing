#include "../include/replica.h"
#include "../include/geometry.h"
#include "../include/spatial_hash.h"
#include "../include/physics.h"
#include "../include/utils.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void fill_legacy_state(State *tmp, const ReplicaState *rs, Workspace *ws, int N, double L) {
    tmp->N = N;
    tmp->L = L;
    tmp->cx = (double *)rs->cx;
    tmp->cy = (double *)rs->cy;
    tmp->th = (double *)rs->th;
    tmp->world = ws->world;
    tmp->aabb = ws->aabb;
    tmp->tri_aabb = ws->tri_aabb;
    tmp->br = ws->br;
    tmp->grid = ws->grid;
}

void workspace_init(Workspace *w, int N, double L, double cell) {
    memset(w, 0, sizeof(*w));
    w->N = N;
    w->L = L;
    w->cell = cell;
    w->br = base_bounding_radius();
    grid_init(&w->grid, N, L, cell);
}

void workspace_free(Workspace *w) {
    if (w->grid.head || w->grid.next || w->grid.prev || w->grid.cell_id) {
        grid_free(&w->grid);
    }
    w->grid.head = NULL;
    w->grid.next = NULL;
    w->grid.prev = NULL;
    w->grid.cell_id = NULL;
}

void replica_init_random(ReplicaState *r, int N, double L,
                         uint64_t seed, double temp_init,
                         int replica_id) {
    RNG tmp;
    memset(r, 0, sizeof(*r));
    r->replica_id = replica_id;
    r->temp = temp_init;

    rng_seed(&tmp, seed);

    for (int i = 0; i < N; i++) {
        r->cx[i] = rng_uniform(&tmp, -0.5 * L, 0.5 * L);
        r->cy[i] = rng_uniform(&tmp, -0.5 * L, 0.5 * L);
        r->th[i] = rng_uniform(&tmp, 0.0, 2.0 * M_PI);
    }

    r->rng_s = tmp.s;
}

void replica_swap(ReplicaState *a, ReplicaState *b) {
    ReplicaState tmp = *a;
    *a = *b;
    *b = tmp;
}

void replica_clone(ReplicaState *dst, const ReplicaState *src, uint64_t new_seed) {
    RNG tmp;
    *dst = *src;
    rng_seed(&tmp, new_seed);
    dst->rng_s = tmp.s;
}

void rebuild_derived(const ReplicaState *rs, Workspace *ws, int N, double L, double cell) {
    State tmp;

    ws->N = N;
    ws->L = L;
    ws->cell = cell;
    ws->br = base_bounding_radius();

    fill_legacy_state(&tmp, rs, ws, N, L);

    for (int i = 0; i < N; i++) {
        update_instance(&tmp, i);
    }

    grid_rebuild(&ws->grid, N, L, cell, rs->cx, rs->cy);
    tmp.grid = ws->grid;
}

void evaluate_full(ReplicaState *r, Workspace *w, int N, double L,
                   const Weights *weights, double eps_feas) {
    State tmp;
    fill_legacy_state(&tmp, r, w, N, L);

    Totals totals = compute_totals_full_grid(&tmp);
    r->totals = totals;
    r->feas = feasibility_metric(&totals);
    r->is_feasible = (r->feas <= eps_feas) ? 1 : 0;
    r->energy = energy_from_totals(&tmp, weights, &totals);
    r->overlap_penalty = totals.overlap_total;
    r->outside_penalty = totals.out_total;
}
