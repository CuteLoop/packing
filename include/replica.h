#ifndef REPLICA_H
#define REPLICA_H

#include "common.h"
#include "utils.h"
#include <stdint.h>

#ifndef MAX_N
#define MAX_N 200
#endif

typedef struct {
    double cx[MAX_N];
    double cy[MAX_N];
    double th[MAX_N];

    double temp;
    double step_xy;
    double step_th;

    Totals totals;
    double energy;
    double overlap_penalty;
    double outside_penalty;
    double feas;
    int is_feasible;

    uint64_t rng_s;

    uint64_t proposals_done;
    uint64_t epoch_proposals;
    int replica_id;
} ReplicaState;

typedef struct {
    int N;
    double L;
    double cell;
    double br;

    Grid grid;
    Vec2 world[MAX_N * NV];
    AABB aabb[MAX_N];
    AABB tri_aabb[MAX_N * NTRI];
} Workspace;

void workspace_init(Workspace *w, int N, double L, double cell);
void workspace_free(Workspace *w);

void replica_init_random(ReplicaState *r, int N, double L,
                         uint64_t seed, double temp_init,
                         int replica_id);
void replica_swap(ReplicaState *a, ReplicaState *b);
void replica_clone(ReplicaState *dst, const ReplicaState *src, uint64_t new_seed);

void rebuild_derived(const ReplicaState *rs, Workspace *ws, int N, double L, double cell);
void evaluate_full(ReplicaState *r, Workspace *w, int N, double L,
                   const Weights *weights, double eps_feas);

#endif // REPLICA_H
