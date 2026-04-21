#pragma once

#include "replica.h"
#include "common.h"
#include <stdint.h>

typedef struct {
    int N;
    int R;
    uint64_t seed;
    uint64_t run_id;
    char out_prefix[256];
    char method[16];
    char run_type[32];     /* smoke | graph | hero | gate_a | gate_b | gate_c | pilot | dev */
    double time_budget_sec;
    double eps_feas;
    Weights weights;
} StudyConfig;

typedef struct {
    int feasible;
    double min_energy;
    double min_feas;
    ReplicaState best_state;
    int has_state;
    double slice_used_sec;
    int resample_events;
} SliceResult;

typedef void (*method_runner_fn)(
    const StudyConfig *cfg,
    double L,
    double slice_budget_sec,
    int probe_idx,
    const ReplicaState *warm_init,
    SliceResult *out
);

typedef struct {
    double L_best;
    double L_lo, L_hi;
    int probes_done;
    int feasible_found;
    ReplicaState best_state;
} BisectionResult;

BisectionResult bisection_run(const StudyConfig *cfg, method_runner_fn runner);

void warmstart_scale(ReplicaState *r, int N, double L_old, double L_new);
void warmstart_repair(ReplicaState *r, Workspace *w, int N, double L,
                      const Weights *weights, double eps_feas,
                      double max_repair_sec);

void runner_ms_stub(
    const StudyConfig *cfg,
    double L,
    double slice_budget_sec,
    int probe_idx,
    const ReplicaState *warm_init,
    SliceResult *out
);
