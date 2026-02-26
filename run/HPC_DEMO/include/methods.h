#pragma once

#include "bisection.h"

void runner_ms(
    const StudyConfig *cfg,
    double L,
    double slice_budget_sec,
    int probe_idx,
    const ReplicaState *warm_init,
    SliceResult *out
);

void runner_erms(
    const StudyConfig *cfg,
    double L,
    double slice_budget_sec,
    int probe_idx,
    const ReplicaState *warm_init,
    SliceResult *out
);

void runner_pt(
    const StudyConfig *cfg,
    double L,
    double slice_budget_sec,
    int probe_idx,
    const ReplicaState *warm_init,
    SliceResult *out
);

double pt_swap_accept_prob(double Ei, double Ej, double Ti, double Tj);
