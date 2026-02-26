#ifndef SOLVER_H
#define SOLVER_H

#include "convex_decomp.h"
#include "energy.h" // Includes Container def

typedef struct {
    int max_iter;
    double initial_beta;
    double final_beta;
    double sigma_trans;
    double sigma_rot;
    int squeeze_interval;
    double squeeze_factor;
    int n_threads; // 1 = serial, >1 = parallel
} SolverParams;

// Serial Implementation (Phase 5 Logic)
void run_solver_serial(const ConvexDecomp *D, Pose *poses, int nInstances,
                       Container *container, SolverParams params);

// Parallel Implementation (Phase 6 Logic)
void run_solver_parallel(const ConvexDecomp *D, Pose *poses, int nInstances,
                         Container *container, SolverParams params);

#endif // SOLVER_H
