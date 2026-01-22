#ifndef SOLVE_FEASIBLE_H
#define SOLVE_FEASIBLE_H

#include <stddef.h>
#include "energy.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int success;
    double L_best;
    double E_best;
    Vec2* X_best;
    size_t N;
    size_t iters;
} FeasibleSolveResult;

FeasibleSolveResult solve_feasible_bisect(
    size_t N, double r,
    double L_lo, double L_hi,
    double eps_feas,
    double L_tol,
    size_t max_iters,
    double alpha,
    double T0, double gamma,
    size_t n_steps, double sigma,
    unsigned long seed,
    int retries_per_L
);

void feasible_free(FeasibleSolveResult* res);

#ifdef __cplusplus
}
#endif

#endif
