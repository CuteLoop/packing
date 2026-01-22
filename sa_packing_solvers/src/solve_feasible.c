#include "solve_feasible.h"
#include "anneal.h"
#include "bounds.h"
#include "energy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static Vec2* xalloc_vec2(size_t N) {
    return (Vec2*)calloc(N, sizeof(Vec2));
}

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
) {
    FeasibleSolveResult out;
    memset(&out, 0, sizeof(out));
    out.N = N;
    out.success = 0;
    out.L_best = 0.0;
    out.E_best = INFINITY;
    out.X_best = NULL;
    out.iters = 0;

    if (N == 0) {
        out.success = 1;
        out.L_best = 0.0;
        out.E_best = 0.0;
        out.X_best = NULL;
        return out;
    }

    if (retries_per_L < 1) retries_per_L = 1;
    if (eps_feas < 0.0) eps_feas = 0.0;
    if (L_tol <= 0.0) L_tol = 1e-3;
    if (max_iters == 0) max_iters = 64;

    const double Lmin = 2.0 * r;
    if (L_lo < Lmin) L_lo = Lmin;
    if (L_hi < L_lo) return out;

    double lo = L_lo;
    double hi = L_hi;

    Vec2* bestX = xalloc_vec2(N);
    if (!bestX) return out;

    /* Quick feasibility check at provided upper bound via grid witness. */
    Vec2* gridX = xalloc_vec2(N);
    if (gridX) {
        init_grid(gridX, N, r, L_hi);
        double ep = energy_pair(gridX, N, r);
        double ew = energy_wall(gridX, N, r, L_hi);
        if (is_feasible(gridX, N, r, L_hi, eps_feas)) {
            out.success = 1;
            out.L_best = L_hi;
            out.E_best = energy_total(gridX, N, r, L_hi, alpha);
            memcpy(bestX, gridX, N * sizeof(Vec2));
            out.X_best = bestX;
            free(gridX);
            return out;
        } else {
            /* Debug: report why grid witness failed (kept minimal). */
            fprintf(stderr, "[solve_feasible] grid witness at L_hi=%g: E_pair=%g E_wall=%g eps=%g\n", L_hi, ep, ew, eps_feas);
        }
        free(gridX);
    }

    for (size_t iter = 0; iter < max_iters; ++iter) {
        out.iters = iter + 1;
        if ((hi - lo) <= L_tol) break;
        const double mid = 0.5 * (lo + hi);

        int found = 0;
        double bestE_mid = INFINITY;
        Vec2* bestX_mid = NULL;

        for (int rep = 0; rep < retries_per_L; ++rep) {
            const unsigned long s = seed + 100000UL * (unsigned long)iter + (unsigned long)rep;

            Vec2* X0 = xalloc_vec2(N);
            if (!X0) continue;
            init_grid(X0, N, r, mid);

            AnnealResult ar = anneal_run(X0, N, mid, r, alpha, T0, gamma, n_steps, sigma, s);
            free(X0);

            if (ar.X_best && is_feasible(ar.X_best, N, r, mid, eps_feas)) {
                found = 1;
                if (ar.E_best < bestE_mid) {
                    bestE_mid = ar.E_best;
                    if (!bestX_mid) bestX_mid = xalloc_vec2(N);
                    if (bestX_mid) memcpy(bestX_mid, ar.X_best, N * sizeof(Vec2));
                }
            } else {
                if (ar.E_best < bestE_mid && ar.X_best) {
                    bestE_mid = ar.E_best;
                }
            }

            anneal_free_result(&ar);
        }

        if (found) {
            hi = mid;
            out.success = 1;
            out.L_best = mid;
            out.E_best = bestE_mid;
            if (bestX_mid) {
                memcpy(bestX, bestX_mid, N * sizeof(Vec2));
            }
        } else {
            lo = mid;
        }

        if (bestX_mid) free(bestX_mid);
    }

    if (out.success) {
        out.X_best = bestX;
    } else {
        free(bestX);
    }

    return out;
}

void feasible_free(FeasibleSolveResult* res) {
    if (!res) return;
    free(res->X_best);
    res->X_best = NULL;
}
