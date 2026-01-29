#include "anneal.h"
#include "energy.h"
#include "propose.h"
#include "metropolis.h"
#include <stdlib.h>
#include <string.h>

static Vec2* xmalloc_vec2(size_t n)
{
    return (Vec2*)malloc(n * sizeof(Vec2));
}

static double* xmalloc_double(size_t n)
{
    return (double*)malloc(n * sizeof(double));
}

static int* xmalloc_int(size_t n)
{
    return (int*)malloc(n * sizeof(int));
}

static size_t* xmalloc_size(size_t n)
{
    return (size_t*)malloc(n * sizeof(size_t));
}

AnnealResult anneal_run(
    const Vec2* X0, size_t N,
    double L, double r, double alpha,
    double T0, double gamma,
    size_t n_steps,
    double sigma,
    uint64_t seed
)
{
    AnnealResult res;
    memset(&res, 0, sizeof(res));

    /* allocate working state */
    Vec2* X = xmalloc_vec2(N);
    Vec2* X_best = xmalloc_vec2(N);
    if (!X || !X_best) return res;

    memcpy(X, X0, N * sizeof(Vec2));
    memcpy(X_best, X0, N * sizeof(Vec2));

    /* allocate trace */
    res.trace.n_steps = n_steps;
    res.trace.E = xmalloc_double(n_steps);
    res.trace.T = xmalloc_double(n_steps);
    res.trace.E_pair = xmalloc_double(n_steps);
    res.trace.E_wall = xmalloc_double(n_steps);
    res.trace.accepted = xmalloc_int(n_steps);
    res.trace.moved = xmalloc_size(n_steps);

    if (!res.trace.E || !res.trace.T || !res.trace.E_pair || !res.trace.E_wall ||
        !res.trace.accepted || !res.trace.moved) {
        free(X); free(X_best);
        return res;
    }

    RNG rng;
    rng_seed(&rng, seed);

    /* initial energy */
    double E = energy_total(X, N, r, L, alpha);
    double E_best = E;
    double T = T0;

    size_t n_accept = 0;

    for (size_t tstep = 0; tstep < n_steps; ++tstep) {
        /* propose move */
        Proposal p = propose_move_one(X, N, L, sigma, &rng);

        /* compute exact delta via O(N) update */
        double dE = delta_energy_move_one(X, N, p.idx, p.new_pos, r, L);

        int acc = metropolis_accept(dE, T, &rng);
        if (acc) {
            X[p.idx] = p.new_pos;
            E += dE;
            n_accept++;
            /* best-so-far monotonicity is enforced by taking min */
            if (E < E_best) {
                E_best = E;
                memcpy(X_best, X, N * sizeof(Vec2));
            }
        }

        /* record diagnostics */
        res.trace.E[tstep] = E;
        res.trace.T[tstep] = T;
        /* diagnostic pair/wall energies (explicit compute) */
        res.trace.E_pair[tstep] = energy_pair(X, N, r);
        res.trace.E_wall[tstep] = energy_wall(X, N, r, L);
        res.trace.accepted[tstep] = acc;
        res.trace.moved[tstep] = p.idx;

        /* geometric cooling */
        T *= gamma;
    }

    res.X_best = X_best;
    res.E_best = E_best;
    res.accept_rate = (n_steps > 0) ? ((double)n_accept / (double)n_steps) : 0.0;

    free(X);
    return res;
}

void anneal_free_result(AnnealResult* res)
{
    if (!res) return;
    free(res->X_best);
    free(res->trace.E);
    free(res->trace.T);
    free(res->trace.E_pair);
    free(res->trace.E_wall);
    free(res->trace.accepted);
    free(res->trace.moved);
    memset(res, 0, sizeof(*res));
}
