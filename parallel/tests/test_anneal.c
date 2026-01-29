#include "anneal.h"
#include "energy.h"
#include <assert.h>
#include <string.h>

int main(void)
{
    /* deterministic reproducibility: same seed => identical traces */
    /* Use compile-time constant for array size so initializer is valid in C */
    #define N 6
    Vec2 X0[N] = { {1,1}, {2,2}, {3,3}, {4,4}, {5,5}, {6,6} };

    double L = 10.0, r = 0.5, alpha = 0.0;
    double T0 = 1.0, gamma = 0.99, sigma = 0.2;
    size_t steps = 2000;
    uint64_t seed = 999;

    AnnealResult A = anneal_run(X0, N, L, r, alpha, T0, gamma, steps, sigma, seed);
    AnnealResult B = anneal_run(X0, N, L, r, alpha, T0, gamma, steps, sigma, seed);

    assert(A.trace.n_steps == B.trace.n_steps);
    for (size_t t = 0; t < steps; ++t) {
        assert(A.trace.E[t] == B.trace.E[t]);
        assert(A.trace.T[t] == B.trace.T[t]);
        assert(A.trace.accepted[t] == B.trace.accepted[t]);
        assert(A.trace.moved[t] == B.trace.moved[t]);
    }

    /* best-so-far monotonicity: E_best is <= every recorded energy */
    for (size_t t = 0; t < steps; ++t) {
        assert(A.E_best <= A.trace.E[t] + 1e-12);
    }

    anneal_free_result(&A);
    anneal_free_result(&B);
    return 0;
}
