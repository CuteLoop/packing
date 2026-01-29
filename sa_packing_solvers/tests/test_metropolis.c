#include "metropolis.h"
#include "rng.h"
#include <assert.h>
#include <math.h>

int main(void)
{
    RNG rng;
    rng_seed(&rng, 7);

    /* downhill always accepted */
    for (int k = 0; k < 1000; ++k) {
        assert(metropolis_accept(-1.0, 1.0, &rng) == 1);
    }

    /* T<=0 => uphill never accepted */
    assert(metropolis_accept(1.0, 0.0, &rng) == 0);

    /* empirical acceptance close to exp(-dE/T) */
    rng_seed(&rng, 123);
    double dE = 1.0, T = 2.0;
    double p = exp(-dE / T);

    int M = 20000;
    int acc = 0;
    for (int i = 0; i < M; ++i) acc += metropolis_accept(dE, T, &rng);

    double phat = (double)acc / (double)M;

    /* Tight enough to catch bugs, loose enough to avoid flakiness. */
    assert(fabs(phat - p) < 0.02);

    return 0;
}
