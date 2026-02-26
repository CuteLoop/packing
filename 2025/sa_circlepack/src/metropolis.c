#include "metropolis.h"
#include <math.h>

int metropolis_accept(double dE, double T, RNG* rng)
{
    /* Deterministic behavior for “downhill” and careful handling of T. */
    if (dE <= 0.0) return 1;
    if (T <= 0.0) return 0;

    double a = exp(-dE / T);      /* acceptance probability */
    double u = rng_u01(rng);      /* uniform in [0,1) */
    return (u < a) ? 1 : 0;
}
