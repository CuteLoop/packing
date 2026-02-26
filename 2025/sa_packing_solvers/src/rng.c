#include "rng.h"
#include <math.h>

static uint64_t xorshift64star(uint64_t* s)
{
    /* xorshift64* : fast deterministic generator */
    uint64_t x = *s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *s = x;
    return x * 2685821657736338717ULL;
}

void rng_seed(RNG* rng, uint64_t seed)
{
    if (!rng) return;
    if (seed == 0) seed = 0x9E3779B97F4A7C15ULL; /* fix-up */
    rng->state = seed;
    rng->has_spare = 0;
    rng->spare = 0.0;
}

double rng_u01(RNG* rng)
{
    /* Convert top 53 bits to double in [0,1). */
    uint64_t x = xorshift64star(&rng->state);
    uint64_t top53 = x >> 11;
    return (double)top53 * (1.0 / 9007199254740992.0); /* 2^53 */
}

double rng_normal(RNG* rng)
{
    /* Box-Muller with caching to reduce log/sqrt calls. */
    if (rng->has_spare) {
        rng->has_spare = 0;
        return rng->spare;
    }

    double u1 = rng_u01(rng);
    double u2 = rng_u01(rng);

    /* Avoid log(0). */
    if (u1 < 1e-300) u1 = 1e-300;

    double R = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;

    double z0 = R * cos(theta);
    double z1 = R * sin(theta);

    rng->spare = z1;
    rng->has_spare = 1;
    return z0;
}
