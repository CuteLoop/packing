#ifndef RNG_H
#define RNG_H

#include <stdint.h>

typedef struct {
    uint64_t state;
    int has_spare;
    double spare;
} RNG;

/* Seed must be nonzero for xorshift; we will fix-up if 0 is provided. */
void rng_seed(RNG* rng, uint64_t seed);

/* Uniform in [0,1). Deterministic across platforms for same seed. */
double rng_u01(RNG* rng);

/* Standard normal N(0,1) using Box-Muller with caching. */
double rng_normal(RNG* rng);

#endif
