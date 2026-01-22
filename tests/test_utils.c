#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "../run/HPC_DEMO/include/utils.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void fail(const char *msg) { fprintf(stderr, "FAIL: %s\n", msg); exit(1); }

int main(void) {
    RNG r1, r2;
    rng_seed(&r1, 12345ULL);
    rng_seed(&r2, 12345ULL);

    double a = rng_u01(&r1);
    double b = rng_u01(&r2);
    if (a != b) fail("rng_u01 not deterministic for same seed");

    // make_trial_seed determinism
    uint64_t s1 = make_trial_seed(1ULL, 2ULL, 3ULL);
    uint64_t s2 = make_trial_seed(1ULL, 2ULL, 3ULL);
    if (s1 != s2) fail("make_trial_seed not deterministic");

    uint64_t s3 = make_trial_seed(1ULL, 2ULL, 4ULL);
    if (s1 == s3) fail("make_trial_seed collision for different trial_id");

    // wrap angle
    double v = wrap_angle_0_2pi(-1.0);
    if (!(v >= 0.0 && v < 2.0 * M_PI)) fail("wrap_angle_0_2pi out of range (neg)");

    double w = wrap_angle_0_2pi(2.0 * M_PI + 0.5);
    if (!(w >= 0.0 && w < 2.0 * M_PI)) fail("wrap_angle_0_2pi out of range (big)");

    printf("test_utils: OK\n");
    return 0;
}
