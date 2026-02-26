#include "rng.h"
#include <assert.h>

int main(void)
{
    RNG a, b;
    rng_seed(&a, 123);
    rng_seed(&b, 123);

    for (int i = 0; i < 1000; ++i) {
        double ua = rng_u01(&a);
        double ub = rng_u01(&b);
        assert(ua == ub);
    }

    rng_seed(&a, 999);
    rng_seed(&b, 999);
    for (int i = 0; i < 1000; ++i) {
        double za = rng_normal(&a);
        double zb = rng_normal(&b);
        assert(za == zb);
    }

    return 0;
}
