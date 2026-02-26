#include "propose.h"
#include <assert.h>

int main(void)
{
    RNG rng;
    rng_seed(&rng, 42);

    Vec2 X[5] = { {1,1}, {2,2}, {3,3}, {4,4}, {5,5} };
    double L = 10.0;
    double sigma = 0.5;

    Proposal p = propose_move_one(X, 5, L, sigma, &rng);

    /* exactly one index chosen and new position is within [0,L] */
    assert(p.idx < 5);
    assert(p.new_pos.x >= 0.0 && p.new_pos.x <= L);
    assert(p.new_pos.y >= 0.0 && p.new_pos.y <= L);

    return 0;
}
