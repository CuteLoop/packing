#include "propose.h"

/* Clip helper. Keeps proposals bounded to avoid numerical blow-up. */
static inline double clip(double x, double lo, double hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

Proposal propose_move_one(const Vec2* X, size_t N, double L, double sigma, RNG* rng)
{
    Proposal p;
    p.idx = 0;
    p.new_pos.x = 0.0;
    p.new_pos.y = 0.0;

    if (!X || N == 0) return p;

    /* Choose index uniformly. */
    double u = rng_u01(rng);
    size_t k = (size_t)(u * (double)N);
    if (k >= N) k = N - 1;

    /* Gaussian step: delta ~ N(0, sigma^2 I) */
    double dx = sigma * rng_normal(rng);
    double dy = sigma * rng_normal(rng);

    Vec2 cand;
    cand.x = X[k].x + dx;
    cand.y = X[k].y + dy;

    /* Naive bounding box clip: keep within [0, L] */
    cand.x = clip(cand.x, 0.0, L);
    cand.y = clip(cand.y, 0.0, L);

    p.idx = k;
    p.new_pos = cand;
    return p;
}
