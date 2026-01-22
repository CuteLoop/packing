#ifndef PROPOSE_H
#define PROPOSE_H

#include <stddef.h>
#include "energy.h"
#include "rng.h"

/* Proposal: choose one circle index uniformly, add Gaussian step N(0,sigma^2 I),
   and clip to a bounding box (here [0, L] for each coordinate). */
typedef struct {
    size_t idx;
    Vec2 new_pos;
} Proposal;

Proposal propose_move_one(const Vec2* X, size_t N, double L, double sigma, RNG* rng);

#endif
