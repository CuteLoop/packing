#ifndef ANNEALING_H
#define ANNEALING_H

#include "common.h"
#include <stdint.h>
#include "config.h"
#include "utils.h"

double try_pack_at_current_L(State *s, RNG *rng, const PhaseParams *A, const PhaseParams *B, int trials,
                             uint64_t seed, uint64_t run_id,
                             double *out_cx, double *out_cy, double *out_th, int verbose);

#endif // ANNEALING_H
