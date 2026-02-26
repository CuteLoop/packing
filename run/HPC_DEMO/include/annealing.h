#ifndef ANNEALING_H
#define ANNEALING_H

#include "common.h"
#include <stdint.h>
#include "config.h"
#include "utils.h"
#include "replica.h"

double try_pack_at_current_L(State *s, RNG *rng, const PhaseParams *A, const PhaseParams *B, int trials,
                             uint64_t seed, uint64_t run_id,
                             double *out_cx, double *out_cy, double *out_th, int verbose);

double try_pack_at_current_L_old(State *s, RNG *rng, const PhaseParams *A, const PhaseParams *B, int trials,
                                 uint64_t seed, uint64_t run_id,
                                 double *out_cx, double *out_cy, double *out_th, int verbose);

int run_sa_epoch(ReplicaState *r, Workspace *w, int N, double L,
                 Weights *weights, double eps_feas,
                 const PhaseParams *pp, int K, volatile int *early_stop);

#endif // ANNEALING_H
