#ifndef ANNEAL_H
#define ANNEAL_H

#include <stddef.h>
#include "energy.h"
#include "rng.h"

/* Small trace: store at every step (minimal but instrumented). */
typedef struct {
    size_t n_steps;
    double* E;      /* energy trace */
    double* T;      /* temperature trace */
    int* accepted;  /* 0/1 per step */
    size_t* moved;  /* which index moved */
} Trace;

typedef struct {
    Vec2* X_best;
    double E_best;
    double accept_rate;
    Trace trace;
} AnnealResult;

/* Caller owns returned memory (free with anneal_free_result). */
AnnealResult anneal_run(
    const Vec2* X0, size_t N,
    double L, double r, double alpha,
    double T0, double gamma,
    size_t n_steps,
    double sigma,
    uint64_t seed
);

void anneal_free_result(AnnealResult* res);

#endif
