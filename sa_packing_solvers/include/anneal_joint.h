#ifndef ANNEAL_JOINT_H
#define ANNEAL_JOINT_H

#include <stddef.h>
#include "energy.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t n_steps;
    double* E;
    double* L;
    double* T;
    int* accepted;
    int* move_type;   /* 0 = moved X, 1 = moved L */
    int* moved_idx;
} JointTrace;

typedef struct {
    Vec2* X_best;
    double L_best;
    double E_best;
    double accept_rate;
    JointTrace trace;
    size_t N;
} JointAnnealResult;

JointAnnealResult anneal_joint_run(
    const Vec2* X0, size_t N,
    double L0, double r, double alpha,
    double T0, double gamma, size_t n_steps,
    double sigma_x, double sigma_L,
    double p_move_x,
    unsigned long seed
);

void anneal_joint_free(JointAnnealResult* res);

#ifdef __cplusplus
}
#endif

#endif
