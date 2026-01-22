#ifndef PHYSICS_H
#define PHYSICS_H

#include "common.h"
 #include <stddef.h>


int tri_sat_penetration_idx(const Vec2 *wi, const Vec2 *wj,
                           int ai0, int ai1, int ai2,
                           int bj0, int bj1, int bj2,
                           double *depth_out);

Totals compute_totals_full_grid(const State *s);

double overlap_pair_penalty(const State *s, int i, int j);
double overlap_sum_for_k_grid(const State *s, int k);
double outside_for_k(const State *s, int k);
double energy_from_totals(const State *s, const Weights *w, const Totals *t);
double feasibility_metric(const Totals *t);

#endif // PHYSICS_H
