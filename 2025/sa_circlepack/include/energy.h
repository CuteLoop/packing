#ifndef ENERGY_H
#define ENERGY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct { double x, y; } Vec2;

/* Pairwise overlap penalty: sum_{i<j} max(0, 2r - d)^2 */
double energy_pair(const Vec2* X, size_t N, double r);

/* Wall penalty:
   sum_i [ max(0, r-x)^2 + max(0, x-(L-r))^2 + same for y ] */
double energy_wall(const Vec2* X, size_t N, double r, double L);

/* Total energy: E = E_pair + E_wall + alpha * L */
double energy_total(const Vec2* X, size_t N, double r, double L, double alpha);

/* Delta energy when moving a single particle i to new_pos (L fixed). */
double delta_energy_move_one(
   const Vec2* X, size_t N, size_t i, Vec2 new_pos,
   double r, double L
);

#ifdef __cplusplus
}
#endif

#endif
