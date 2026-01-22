#ifndef BOUNDS_H
#define BOUNDS_H

#include <stddef.h>
#include "energy.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Lower bound from area: L >= r * sqrt(N*pi). */
double lower_bound_area(size_t N, double r);

/* Lower bound from diameter: L >= 2r. */
double lower_bound_diameter(double r);

/* Combined basic lower bound: max(2r, r*sqrt(N*pi)). */
double lower_bound_basic(size_t N, double r);

/* Guaranteed grid upper bound: L = 2r * ceil(sqrt(N)). */
double upper_bound_grid(size_t N, double r);

/* Construct a feasible witness on a k x k grid with spacing 2r inside [0,L]^2. */
void init_grid(Vec2* X, size_t N, double r, double L);

#ifdef __cplusplus
}
#endif

#endif
