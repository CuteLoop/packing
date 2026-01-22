#ifndef IO_H
#define IO_H

#include <stddef.h>
#include "anneal.h"
#include "energy.h"

/* Write step-by-step trace to CSV.
   Columns: step,E,T,accepted,moved */
int write_trace_csv(const char* path, const Trace* tr);

/* Write circle centers to CSV.
   Columns: i,x,y */
int write_centers_csv(const char* path, const Vec2* X, size_t N);

/* Write an SVG showing the square [0,L]x[0,L] and circles of radius r.
   - If you pass X_best, it draws those centers.
   - Output is viewBox="0 0 L L" (Y increases downward in SVG; that is fine for visualization).
   - Returns 0 on success, non-zero on failure.
*/

/* Draw circle centers as SVG. `X` may be NULL to produce an empty canvas. */
int write_best_svg(const char* path, const Vec2* X, size_t N, double L, double r);

#endif /* IO_H */
