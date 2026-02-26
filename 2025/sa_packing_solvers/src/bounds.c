#include "bounds.h"
#include <math.h>
#include <stddef.h>

double lower_bound_area(size_t N, double r) {
    if (N == 0 || r <= 0.0) return 0.0;
    return r * sqrt((double)N * M_PI);
}

double lower_bound_diameter(double r) {
    return (r > 0.0) ? (2.0 * r) : 0.0;
}

double lower_bound_basic(size_t N, double r) {
    double a = lower_bound_area(N, r);
    double d = lower_bound_diameter(r);
    return (a > d) ? a : d;
}

double upper_bound_grid(size_t N, double r) {
    if (N == 0 || r <= 0.0) return 0.0;
    double k = ceil(sqrt((double)N));
    return 2.0 * r * k;
}

void init_grid(Vec2* X, size_t N, double r, double L) {
    if (!X || N == 0) return;

    const size_t k = (size_t)ceil(sqrt((double)N));
    size_t idx = 0;

    for (size_t j = 0; j < k && idx < N; ++j) {
        for (size_t i = 0; i < k && idx < N; ++i) {
            X[idx].x = r + 2.0 * r * (double)i;
            X[idx].y = r + 2.0 * r * (double)j;
            (void)L;
            idx++;
        }
    }
}
