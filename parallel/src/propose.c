#include "propose.h"
#include <math.h>
#include <stdlib.h>

static double reflect_into_box(double x, double limit) {
    if (limit <= 0.0) return 0.0;
    while (x > limit || x < -limit) {
        if (x > limit) {
            x = limit - (x - limit);
        } else if (x < -limit) {
            x = -limit + (-limit - x);
        }
    }
    return x;
}

Pose propose_move(Pose old, double sigma_t, double sigma_r, double box_size) {
    Pose p = old;
    double limit = box_size * 0.5;

    double dx = (((double)rand() / (double)RAND_MAX) - 0.5) * 2.0 * sigma_t;
    double dy = (((double)rand() / (double)RAND_MAX) - 0.5) * 2.0 * sigma_t;
    double da = (((double)rand() / (double)RAND_MAX) - 0.5) * 2.0 * sigma_r;

    p.t.x += dx;
    p.t.y += dy;
    p.ang += da;

    p.t.x = reflect_into_box(p.t.x, limit);
    p.t.y = reflect_into_box(p.t.y, limit);

    while (p.ang > M_PI)  p.ang -= 2.0 * M_PI;
    while (p.ang < -M_PI) p.ang += 2.0 * M_PI;

    return p;
}

