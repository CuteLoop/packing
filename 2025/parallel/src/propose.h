#ifndef PROPOSE_H
#define PROPOSE_H

#include "geom_vec2.h"

// Suggest a new pose based on the old one, reflected into a square box.
// sigma_t: translation step scale
// sigma_r: rotation step scale (radians)
// box_size: square side length
Pose propose_move(Pose old, double sigma_t, double sigma_r, double box_size);

#endif
