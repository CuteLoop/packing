#ifndef ENERGY_H
#define ENERGY_H

#include "geom_vec2.h"
#include "aabb.h"
#include "instance_cache.h"
#include "grid_hash.h"
#include "collide_sat.h"
#include "convex_decomp.h"

// Definition of the packing container (Axis-Aligned Box centered at 0,0)
typedef struct {
    double width;
    double height;
} Container;

// Calculate penalty for an AABB leaving the container.
// Returns 0.0 if inside. Increases linearly with depth of violation.
double outside_penalty_aabb(AABB box, Container container);

// Wall penalty for a single pose (checks all vertices against square box).
double wall_penalty(const ConvexDecomp *D, Pose p, double box_size);

// Debug: Calculate total energy of the entire system from scratch.
// E = (Collision Count) * weight + (Boundary Penalty)
double energy_full(const ConvexDecomp *D, const InstanceCache *C, 
                   GridHash *grid, Container container);

// Returns (E_new - E_old).
// 'oldPose' is required to calculate the baseline energy before the move.
double delta_energy_move_one(const ConvexDecomp *D, const InstanceCache *C, 
                             GridHash *grid, Container container,
                             int idx, Pose oldPose, Pose newPose);

// Pairwise collision energy for two poses (symmetry check).
double get_pair_energy(const ConvexDecomp *D, Pose A, Pose B);

// Full recompute of total energy (pairwise + wall penalties).
double total_energy(const ConvexDecomp *D, Pose *poses, int nInstances, double box_size);

// Incremental delta (pairwise + wall penalties) for a proposed move.
double delta_energy(const ConvexDecomp *D, Pose *poses, int nInstances, int moved_idx, Pose new_pose, double box_size);

#endif
