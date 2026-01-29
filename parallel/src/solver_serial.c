#include "solver.h"
#include "instance_cache.h"
#include "grid_hash.h"
#include "propose.h"
#include "accept.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Helper to interpolate Beta (Annealing schedule)
static double get_beta(int iter, int max_iter, double start, double end) {
    double t = (double)iter / (double)max_iter;
    return start + t * (end - start);
}

void run_solver_serial(const ConvexDecomp *D, Pose *poses, int nInstances,
                       Container *container, SolverParams params)
{
    printf("--- Initializing Solver (serial) ---\n");

    // 1. Initialize Acceleration Structures
    InstanceCache cache = build_instance_cache(D, poses, nInstances);

    GridHash *grid = grid_init(2.0, nInstances);
    grid_build_all(grid, cache.aabb, nInstances);

    double current_energy = energy_full(D, &cache, grid, *container);
    printf("Start Energy: %.4f | Beta: %.2f\n", current_energy, params.initial_beta);

    int accepted = 0;

    for (int iter = 0; iter < params.max_iter; iter++) {

        if (params.squeeze_interval > 0 && iter > 0 && iter % params.squeeze_interval == 0) {
            container->width  *= params.squeeze_factor;
            container->height *= params.squeeze_factor;
            current_energy = energy_full(D, &cache, grid, *container);
        }

        int idx = rand() % nInstances;
        Pose oldPose = poses[idx];
        AABB oldAabb = cache.aabb[idx];

        double L = fmin(container->width, container->height);
        Pose newPose = propose_move(oldPose, params.sigma_trans, params.sigma_rot, L);

        double dE = delta_energy_move_one(D, &cache, grid, *container, idx, oldPose, newPose);

        double beta = get_beta(iter, params.max_iter, params.initial_beta, params.final_beta);

        if (accept_proposal(dE, beta)) {
            poses[idx] = newPose;
            cache_update_one(D, &cache, idx, newPose);
            grid_update_one(grid, idx, oldAabb, cache.aabb[idx]);
            current_energy += dE;
            accepted++;
        }

        if (iter % 10000 == 0) {
            printf("Iter %d | E: %.2f | Acc: %.1f%%\r",
                   iter, current_energy, 100.0 * (double)accepted / 10000.0);
            fflush(stdout);
            accepted = 0;
        }
    }
    printf("\n--- Solver Finished. Final Energy: %.4f ---\n", current_energy);

    free_instance_cache(&cache);
    grid_free(grid);
}
