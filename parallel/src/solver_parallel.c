#include "solver.h"
#include "instance_cache.h"
#include "grid_hash.h"
#include "propose.h"
#include "accept.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>

#define BATCH_SIZE 256

typedef struct {
    int idx;
    Pose oldPose;
    Pose newPose;
    AABB oldAabb;
    double dE;
    bool accepted;
    bool valid;
} MoveRequest;

static inline bool is_conflict(AABB a, AABB b, double safety) {
    AABB a_expanded = { .min = {a.min.x - safety, a.min.y - safety}, .max = {a.max.x + safety, a.max.y + safety} };
    return aabb_overlap(a_expanded, b);
}

static double get_beta(int iter, int max_iter, double start, double end) {
    double t = (double)iter / (double)max_iter;
    return start + t * (end - start);
}

void run_solver_parallel(const ConvexDecomp *D, Pose *poses, int nInstances,
                         Container *container, SolverParams params)
{
    printf("--- Initializing Parallel Solver (%d threads) ---\n", params.n_threads);
    omp_set_num_threads(params.n_threads);

    InstanceCache cache = build_instance_cache(D, poses, nInstances);
    GridHash *grid = grid_init(2.0, nInstances);
    grid_build_all(grid, cache.aabb, nInstances);

    double current_energy = energy_full(D, &cache, grid, *container);
    printf("Start Energy: %.4f\n", current_energy);

    MoveRequest *batch = malloc(sizeof(MoveRequest) * BATCH_SIZE);
    double safety_margin = params.sigma_trans * 2.0 + 1.0;
    int total_accepted = 0;

    for (int iter = 0; iter < params.max_iter; iter += BATCH_SIZE) {

        if (params.squeeze_interval > 0 && iter > 0 && iter % params.squeeze_interval < BATCH_SIZE) {
            container->width  *= params.squeeze_factor;
            container->height *= params.squeeze_factor;
            current_energy = energy_full(D, &cache, grid, *container);
        }

        double L = fmin(container->width, container->height);
        for(int k=0; k<BATCH_SIZE; k++){
            int idx = rand() % nInstances;
            batch[k].idx = idx;
            batch[k].oldPose = poses[idx];
            batch[k].oldAabb = cache.aabb[idx];
            batch[k].newPose = propose_move(batch[k].oldPose, params.sigma_trans, params.sigma_rot, L);
            batch[k].valid = true;
            batch[k].accepted = false;
        }

        for(int i=0; i<BATCH_SIZE; i++){
            if(!batch[i].valid) continue;
            for(int j=i+1; j<BATCH_SIZE; j++){
                if(!batch[j].valid) continue;
                if(batch[i].idx == batch[j].idx) { batch[j].valid = false; continue; }
                if(is_conflict(batch[i].oldAabb, batch[j].oldAabb, safety_margin)) {
                    batch[j].valid = false;
                }
            }
        }

        double beta = get_beta(iter, params.max_iter, params.initial_beta, params.final_beta);

        #pragma omp parallel for schedule(dynamic)
        for(int k=0; k<BATCH_SIZE; k++){
            if(!batch[k].valid) continue;
            batch[k].dE = delta_energy_move_one(D, &cache, grid, *container, batch[k].idx, batch[k].oldPose, batch[k].newPose);
            batch[k].accepted = accept_proposal(batch[k].dE, beta);
        }

        for(int k=0; k<BATCH_SIZE; k++){
            if(batch[k].valid && batch[k].accepted) {
                int idx = batch[k].idx;
                poses[idx] = batch[k].newPose;
                cache_update_one(D, &cache, idx, batch[k].newPose);
                grid_update_one(grid, idx, batch[k].oldAabb, cache.aabb[idx]);
                current_energy += batch[k].dE;
                total_accepted++;
            }
        }

        if (iter % 10000 < BATCH_SIZE) {
             printf("Iter %d | E: %.2f | Acc: %d\r", iter, current_energy, total_accepted);
             fflush(stdout);
             total_accepted = 0;
        }
    }
    printf("\n--- Parallel Solver Finished. Final Energy: %.4f ---\n", current_energy);

    free(batch);
    free_instance_cache(&cache);
    grid_free(grid);
}
