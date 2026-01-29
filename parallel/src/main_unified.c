#include "solver.h"
#include "convex_decomp.h"
#include "geom_vec2.h"
#include "shape_tree.h"
#include "triangulate_earclip.h"
#include "instance_cache.h"
#include "grid_hash.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "export.h"

static void setup_geometry(ConvexDecomp *D)
{
    Poly tree = make_tree_poly_local();
    Triangulation T = triangulate_earclip(&tree);
    *D = convex_decomp_merge_tris(&tree, &T);
    free_triangulation(&T);
    free(tree.v);
}

static void init_random_poses(Pose *poses, int N, double L)
{
    for(int i=0;i<N;i++){
        poses[i].t.x = (rand()/(double)RAND_MAX) * L  - 0.5 * L;
        poses[i].t.y = (rand()/(double)RAND_MAX) * L  - 0.5 * L;
        poses[i].ang = ((rand()%100)/100.0) * 6.283185307179586;
    }
}

static double eval_energy(const ConvexDecomp *D, Pose *poses, int N, Container box)
{
    InstanceCache cache = build_instance_cache(D, poses, N);
    GridHash *grid = grid_init(2.0, N);
    grid_build_all(grid, cache.aabb, N);
    double e = energy_full(D, &cache, grid, box);
    free_instance_cache(&cache);
    grid_free(grid);
    return e;
}

int main(int argc, char **argv) {
    if(argc < 2) {
        printf("Usage: ./packer_unified [serial|parallel] [--n N] [--max_iter M] [--seed S] [--threads T] [--L L] [--search] [--L_low A] [--L_high B] [--search_iter K]\n");
        return 1;
    }

    int N = 200;
    int max_iter = 50000;
    unsigned seed = (unsigned)time(NULL);
    int n_threads = 4;
    double L = 40.0;
    int do_search = 0;
    int search_iter = 8;
    double L_low = 0.0, L_high = 0.0;
    int have_low = 0, have_high = 0;

    for(int i=2;i<argc;i++){
        if(strcmp(argv[i], "--n") == 0 && i+1 < argc) { N = atoi(argv[++i]); }
        else if(strcmp(argv[i], "--max_iter") == 0 && i+1 < argc) { max_iter = atoi(argv[++i]); }
        else if(strcmp(argv[i], "--seed") == 0 && i+1 < argc) { seed = (unsigned)atoi(argv[++i]); }
        else if(strcmp(argv[i], "--threads") == 0 && i+1 < argc) { n_threads = atoi(argv[++i]); }
        else if(strcmp(argv[i], "--L") == 0 && i+1 < argc) { L = atof(argv[++i]); }
        else if(strcmp(argv[i], "--search") == 0) { do_search = 1; }
        else if(strcmp(argv[i], "--L_low") == 0 && i+1 < argc) { L_low = atof(argv[++i]); have_low = 1; }
        else if(strcmp(argv[i], "--L_high") == 0 && i+1 < argc) { L_high = atof(argv[++i]); have_high = 1; }
        else if(strcmp(argv[i], "--search_iter") == 0 && i+1 < argc) { search_iter = atoi(argv[++i]); }
    }

    ConvexDecomp D;
    setup_geometry(&D);

    SolverParams p = {
        .max_iter = max_iter,
        .initial_beta = 0.5,
        .final_beta = 10.0,
        .sigma_trans = 0.2,
        .sigma_rot = 0.1,
        .squeeze_interval = 2000,
        .squeeze_factor = 0.98,
        .n_threads = n_threads
    };

    if(do_search) {
        double high = have_high ? L_high : L;
        double low = have_low ? L_low : (0.5 * high);
        double bestL = high;

        for(int iter=0; iter<search_iter; iter++){
            double mid = 0.5 * (low + high);
            Container box = { .width = mid, .height = mid };
            Pose *poses = malloc(sizeof(Pose)*N);

            srand(seed + (unsigned)(iter * 101));
            init_random_poses(poses, N, mid);

            if(strcmp(argv[1], "parallel") == 0) {
                run_solver_parallel(&D, poses, N, &box, p);
            } else {
                run_solver_serial(&D, poses, N, &box, p);
            }

            double e = eval_energy(&D, poses, N, box);
            if(e <= 1e-9) {
                bestL = mid;
                high = mid;
            } else {
                low = mid;
            }

            free(poses);
        }

        Container box = { .width = bestL, .height = bestL };
        Pose *poses = malloc(sizeof(Pose)*N);
        srand(seed + 12345u);
        init_random_poses(poses, N, bestL);

        if(strcmp(argv[1], "parallel") == 0) {
            run_solver_parallel(&D, poses, N, &box, p);
        } else {
            run_solver_serial(&D, poses, N, &box, p);
        }

        export_svg("packed_result.svg", &D, poses, N, box, 20.0);
        free(poses);
    } else {
        Container box = { .width = L, .height = L };
        Pose *poses = malloc(sizeof(Pose)*N);

        srand(seed);
        init_random_poses(poses, N, L);

        if(strcmp(argv[1], "parallel") == 0) {
            run_solver_parallel(&D, poses, N, &box, p);
        } else {
            run_solver_serial(&D, poses, N, &box, p);
        }

        export_svg("packed_result.svg", &D, poses, N, box, 20.0);
        free(poses);
    }

    free_convex_decomp(&D);

    return 0;
}
