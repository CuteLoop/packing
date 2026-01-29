// test_invariants.c - correctness audit for polygon SAT + delta energy
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "energy.h"
#include "propose.h"
#include "shape_tree.h"
#include "triangulate_earclip.h"
#include "convex_decomp.h"
#include "rng.h"

#define PASS "\033[32mPASS\033[0m"
#define FAIL "\033[31mFAIL\033[0m"

static void test_symmetry(const ConvexDecomp *D) {
    printf("1. Checking Pairwise Symmetry... ");
    Pose A = { {0.0, 0.0}, 0.0 };
    Pose B = { {0.2, 0.1}, 0.5 };

    double E_ab = get_pair_energy(D, A, B);
    double E_ba = get_pair_energy(D, B, A);

    if (fabs(E_ab - E_ba) < 1e-9) {
        printf("%s (E=%.4f)\n", PASS, E_ab);
    } else {
        printf("%s\n   E_ab: %.6f\n   E_ba: %.6f\n", FAIL, E_ab, E_ba);
        exit(1);
    }
}

static void test_delta_exactness(const ConvexDecomp *D) {
    printf("2. Checking Delta-E Exactness... ");

    int N = 5;
    double L = 10.0;
    Pose *poses = malloc(sizeof(Pose) * (size_t)N);

    RNG rng;
    rng_seed(&rng, 42);

    for(int i=0; i<N; i++) {
        poses[i].t.x = (rng_u01(&rng) - 0.5) * L;
        poses[i].t.y = (rng_u01(&rng) - 0.5) * L;
        poses[i].ang = rng_u01(&rng) * 6.283185307179586;
    }

    double E_initial = total_energy(D, poses, N, L);

    int idx = 0;
    Pose old_pose = poses[idx];
    Pose new_pose = propose_move(old_pose, 0.5, 0.1, L);

    double dE_inc = delta_energy(D, poses, N, idx, new_pose, L);

    poses[idx] = new_pose;
    double E_final = total_energy(D, poses, N, L);
    double dE_abs = E_final - E_initial;

    double error = fabs(dE_inc - dE_abs);
    if (error < 1e-5) {
        printf("%s\n", PASS);
    } else {
        printf("%s\n", FAIL);
        printf("   Initial E: %.5f\n", E_initial);
        printf("   Final E:   %.5f\n", E_final);
        printf("   dE (Inc):  %.5f\n", dE_inc);
        printf("   dE (Abs):  %.5f\n", dE_abs);
        printf("   Error:     %.9f\n", error);
        exit(1);
    }
    free(poses);
}

static void test_wall_consistency(const ConvexDecomp *D) {
    printf("3. Checking Wall Penalties... ");
    double L = 10.0;
    Pose center = { {0.0, 0.0}, 0.0 };
    Pose out = { {6.0, 6.0}, 0.0 };

    if (wall_penalty(D, center, L) == 0.0 && wall_penalty(D, out, L) > 0.0) {
        printf("%s\n", PASS);
    } else {
        printf("%s (Check wall logic)\n", FAIL);
        exit(1);
    }
}

int main(void) {
    printf("=== PHASE 1: PHYSICS AUDIT ===\n");

    Poly tree = make_tree_poly_local();
    Triangulation T = triangulate_earclip(&tree);
    ConvexDecomp D = convex_decomp_merge_tris(&tree, &T);

    test_symmetry(&D);
    test_wall_consistency(&D);
    test_delta_exactness(&D);

    printf("=== ALL SYSTEMS GO ===\n");

    free_triangulation(&T);
    free(tree.v);
    free_convex_decomp(&D);
    return 0;
}