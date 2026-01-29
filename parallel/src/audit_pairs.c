// audit_pairs.c - generate deterministic pose-pair dataset + CPU audit
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "shape_tree.h"
#include "triangulate_earclip.h"
#include "convex_decomp.h"
#include "energy.h"
#include "rng.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N_PAIRS 1000

static double max_radius(const ConvexDecomp *D)
{
    double r2 = 0.0;
    for (int i = 0; i < D->nParts; i++) {
        const ConvexPart *p = &D->parts[i];
        for (int v = 0; v < p->n; v++) {
            double d2 = p->v[v].x * p->v[v].x + p->v[v].y * p->v[v].y;
            if (d2 > r2) r2 = d2;
        }
    }
    return sqrt(r2);
}

static double rand_u01(RNG *rng)
{
    return rng_u01(rng);
}

int main(void)
{
    Poly tree = make_tree_poly_local();
    Triangulation T = triangulate_earclip(&tree);
    ConvexDecomp D = convex_decomp_merge_tris(&tree, &T);

    RNG rng;
    rng_seed(&rng, 42);

    double R = max_radius(&D);
    double overlap_dist = 0.2 * R;
    double touch_dist = 2.0 * R;
    double clear_dist = 2.5 * R;

    FILE *f = fopen("data/pose_pairs.csv", "w");
    if (!f) {
        fprintf(stderr, "Failed to write data/pose_pairs.csv\n");
        return 1;
    }

    fprintf(f, "ax,ay,ang_a,bx,by,ang_b,label\n");

    for (int i = 0; i < N_PAIRS; i++) {
        Pose A, B;
        A.t.x = (rand_u01(&rng) - 0.5) * 10.0;
        A.t.y = (rand_u01(&rng) - 0.5) * 10.0;
        A.ang = rand_u01(&rng) * 2.0 * M_PI;

        double theta = rand_u01(&rng) * 2.0 * M_PI;
        double dist;
        if (i < N_PAIRS/3) dist = overlap_dist;
        else if (i < 2*N_PAIRS/3) dist = touch_dist;
        else dist = clear_dist;

        B.t.x = A.t.x + dist * cos(theta);
        B.t.y = A.t.y + dist * sin(theta);
        B.ang = rand_u01(&rng) * 2.0 * M_PI;

        int label = (get_pair_energy(&D, A, B) > 0.0) ? 1 : 0;
        fprintf(f, "%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%d\n",
                A.t.x, A.t.y, A.ang, B.t.x, B.t.y, B.ang, label);
    }

    fclose(f);

    // Re-read and verify determinism
    f = fopen("data/pose_pairs.csv", "r");
    if (!f) {
        fprintf(stderr, "Failed to read data/pose_pairs.csv\n");
        return 1;
    }

    char line[256];
    int line_no = 0;
    int mismatches = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line_no++ == 0) continue; // header
        Pose A, B;
        int label = 0;
        if (sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%d",
                   &A.t.x, &A.t.y, &A.ang, &B.t.x, &B.t.y, &B.ang, &label) != 7) {
            continue;
        }
        int got = (get_pair_energy(&D, A, B) > 0.0) ? 1 : 0;
        if (got != label) mismatches++;
    }
    fclose(f);

    if (mismatches == 0) {
        printf("[ok] wrote data/pose_pairs.csv (%d pairs), CPU audit PASS\n", N_PAIRS);
    } else {
        printf("[fail] CPU audit mismatches: %d\n", mismatches);
    }

    free_triangulation(&T);
    free(tree.v);
    free_convex_decomp(&D);
    return mismatches ? 2 : 0;
}
