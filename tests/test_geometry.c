#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../run/HPC_DEMO/include/common.h"
#include "../run/HPC_DEMO/include/geometry.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void fail(const char *msg) { fprintf(stderr, "FAIL: %s\n", msg); exit(1); }

int main(void) {
    double area = base_polygon_area();
    if (!(area > 0.0)) fail("base_polygon_area should be positive");

    double br = base_bounding_radius();
    if (!(br > 0.0)) fail("base_bounding_radius should be positive");

    Vec2 world[NV];
    build_world_verts(world, 1.0, 2.0, M_PI/2.0);
    // For theta = pi/2, rotation: (x,y) -> (-y, x) then + (1,2)
    double ex = -BASE_V[0].y + 1.0;
    double ey =  BASE_V[0].x + 2.0;
    double eps = 1e-9;
    if (fabs(world[0].x - ex) > eps || fabs(world[0].y - ey) > eps) fail("build_world_verts produced unexpected transform");

    printf("test_geometry: OK\n");
    return 0;
}
