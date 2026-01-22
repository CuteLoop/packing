#include <stdio.h>
#include <stdlib.h>
#include "../run/HPC_DEMO/include/common.h"

static void fail(const char *msg) { fprintf(stderr, "FAIL: %s\n", msg); exit(1); }

int main(void) {
    AABB a = { .minx = -0.6, .miny = -0.6, .maxx = 0.5, .maxy = 0.5 };
    AABB b = { .minx = 0.4, .miny = 0.4, .maxx = 1.5, .maxy = 1.5 };
    AABB c = { .minx = 1.6, .miny = 1.6, .maxx = 2.0, .maxy = 2.0 };

    if (!aabb_overlap(&a, &b)) fail("a and b should overlap");
    if (aabb_overlap(&a, &c)) fail("a and c should NOT overlap");

    double pen = outside_penalty_aabb(&a, 1.0);
    if (!(pen > 0.0)) fail("outside_penalty_aabb should be > 0 when AABB exceeds L/2");

    printf("test_aabb: OK\n");
    return 0;
}
