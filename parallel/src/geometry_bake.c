#include "geometry_bake.h"
#include <stdlib.h>
#include <string.h>

int baked_geometry_build(const ConvexDecomp *D, BakedGeometry *out)
{
    if (!D || !out) return 0;
    memset(out, 0, sizeof(*out));

    int total = 0;
    for (int i = 0; i < D->nParts; i++) {
        total += D->parts[i].n;
    }

    out->nParts = D->nParts;
    out->totalVerts = total;
    out->verts = (Vec2*)malloc(sizeof(Vec2) * (size_t)total);
    out->axes = (Vec2*)malloc(sizeof(Vec2) * (size_t)total);
    out->partAabb = (AABB*)malloc(sizeof(AABB) * (size_t)D->nParts);
    out->partStart = (int*)malloc(sizeof(int) * (size_t)D->nParts);
    out->partCount = (int*)malloc(sizeof(int) * (size_t)D->nParts);

    if (!out->verts || !out->axes || !out->partAabb || !out->partStart || !out->partCount) {
        baked_geometry_free(out);
        return 0;
    }

    int cursor = 0;
    for (int i = 0; i < D->nParts; i++) {
        const ConvexPart *p = &D->parts[i];
        if (p->n < 0 || cursor < 0 || (cursor + p->n) > total) {
            fprintf(stderr, "FATAL: geometry bake overflow (part %d) cursor=%d part_n=%d total=%d\n",
                    i, cursor, p->n, total);
            baked_geometry_free(out);
            return 0;
        }
        out->partStart[i] = cursor;
        out->partCount[i] = p->n;

        AABB box = aabb_empty();
        for (int v = 0; v < p->n; v++) {
            out->verts[cursor + v] = p->v[v];
            out->axes[cursor + v] = p->axis[v];
            aabb_add_point(&box, p->v[v]);
        }
        out->partAabb[i] = box;
        cursor += p->n;
    }

    return 1;
}

void baked_geometry_free(BakedGeometry *g)
{
    if (!g) return;
    free(g->verts);
    free(g->axes);
    free(g->partAabb);
    free(g->partStart);
    free(g->partCount);
    memset(g, 0, sizeof(*g));
}
