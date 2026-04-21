/*
 * poly.c — Actual polygon geometry from common.h / geometry.c.
 *
 * Matches exactly:
 *   NV   = 15  (BASE_V[])
 *   NTRI = 13  (TRIS[])
 *
 * The polygon is a non-convex tree/arrow shape.  It has no convex
 * decomposition in the packing solver, so Methods B and C use the
 * same 13 triangles as Method A, just with different compute strategies.
 *
 * bounding circle radius br = max distance from centroid to any vertex.
 *
 *         0 (0, 0.8)  ← apex
 *        / \
 *  14--13   1--2      ← upper notches (concave)
 *  12--11   3--4      ← lower notches (concave)
 *  10        5        ← arm tips at y = 0
 *   9--...--6         ← inner corners
 *      |  |
 *      8  7           ← trunk bottom  (y = -0.2)
 */

#include "bench.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ── Actual polygon data (copied verbatim from common.h) ─────── */
#define NV   15
#define NTRI 13

static const Vec2 BASE_V[NV] = {
    {  0.0,      0.8  }, /*  0 apex                  */
    {  0.125,    0.5  }, /*  1                        */
    {  0.0625,   0.5  }, /*  2                        */
    {  0.2,      0.25 }, /*  3                        */
    {  0.1,      0.25 }, /*  4                        */
    {  0.35,     0.0  }, /*  5 right arm tip          */
    {  0.075,    0.0  }, /*  6                        */
    {  0.075,   -0.2  }, /*  7 trunk bottom-right     */
    { -0.075,   -0.2  }, /*  8 trunk bottom-left      */
    { -0.075,    0.0  }, /*  9                        */
    { -0.35,     0.0  }, /* 10 left arm tip           */
    { -0.1,      0.25 }, /* 11                        */
    { -0.2,      0.25 }, /* 12                        */
    { -0.0625,   0.5  }, /* 13                        */
    { -0.125,    0.5  }  /* 14                        */
};

/* Triangle index triples — copied verbatim from geometry.c */
static const int TRIS[NTRI][3] = {
    { 0,  1,  2},
    { 2,  3,  4},
    { 0,  2,  4},
    { 4,  5,  6},
    { 0,  4,  6},
    { 0,  6,  7},
    { 0,  7,  8},
    { 0,  8,  9},
    { 9, 10, 11},
    { 0,  9, 11},
    {11, 12, 13},
    { 0, 11, 13},
    { 0, 13, 14}
};

/* ── helpers ─────────────────────────────────────────────────── */

static AABB tri_aabb(const Vec2 *a, const Vec2 *b, const Vec2 *c) {
    double xmin = a->x, xmax = a->x;
    double ymin = a->y, ymax = a->y;
    if (b->x < xmin) xmin = b->x; if (b->x > xmax) xmax = b->x;
    if (c->x < xmin) xmin = c->x; if (c->x > xmax) xmax = c->x;
    if (b->y < ymin) ymin = b->y; if (b->y > ymax) ymax = b->y;
    if (c->y < ymin) ymin = c->y; if (c->y > ymax) ymax = c->y;
    return (AABB){ xmin, xmax, ymin, ymax };
}

/* ── public ──────────────────────────────────────────────────── */
void poly_build_tree(PolyDef *p) {
    memset(p, 0, sizeof(*p));

    /* Centroid of the whole polygon (simple vertex average) */
    double cx = 0, cy = 0;
    for (int i = 0; i < NV; i++) { cx += BASE_V[i].x; cy += BASE_V[i].y; }
    cx /= NV; cy /= NV;

    /* ── Build 13 triangle ConvexParts ───────────────────────── */
    for (int t = 0; t < NTRI; t++) {
        ConvexPart *cp = &p->tris[t];
        memset(cp, 0, sizeof(*cp));
        cp->nv = 3;

        const Vec2 *a = &BASE_V[ TRIS[t][0] ];
        const Vec2 *b = &BASE_V[ TRIS[t][1] ];
        const Vec2 *c = &BASE_V[ TRIS[t][2] ];

        cp->v[0] = *a;
        cp->v[1] = *b;
        cp->v[2] = *c;

        /* centroid */
        cp->centroid.x = (a->x + b->x + c->x) / 3.0;
        cp->centroid.y = (a->y + b->y + c->y) / 3.0;
        cp->local_aabb = tri_aabb(a, b, c);

        /*
         * Edge normals: UNNORMALIZED perpendicular (-dy, dx).
         * Matches physics.c exactly.  Degenerate edges flagged (0,0).
         */
        for (int i = 0; i < 3; i++) {
            int j = (i + 1) % 3;
            double dx = cp->v[j].x - cp->v[i].x;
            double dy = cp->v[j].y - cp->v[i].y;
            if (dx*dx + dy*dy < 1e-30)
                cp->n[i] = (Vec2){0, 0};
            else
                cp->n[i] = (Vec2){-dy, dx};
        }
    }
    p->n_tris = NTRI;

    /*
     * Methods B and C: parts[] = same triangles.
     * No convex decomposition exists in the solver.
     */
    for (int t = 0; t < NTRI; t++) p->parts[t] = p->tris[t];
    p->n_parts = NTRI;

    /* Bounding circle radius */
    double br = 0;
    for (int i = 0; i < NV; i++) {
        double dx = BASE_V[i].x - cx;
        double dy = BASE_V[i].y - cy;
        double d  = sqrt(dx*dx + dy*dy);
        if (d > br) br = d;
    }
    p->br = br;

    /* Area via shoelace on full polygon outline */
    double area = 0;
    for (int i = 0; i < NV; i++) {
        int j = (i + 1) % NV;
        area += BASE_V[i].x * BASE_V[j].y
              - BASE_V[j].x * BASE_V[i].y;
    }
    p->area = fabs(area) * 0.5;

    printf("[poly] tree: NV=%d  NTRI=%d  br=%.4f  area=%.4f\n",
           NV, NTRI, p->br, p->area);
    printf("       centroid=(%.4f, %.4f)\n", cx, cy);
}
