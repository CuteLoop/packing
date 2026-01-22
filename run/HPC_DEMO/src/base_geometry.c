#include "../include/common.h"
#include <math.h>

// Definitions for base polygon vertices and triangle indices
const Tri TRIS[NTRI] = {
    {0, 1, 2}, {2, 3, 4}, {0, 2, 4}, {4, 5, 6}, {0, 4, 6}, {0, 6, 7},
    {0, 7, 8}, {0, 8, 9}, {9, 10, 11}, {0, 9, 11}, {11, 12, 13}, {0, 11, 13}, {0, 13, 14}
};

const Vec2 BASE_V[NV] = {
    {  0.0,     0.8  }, {  0.125,   0.5  }, {  0.0625,  0.5  }, {  0.2,     0.25 },
    {  0.1,     0.25 }, {  0.35,    0.0  }, {  0.075,   0.0  }, {  0.075,  -0.2  },
    { -0.075,  -0.2  }, { -0.075,   0.0  }, { -0.35,    0.0  }, { -0.1,     0.25 },
    { -0.2,     0.25 }, { -0.0625,  0.5  }, { -0.125,   0.5  }
};

void build_world_verts(Vec2 *world, double cx, double cy, double theta) {
    double c = cos(theta), s = sin(theta);
    for (int i = 0; i < NV; ++i) {
        world[i].x = c * BASE_V[i].x - s * BASE_V[i].y + cx;
        world[i].y = s * BASE_V[i].x + c * BASE_V[i].y + cy;
    }
}

double base_polygon_area(void) {
    double a = 0.0;
    for (int i = 0; i < NV; ++i) {
        int j = (i + 1) % NV;
        a += BASE_V[i].x * BASE_V[j].y - BASE_V[j].x * BASE_V[i].y;
    }
    return 0.5 * fabs(a);
}

double base_bounding_radius(void) {
    double rmax2 = 0.0;
    for (int i = 0; i < NV; ++i) {
        double d2 = BASE_V[i].x * BASE_V[i].x + BASE_V[i].y * BASE_V[i].y;
        if (d2 > rmax2) rmax2 = d2;
    }
    return sqrt(rmax2);
}

void update_instance(State *s, int i) {
    double c = cos(s->th[i]);
    double sn = sin(s->th[i]);

    Vec2 *w = &s->world[(size_t)i * (size_t)NV];
    for (int v = 0; v < NV; v++) {
        double bx = BASE_V[v].x;
        double by = BASE_V[v].y;
        w[v].x = (bx * c - by * sn) + s->cx[i];
        w[v].y = (bx * sn + by * c) + s->cy[i];
    }

    double minx = w[0].x;
    double maxx = w[0].x;
    double miny = w[0].y;
    double maxy = w[0].y;
    for (int v = 1; v < NV; v++) {
        double x = w[v].x;
        double y = w[v].y;
        if (x < minx) minx = x;
        if (x > maxx) maxx = x;
        if (y < miny) miny = y;
        if (y > maxy) maxy = y;
    }

    s->aabb[i].minx = minx;
    s->aabb[i].maxx = maxx;
    s->aabb[i].miny = miny;
    s->aabb[i].maxy = maxy;

    for (int t = 0; t < NTRI; t++) {
        int i0 = TRIS[t].a;
        int i1 = TRIS[t].b;
        int i2 = TRIS[t].c;
        double t_minx = w[i0].x;
        double t_maxx = t_minx;
        double t_miny = w[i0].y;
        double t_maxy = t_miny;

        if (w[i1].x < t_minx) t_minx = w[i1].x;
        if (w[i1].x > t_maxx) t_maxx = w[i1].x;
        if (w[i1].y < t_miny) t_miny = w[i1].y;
        if (w[i1].y > t_maxy) t_maxy = w[i1].y;

        if (w[i2].x < t_minx) t_minx = w[i2].x;
        if (w[i2].x > t_maxx) t_maxx = w[i2].x;
        if (w[i2].y < t_miny) t_miny = w[i2].y;
        if (w[i2].y > t_maxy) t_maxy = w[i2].y;

        s->tri_aabb[(size_t)i * (size_t)NTRI + (size_t)t].minx = t_minx;
        s->tri_aabb[(size_t)i * (size_t)NTRI + (size_t)t].maxx = t_maxx;
        s->tri_aabb[(size_t)i * (size_t)NTRI + (size_t)t].miny = t_miny;
        s->tri_aabb[(size_t)i * (size_t)NTRI + (size_t)t].maxy = t_maxy;
    }
}
