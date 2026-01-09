// grid_init_trees.c
// Deterministic "compact grid" initializer for N identical ChristmasTree polygons
// in a square, plus SVG output.
//
// Non-overlap guarantee: AABB-separated at a FIXED rotation theta.
//
// Build:
//   gcc -O2 -std=c11 -Wall -Wextra -pedantic grid_init_trees.c -o grid_init_trees -lm
//
// Run:
//   ./grid_init_trees 49 --gap 0.002 --theta 0.0 --snake 0 --out grid.csv --svg grid.svg

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct { double x, y; } Vec2;

typedef struct {
    double minx, miny;
    double maxx, maxy;
} AABB;

// ----------------------- ChristmasTree polygon (UNSCALED) ---------------------
// Direct translation of your Python polygon vertices (before rotate/translate).
//
// Parameters from Python:
// trunk_w = 0.15, trunk_h = 0.2
// base_w  = 0.7,  mid_w   = 0.4, top_w = 0.25
// tip_y   = 0.8,  tier_1_y= 0.5, tier_2_y=0.25, base_y=0.0
// trunk_bottom_y = -trunk_h = -0.2
//
// Vertex order matches your Python list exactly.
#define NV 15
static const Vec2 BASE_V[NV] = {
    // Tip
    {  0.0,        0.8  },

    // Right side - Top Tier
    {  0.25/2.0,   0.5  },   //  top_w/2
    {  0.25/4.0,   0.5  },   //  top_w/4

    // Right side - Middle Tier
    {  0.4/2.0,    0.25 },   //  mid_w/2
    {  0.4/4.0,    0.25 },   //  mid_w/4

    // Right side - Bottom Tier
    {  0.7/2.0,    0.0  },   //  base_w/2

    // Right Trunk
    {  0.15/2.0,   0.0  },   //  trunk_w/2
    {  0.15/2.0,  -0.2  },   //  trunk_bottom_y

    // Left Trunk
    { -0.15/2.0,  -0.2  },
    { -0.15/2.0,   0.0  },

    // Left side - Bottom Tier
    { -0.7/2.0,    0.0  },

    // Left side - Middle Tier
    { -0.4/4.0,    0.25 },
    { -0.4/2.0,    0.25 },

    // Left side - Top Tier
    { -0.25/4.0,   0.5  },
    { -0.25/2.0,   0.5  },
};
// ---------------------------------------------------------------------------

static double wrap_angle_0_2pi(double t) {
    t = fmod(t, 2.0 * M_PI);
    if (t < 0.0) t += 2.0 * M_PI;
    return t;
}

static AABB aabb_of_verts(const Vec2 *v, int n) {
    AABB b;
    b.minx = b.maxx = v[0].x;
    b.miny = b.maxy = v[0].y;
    for (int i = 1; i < n; i++) {
        if (v[i].x < b.minx) b.minx = v[i].x;
        if (v[i].x > b.maxx) b.maxx = v[i].x;
        if (v[i].y < b.miny) b.miny = v[i].y;
        if (v[i].y > b.maxy) b.maxy = v[i].y;
    }
    return b;
}

// Rotate BASE_V by theta about origin, no translation, return its AABB.
static AABB base_aabb_rot(double theta) {
    double c = cos(theta), s = sin(theta);
    Vec2 w[NV];
    for (int i = 0; i < NV; i++) {
        w[i].x = c * BASE_V[i].x - s * BASE_V[i].y;
        w[i].y = s * BASE_V[i].x + c * BASE_V[i].y;
    }
    return aabb_of_verts(w, NV);
}

// For a placed instance: AABB = rotated_base_aabb shifted by (cx,cy)
static AABB instance_aabb(const AABB *rot_base, double cx, double cy) {
    AABB b = *rot_base;
    b.minx += cx; b.maxx += cx;
    b.miny += cy; b.maxy += cy;
    return b;
}

static void global_aabb_of_instances(const AABB *aabbs, int N, AABB *out) {
    *out = aabbs[0];
    for (int i = 1; i < N; i++) {
        if (aabbs[i].minx < out->minx) out->minx = aabbs[i].minx;
        if (aabbs[i].maxx > out->maxx) out->maxx = aabbs[i].maxx;
        if (aabbs[i].miny < out->miny) out->miny = aabbs[i].miny;
        if (aabbs[i].maxy > out->maxy) out->maxy = aabbs[i].maxy;
    }
}

static double tight_square_L_from_global_aabb(const AABB *g) {
    double w = g->maxx - g->minx;
    double h = g->maxy - g->miny;
    return (w > h) ? w : h;
}

// Shift centers so global AABB center is at origin.
static void recenter_to_origin(double *cx, double *cy, const AABB *global, int N) {
    double gx = 0.5 * (global->minx + global->maxx);
    double gy = 0.5 * (global->miny + global->maxy);
    for (int i = 0; i < N; i++) {
        cx[i] -= gx;
        cy[i] -= gy;
    }
}

// Build world verts for an instance into out[NV]
static void build_world_verts(Vec2 *out, double theta, double cx, double cy) {
    double c = cos(theta), s = sin(theta);
    for (int k = 0; k < NV; k++) {
        double x = BASE_V[k].x;
        double y = BASE_V[k].y;
        double xr = c * x - s * y;
        double yr = s * x + c * y;
        out[k].x = xr + cx;
        out[k].y = yr + cy;
    }
}

// SVG mapping: world->screen with y flip
static double map_x(double x, double xmin, double xmax, double W) {
    return (x - xmin) * (W / (xmax - xmin));
}
static double map_y(double y, double ymin, double ymax, double H) {
    return (ymax - y) * (H / (ymax - ymin));
}

static int write_svg(const char *path,
                     const double *cx, const double *cy, const double *th,
                     int N, double L)
{
    // Use the square bounds as view, with a small padding
    double pad = 0.02 * L;
    double xmin = -0.5 * L - pad;
    double xmax =  0.5 * L + pad;
    double ymin = -0.5 * L - pad;
    double ymax =  0.5 * L + pad;

    const double W = 1000.0;
    const double H = 1000.0;

    FILE *f = fopen(path, "w");
    if (!f) return 0;

    fprintf(f,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%.0f\" height=\"%.0f\" viewBox=\"0 0 %.0f %.0f\">\n",
        W, H, W, H
    );

    // Background
    fprintf(f, "  <rect x=\"0\" y=\"0\" width=\"%.0f\" height=\"%.0f\" fill=\"white\" />\n", W, H);

    // Container square [-L/2, L/2]^2
    {
        double x0 = map_x(-0.5 * L, xmin, xmax, W);
        double y0 = map_y( 0.5 * L, ymin, ymax, H);
        double x1 = map_x( 0.5 * L, xmin, xmax, W);
        double y1 = map_y(-0.5 * L, ymin, ymax, H);
        fprintf(f,
            "  <rect x=\"%.6f\" y=\"%.6f\" width=\"%.6f\" height=\"%.6f\" fill=\"none\" stroke=\"black\" stroke-width=\"2\" />\n",
            x0, y0, x1 - x0, y1 - y0
        );
    }

    // Polygons
    for (int i = 0; i < N; i++) {
        Vec2 wv[NV];
        build_world_verts(wv, th[i], cx[i], cy[i]);

        fprintf(f, "  <path d=\"M ");
        for (int k = 0; k < NV; k++) {
            double sx = map_x(wv[k].x, xmin, xmax, W);
            double sy = map_y(wv[k].y, ymin, ymax, H);
            fprintf(f, "%.6f %.6f ", sx, sy);
            if (k == 0) fprintf(f, "L ");
        }
        fprintf(f, "Z\" fill=\"none\" stroke=\"%s\" stroke-width=\"1\" />\n",
                (i % 2 == 0) ? "#444444" : "#777777");
    }

    // Footer
    fprintf(f,
        "  <text x=\"10\" y=\"20\" font-family=\"monospace\" font-size=\"14\" fill=\"black\">"
        "N=%d  L=%.6g"
        "</text>\n",
        N, L
    );

    fprintf(f, "</svg>\n");
    fclose(f);
    return 1;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s N [--gap g] [--theta t] [--snake 0|1] [--out file.csv] [--svg file.svg]\n"
        "  N          number of trees (positive int)\n"
        "  --gap g    extra AABB spacing (default 0.0)\n"
        "  --theta t  fixed rotation in radians (default 0.0)\n"
        "  --snake s  0 row-major, 1 snake rows (default 0)\n"
        "  --out f    output CSV (default grid_init.csv)\n"
        "  --svg f    output SVG (default grid_init.svg)\n",
        prog
    );
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    int N = atoi(argv[1]);
    if (N <= 0) { fprintf(stderr, "ERROR: N must be positive.\n"); return 1; }

    double gap = 0.0;
    double theta = 0.0;
    int snake = 0;
    const char *out_csv = "grid_init.csv";
    const char *out_svg = "grid_init.svg";

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--gap") && i + 1 < argc) {
            gap = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--theta") && i + 1 < argc) {
            theta = atof(argv[++i]);
        } else if (!strcmp(argv[i], "--snake") && i + 1 < argc) {
            snake = atoi(argv[++i]) ? 1 : 0;
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            out_csv = argv[++i];
        } else if (!strcmp(argv[i], "--svg") && i + 1 < argc) {
            out_svg = argv[++i];
        } else {
            fprintf(stderr, "Unknown/invalid arg: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    theta = wrap_angle_0_2pi(theta);
    if (gap < 0.0) gap = 0.0;

    // Allocate
    double *cx = (double*)calloc((size_t)N, sizeof(double));
    double *cy = (double*)calloc((size_t)N, sizeof(double));
    double *th = (double*)calloc((size_t)N, sizeof(double));
    AABB   *bb = (AABB*)calloc((size_t)N, sizeof(AABB));
    if (!cx || !cy || !th || !bb) {
        fprintf(stderr, "ERROR: allocation failed.\n");
        return 1;
    }

    // Near-square grid
    int cols = (int)ceil(sqrt((double)N));
    if (cols < 1) cols = 1;
    int rows = (int)ceil((double)N / (double)cols);
    if (rows < 1) rows = 1;

    // AABB spacing at fixed theta
    AABB rot_base = base_aabb_rot(theta);
    double w = rot_base.maxx - rot_base.minx;
    double h = rot_base.maxy - rot_base.miny;

    double sx = w + gap;
    double sy = h + gap;
    if (sx <= 0.0) sx = 1e-9;
    if (sy <= 0.0) sy = 1e-9;

    // Place in grid order
    int idx = 0;
    for (int r = 0; r < rows && idx < N; r++) {
        int c_start = 0, c_end = cols, c_step = 1;
        if (snake && (r % 2 == 1)) { c_start = cols - 1; c_end = -1; c_step = -1; }

        for (int c = c_start; c != c_end && idx < N; c += c_step) {
            cx[idx] = (double)c * sx;
            cy[idx] = (double)r * sy;
            th[idx] = theta;
            idx++;
        }
    }

    // AABBs and global, recenter, recompute
    for (int i = 0; i < N; i++) bb[i] = instance_aabb(&rot_base, cx[i], cy[i]);
    AABB global0;
    global_aabb_of_instances(bb, N, &global0);

    recenter_to_origin(cx, cy, &global0, N);

    for (int i = 0; i < N; i++) bb[i] = instance_aabb(&rot_base, cx[i], cy[i]);
    AABB global1;
    global_aabb_of_instances(bb, N, &global1);

    double L_tight = tight_square_L_from_global_aabb(&global1);

    // CSV
    FILE *f = fopen(out_csv, "w");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open output file: %s\n", out_csv);
        return 1;
    }
    fprintf(f, "i,cx,cy,theta\n");
    for (int i = 0; i < N; i++) {
        fprintf(f, "%d,%.17g,%.17g,%.17g\n", i, cx[i], cy[i], th[i]);
    }
    fclose(f);

    // SVG
    if (!write_svg(out_svg, cx, cy, th, N, L_tight)) {
        fprintf(stderr, "ERROR: cannot write SVG file: %s\n", out_svg);
        return 1;
    }

    printf("N=%d cols=%d rows=%d\n", N, cols, rows);
    printf("theta=%.17g rad, gap=%.17g\n", theta, gap);
    printf("L_tight=%.17g\n", L_tight);
    printf("CSV written: %s\n", out_csv);
    printf("SVG written: %s\n", out_svg);

    free(cx); free(cy); free(th); free(bb);
    return 0;
}
