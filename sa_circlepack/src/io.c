#include "io.h"
#include <stdio.h>

/* Return 0 on success, 1 on failure. */
int write_trace_csv(const char* path, const Trace* tr)
{
    if (!path || !tr) return 1;

    FILE* f = fopen(path, "w");
    if (!f) return 1;

    /* include diagnostic E_pair and E_wall if available in Trace */
    fprintf(f, "step,E,T,E_pair,E_wall,accepted,moved\n");
    for (size_t t = 0; t < tr->n_steps; ++t) {
        fprintf(f, "%zu,%.17g,%.17g,%.17g,%.17g,%d,%zu\n",
                t,
                tr->E[t],
                tr->T[t],
                tr->E_pair[t],
                tr->E_wall[t],
                tr->accepted[t],
                tr->moved[t]);
    }

    fclose(f);
    return 0;
}

int write_centers_csv(const char* path, const Vec2* X, size_t N)
{
    if (!path || (!X && N > 0)) return 1;

    FILE* f = fopen(path, "w");
    if (!f) return 1;

    fprintf(f, "i,x,y\n");
    for (size_t i = 0; i < N; ++i) {
        fprintf(f, "%zu,%.17g,%.17g\n", i, X[i].x, X[i].y);
    }

    fclose(f);
    return 0;
}

int write_best_svg(const char* path, const Vec2* X, size_t N, double L, double r)
{
    if (!path || (!X && N > 0) || L <= 0.0 || r < 0.0) return 1;

    FILE* f = fopen(path, "w");
    if (!f) return 1;

    /* Minimal SVG:
       - viewBox: 0..L in both axes
       - draw container square
       - draw circles
       Note: SVG y-axis points downward. That’s fine for a quick diagnostic. */
    fprintf(f,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        "width=\"800\" height=\"800\" viewBox=\"0 0 %.17g %.17g\">\n",
        L, L);

    /* White background */
    fprintf(f, "  <rect x=\"0\" y=\"0\" width=\"%.17g\" height=\"%.17g\" fill=\"white\"/>\n", L, L);

    /* Container boundary */
    fprintf(f,
        "  <rect x=\"0\" y=\"0\" width=\"%.17g\" height=\"%.17g\" "
        "fill=\"none\" stroke=\"black\" stroke-width=\"%.17g\"/>\n",
        L, L, (L > 0 ? L * 0.002 : 1.0));

    /* Draw circles */
    for (size_t i = 0; i < N; ++i) {
        double sw = (L > 0 ? L * 0.0015 : 0.5);
        fprintf(f,
            "  <circle cx=\"%.17g\" cy=\"%.17g\" r=\"%.17g\" "
            "fill=\"none\" stroke=\"black\" stroke-width=\"%.17g\"/>\n",
            X[i].x, X[i].y, r, sw);
    }

    fprintf(f, "</svg>\n");
    fclose(f);
    return 0;
}

