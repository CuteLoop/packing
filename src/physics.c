
#include "../include/physics.h"
#include "../include/geometry.h"
#include "../include/spatial_hash.h"
#include "../include/utils.h"
#include "../include/config.h"
#include <math.h>

// --- SAT helpers (local) ---
static inline void proj3_idx(const Vec2 *w, int i0, int i1, int i2,
                             double ax, double ay, double *mn, double *mx)
{
    double v0 = w[i0].x * ax + w[i0].y * ay;
    double v1 = w[i1].x * ax + w[i1].y * ay;
    double v2 = w[i2].x * ax + w[i2].y * ay;

    double lo = v0, hi = v0;
    if (v1 < lo) lo = v1;
    if (v1 > hi) hi = v1;
    if (v2 < lo) lo = v2;
    if (v2 > hi) hi = v2;

    *mn = lo; *mx = hi;
}

int tri_sat_penetration_idx(const Vec2 *wi, const Vec2 *wj,
                           int ai0, int ai1, int ai2,
                           int bj0, int bj1, int bj2,
                           double *depth_out)
{
    double min_overlap = 1e300;

    for (int pass = 0; pass < 2; pass++) {
        const Vec2 *w = (pass == 0) ? wi : wj;
        int i0 = (pass == 0) ? ai0 : bj0;
        int i1 = (pass == 0) ? ai1 : bj1;
        int i2 = (pass == 0) ? ai2 : bj2;

        int idx[3] = { i0, i1, i2 };

        for (int e = 0; e < 3; e++) {
            Vec2 p0 = w[idx[e]];
            Vec2 p1 = w[idx[(e + 1) % 3]];
            double ex = p1.x - p0.x;
            double ey = p1.y - p0.y;

            double ax = -ey;
            double ay =  ex;

            double len2 = ax * ax + ay * ay;
            if (len2 < 1e-30) continue;

            double amin, amax, bmin, bmax;
            proj3_idx(wi, ai0, ai1, ai2, ax, ay, &amin, &amax);
            proj3_idx(wj, bj0, bj1, bj2, ax, ay, &bmin, &bmax);

            double o = fmin(amax, bmax) - fmax(amin, bmin);
            if (o <= 0.0) return 0;
            if (o < min_overlap) min_overlap = o;
        }
    }

    if (min_overlap > 1e200) min_overlap = 0.0;
    *depth_out = min_overlap;
    return 1;
}

// --- Pair overlap penalty ---
int bounding_circle_reject(const State *s, int i, int j) {
    double dx = s->cx[i] - s->cx[j];
    double dy = s->cy[i] - s->cy[j];
    double d2 = dx * dx + dy * dy;
    double R = 2.0 * s->br;
    return (d2 > R * R);
}

double overlap_pair_penalty(const State *s, int i, int j) {
    if (!aabb_overlap(&s->aabb[i], &s->aabb[j])) return 0.0;
    if (bounding_circle_reject(s, i, j)) return 0.0;

    const Vec2 *wi = Wc(s, i);
    const Vec2 *wj = Wc(s, j);
    const AABB *ai = TAc(s, i);
    const AABB *aj = TAc(s, j);

    double pen = 0.0;
    for (int ta = 0; ta < NTRI; ta++) {
        const AABB *aTa = &ai[ta];
        int ai0 = TRIS[ta].a, ai1 = TRIS[ta].b, ai2 = TRIS[ta].c;

        for (int tb = 0; tb < NTRI; tb++) {
            if (!aabb_overlap(aTa, &aj[tb])) continue;
            int bj0 = TRIS[tb].a, bj1 = TRIS[tb].b, bj2 = TRIS[tb].c;
            double depth = 0.0;
            if (tri_sat_penetration_idx(wi, wj, ai0, ai1, ai2, bj0, bj1, bj2, &depth)) {
                pen += depth * depth;
            }
        }
    }

    return pen;
}

double overlap_sum_for_k_grid(const State *s, int k) {
    const Grid *g = &s->grid;
    int cid = g->cell_id[k];
    if (cid < 0) {
        double sum = 0.0;
        for (int j = 0; j < s->N; j++) if (j != k) sum += overlap_pair_penalty(s, k, j);
        return sum;
    }

    int kx = cid % g->nx;
    int ky = cid / g->nx;
    int R = grid_R_cells(s);

    double sum = 0.0;
    for (int dy = -R; dy <= R; dy++) {
        int yy = ky + dy;
        if (yy < 0 || yy >= g->ny) continue;

        for (int dx = -R; dx <= R; dx++) {
            int xx = kx + dx;
            if (xx < 0 || xx >= g->nx) continue;

            int c = grid_index(g, xx, yy);
            for (int j = g->head[c]; j != -1; j = g->next[j]) {
                if (j == k) continue;
                sum += overlap_pair_penalty(s, k, j);
            }
        }
    }
    return sum;
}

double outside_for_k(const State *s, int k) {
    return outside_penalty_aabb(&s->aabb[k], s->L);
}

Totals compute_totals_full_grid(const State *s) {
    Totals t;
    t.overlap_total = 0.0;
    t.out_total = 0.0;

    for (int i = 0; i < s->N; i++) t.out_total += outside_for_k(s, i);

    const Grid *g = &s->grid;
    int R = grid_R_cells(s);

    for (int i = 0; i < s->N; i++) {
        int cid = g->cell_id[i];
        if (cid < 0) continue;

        int ix = cid % g->nx;
        int iy = cid / g->nx;

        for (int dy = -R; dy <= R; dy++) {
            int yy = iy + dy;
            if (yy < 0 || yy >= g->ny) continue;

            for (int dx = -R; dx <= R; dx++) {
                int xx = ix + dx;
                if (xx < 0 || xx >= g->nx) continue;

                int c = grid_index(g, xx, yy);
                for (int j = g->head[c]; j != -1; j = g->next[j]) {
                    if (j <= i) continue;
                    t.overlap_total += overlap_pair_penalty(s, i, j);
                }
            }
        }
    }

    return t;
}

double energy_from_totals(const State *s, const Weights *w, const Totals *t) {
    return w->alpha_L * s->L + w->lambda_ov * t->overlap_total + w->mu_out * t->out_total;
}

double feasibility_metric(const Totals *t) {
    double ov = t->overlap_total;
    double out = t->out_total;
    if (ov < 0.0) ov = 0.0;
    if (out < 0.0) out = 0.0;
    return ov + out;
}

