#include "energy.h"
#include <math.h>

static inline double pos(double t) { return (t > 0.0) ? t : 0.0; }
static inline double sq(double a) { return a * a; }

static inline double phi_overlap(double d, double two_r)
{
    /* phi(d) = max(0, 2r - d)^2 */
    double t = pos(two_r - d);
    return t * t;
}

double energy_pair(const Vec2* X, size_t N, double r)
{
    if (!X || N == 0) return 0.0;
    double two_r = 2.0 * r;
    double E = 0.0;

    for (size_t i = 0; i < N; ++i) {
        for (size_t j = i + 1; j < N; ++j) {
            double dx = X[i].x - X[j].x;
            double dy = X[i].y - X[j].y;
            double d  = hypot(dx, dy);
            E += phi_overlap(d, two_r);
        }
    }
    return E;
}

double energy_wall(const Vec2* X, size_t N, double r, double L)
{
    if (!X || N == 0) return 0.0;

    /* Wall penalty uses [r, L-r] as interior window. */
    double lo = r;
    double hi = L - r;

    double E = 0.0;
    for (size_t i = 0; i < N; ++i) {
        double x = X[i].x;
        double y = X[i].y;

        E += sq(pos(lo - x));
        E += sq(pos(x - hi));
        E += sq(pos(lo - y));
        E += sq(pos(y - hi));
    }
    return E;
}

double energy_total(const Vec2* X, size_t N, double r, double L, double alpha)
{
    /* For your “E >= 0 always” inquiry to be true, require alpha >= 0 and L >= 0. */
    if (alpha < 0.0 || L < 0.0 || r < 0.0) return NAN;
    return energy_pair(X, N, r) + energy_wall(X, N, r, L) + alpha * L;
}

double delta_energy_move_one(
    const Vec2* X, size_t N, size_t i, Vec2 new_pos,
    double r, double L
)
{
    /* Exact delta for moving only X[i] (L fixed, so alpha*L cancels). */
    if (!X || N == 0) return 0.0;

    double lo = r;
    double hi = L - r;
    double two_r = 2.0 * r;

    /* old wall contribution for i */
    double oldx = X[i].x, oldy = X[i].y;
    double old_wall = 0.0;
    old_wall += sq(pos(lo - oldx));
    old_wall += sq(pos(oldx - hi));
    old_wall += sq(pos(lo - oldy));
    old_wall += sq(pos(oldy - hi));

    /* new wall contribution for i */
    double new_wall = 0.0;
    new_wall += sq(pos(lo - new_pos.x));
    new_wall += sq(pos(new_pos.x - hi));
    new_wall += sq(pos(lo - new_pos.y));
    new_wall += sq(pos(new_pos.y - hi));

    /* old pair links involving i */
    double old_pair = 0.0;
    double new_pair = 0.0;

    for (size_t j = 0; j < N; ++j) {
        if (j == i) continue;

        /* old distance */
        double dxo = oldx - X[j].x;
        double dyo = oldy - X[j].y;
        double do_ = hypot(dxo, dyo);
        old_pair += phi_overlap(do_, two_r);

        /* new distance */
        double dxn = new_pos.x - X[j].x;
        double dyn = new_pos.y - X[j].y;
        double dn  = hypot(dxn, dyn);
        new_pair += phi_overlap(dn, two_r);
    }

    /* Each pair term involving i appears exactly once in energy_pair (i<j).
       The sum over j!=i counts each of those terms once, so it matches. */
    return (new_pair - old_pair) + (new_wall - old_wall);
}

int is_feasible(const Vec2* X, size_t N, double r, double L, double eps) {
    if (!X) return 0;
    if (N == 0) return 1;
    if (L < 2.0 * r) return 0;
    if (eps < 0.0) eps = 0.0;

    double ep = energy_pair(X, N, r);
    if (ep > eps) return 0;

    double ew = energy_wall(X, N, r, L);
    if (ew > eps) return 0;

    return 1;
}
