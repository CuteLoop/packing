#include "energy.h"
#include <assert.h>
#include <math.h>

static int approx0(double x) { return fabs(x) <= 1e-12; }

int main(void)
{
    double r = 1.0, L = 10.0, alpha = 0.1;

    /* feasible interior, well-separated -> pair & wall = 0 */
    Vec2 X[3] = { {2,2}, {5,5}, {8,8} };
    assert(approx0(energy_pair(X, 3, r)));
    assert(approx0(energy_wall(X, 3, r, L)));

    /* total energy >= 0 (alpha>=0) */
    double E = energy_total(X, 3, r, L, alpha);
    assert(E >= 0.0);

    /* overlap case: distance 1 < 2r => (2-1)^2 = 1 */
    Vec2 Y[2] = { {0,0}, {1,0} };
    double Ep = energy_pair(Y, 2, r);
    assert(fabs(Ep - 1.0) <= 1e-12);

    /* delta energy check: E(new)-E(old) equals delta */
    Vec2 Z[4] = { {2,2}, {4,4}, {6,6}, {8,8} };
    size_t i = 1;
    Vec2 newpos = {4.2, 4.1};
    double E0 = energy_total(Z, 4, r, L, alpha);
    double dE = delta_energy_move_one(Z, 4, i, newpos, r, L);
    Vec2 Z2[4];
    for (int k = 0; k < 4; ++k) Z2[k] = Z[k];
    Z2[i] = newpos;
    double E1 = energy_total(Z2, 4, r, L, alpha);
    assert(fabs((E1 - E0) - dE) <= 1e-10);

    return 0;
}
