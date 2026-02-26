#include <assert.h>
#include <math.h>
#include "bounds.h"
#include "energy.h"

int main(void) {
    const size_t N = 20;
    const double r = 1.0;

    double Llo = lower_bound_basic(N, r);
    double Lhi = upper_bound_grid(N, r);

    assert(Llo >= 2.0*r - 1e-12);
    assert(Lhi >= Llo - 1e-12);

    Vec2 X[20];
    init_grid(X, N, r, Lhi);

    assert(is_feasible(X, N, r, Lhi, 1e-12) == 1);

    return 0;
}
