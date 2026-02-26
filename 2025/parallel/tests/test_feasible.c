#include <assert.h>
#include "solve_feasible.h"
#include "bounds.h"
#include "energy.h"

int main(void) {
    const size_t N = 9;
    const double r = 1.0;

    const double Llo = lower_bound_basic(N, r);
    const double Lhi = upper_bound_grid(N, r);

    FeasibleSolveResult res = solve_feasible_bisect(
        N, r,
        Llo, Lhi,
        1e-6,
        1e-2,
        20,
        0.0,
        2.0, 0.9995,
        20000, 0.30,
        123,
        2
    );

    assert(res.success == 1);
    assert(res.L_best >= 2.0*r - 1e-9);
    assert(res.L_best <= Lhi + 1e-9);

    assert(res.X_best != NULL);
    assert(is_feasible(res.X_best, N, r, res.L_best, 1e-8) == 1);

    feasible_free(&res);
    return 0;
}
