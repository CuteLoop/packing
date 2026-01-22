#include <assert.h>
#include <stdlib.h>
#include "anneal_joint.h"
#include "bounds.h"

int main(void) {
    const size_t N = 16;
    const double r = 1.0;

    double L0 = upper_bound_grid(N, r);
    Vec2* X0 = (Vec2*)calloc(N, sizeof(Vec2));
    assert(X0);
    init_grid(X0, N, r, L0);

    JointAnnealResult jr = anneal_joint_run(
        X0, N,
        L0, r, 0.05,
        2.0, 0.9999, 2000,
        0.30, 0.05 * L0,
        0.90,
        123
    );

    assert(jr.X_best != NULL);
    assert(jr.L_best >= 2.0 * r - 1e-12);
    assert(jr.trace.n_steps == 2000);

    for (size_t t = 0; t < jr.trace.n_steps; ++t) {
        assert(jr.trace.L[t] >= 2.0 * r - 1e-12);
    }

    anneal_joint_free(&jr);
    free(X0);
    return 0;
}
