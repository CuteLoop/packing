#include "../include/methods.h"
#include "../include/bisection.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static StudyConfig make_test_cfg(int N, int R, uint64_t seed) {
    StudyConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.N = N;
    cfg.R = R;
    cfg.seed = seed;
    cfg.run_id = 0;
    cfg.eps_feas = 1e-6;
    cfg.weights.alpha_L = 0.0;
    cfg.weights.lambda_ov = 1.0;
    cfg.weights.mu_out = 1.0;
    strncpy(cfg.method, "pt", sizeof(cfg.method) - 1);
    return cfg;
}

int main(void) {
    printf("=== PT Quench Tests ===\n");
    StudyConfig cfg = make_test_cfg(3, 4, 12345);
    SliceResult res;
    runner_pt(&cfg, 200.0, 5.0, 0, NULL, &res);
    assert(res.has_state == 1);
    assert(!isnan(res.min_energy));
    printf("feasible=%d\n", res.feasible);
    printf("=== PT Quench tests passed ===\n");
    return 0;
}
