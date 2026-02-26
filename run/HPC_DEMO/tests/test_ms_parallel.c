#include "../include/methods.h"
#include "../include/bisection.h"
#include "../include/utils.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>

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
    strncpy(cfg.method, "ms", sizeof(cfg.method) - 1);
    return cfg;
}

static void test_r1_no_crash(void) {
    printf("test_r1_no_crash... ");
    StudyConfig cfg = make_test_cfg(5, 1, 12345);
    SliceResult res;
    double L = 100.0;
    runner_ms(&cfg, L, 5.0, 0, NULL, &res);

    assert(res.has_state == 1);
    assert(!isnan(res.min_energy));
    assert(!isnan(res.min_feas));
    assert(res.slice_used_sec >= 0.0);
    assert(res.slice_used_sec <= 6.0);
    printf("PASS\n");
}

static void test_r4_no_crash(void) {
    printf("test_r4_no_crash... ");
    StudyConfig cfg = make_test_cfg(10, 4, 54321);
    SliceResult res;
    double L = 100.0;
    runner_ms(&cfg, L, 5.0, 0, NULL, &res);

    assert(res.has_state == 1);
    assert(!isnan(res.min_energy));
    printf("feasible=%d min_energy=%.4e  ", res.feasible, res.min_energy);
    printf("PASS\n");
}

static void test_easy_feasibility(void) {
    printf("test_easy_feasibility... ");
    StudyConfig cfg = make_test_cfg(3, 4, 99999);
    SliceResult res;
    double L = 200.0;
    runner_ms(&cfg, L, 10.0, 0, NULL, &res);

    assert(res.feasible == 1);
    assert(res.has_state == 1);
    assert(res.best_state.is_feasible == 1);
    printf("PASS (energy=%.4e)\n", res.best_state.energy);
}

static void test_deterministic_selection(void) {
    printf("test_deterministic_selection... ");
    StudyConfig cfg = make_test_cfg(10, 4, 77777);
    SliceResult res1, res2;
    double L = 50.0;

    runner_ms(&cfg, L, 3.0, 0, NULL, &res1);
    runner_ms(&cfg, L, 3.0, 0, NULL, &res2);

    assert(res1.best_state.replica_id == res2.best_state.replica_id);
    double ediff = fabs(res1.min_energy - res2.min_energy);
    printf("replica_id=%d ediff=%.4e  ", res1.best_state.replica_id, ediff);
    printf("PASS\n");
}

static void test_warm_start(void) {
    printf("test_warm_start... ");
    StudyConfig cfg = make_test_cfg(5, 2, 11111);
    double L = 80.0;

    SliceResult res1;
    runner_ms(&cfg, L, 3.0, 0, NULL, &res1);

    SliceResult res2;
    runner_ms(&cfg, L, 3.0, 1, &res1.best_state, &res2);

    assert(res2.has_state == 1);
    assert(!isnan(res2.min_energy));
    printf("cold_E=%.4e warm_E=%.4e  ", res1.min_energy, res2.min_energy);
    printf("PASS\n");
}

int main(void) {
    omp_set_num_threads(4);
    printf("=== MS Parallel Runner Tests ===\n");
    test_r1_no_crash();
    test_r4_no_crash();
    test_easy_feasibility();
    test_deterministic_selection();
    test_warm_start();
    printf("=== All tests passed ===\n");
    return 0;
}
