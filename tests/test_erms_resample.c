#include "../include/methods.h"
#include "../include/replica.h"
#include "../include/bisection.h"
#include "../include/utils.h"
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
    strncpy(cfg.method, "erms", sizeof(cfg.method) - 1);
    return cfg;
}

static void test_no_crash_r1(void) {
    printf("test_no_crash_r1... ");
    StudyConfig cfg = make_test_cfg(10, 1, 11111);
    SliceResult res;
    runner_erms(&cfg, 100.0, 1.0, 0, NULL, &res);
    assert(res.has_state);
    assert(!isnan(res.min_energy));
    assert(res.resample_events == 0);
    printf("PASS\n");
}

static void test_no_crash_r8(void) {
    printf("test_no_crash_r8... ");
    StudyConfig cfg = make_test_cfg(10, 8, 22222);
    SliceResult res;
    runner_erms(&cfg, 100.0, 1.0, 0, NULL, &res);
    assert(res.has_state);
    assert(!isnan(res.min_energy));
    printf("PASS (energy=%.4e resample_events=%d)\n",
           res.min_energy, res.resample_events);
}

static void test_resample_occurs(void) {
    printf("test_resample_occurs... ");
    StudyConfig cfg = make_test_cfg(10, 8, 33333);
    cfg.eps_feas = -1.0;
    SliceResult res;
    runner_erms(&cfg, 10.0, 2.0, 0, NULL, &res);
    assert(res.has_state);
    assert(res.resample_events > 0);
    printf("resample_events=%d feasible=%d\n", res.resample_events, res.feasible);
}

static void test_clone_rng_diverges(void) {
    printf("test_clone_rng_diverges... ");
    int N = 10;
    ReplicaState r1, r2;
    replica_init_random(&r1, N, 50.0, 44444ULL, 1.0, 0);
    replica_clone(&r2, &r1, 55555ULL);
    assert(r1.rng_s != r2.rng_s);
    printf("PASS\n");
}

static void test_easy_feasibility(void) {
    printf("test_easy_feasibility... ");
    StudyConfig cfg = make_test_cfg(10, 4, 55555);
    SliceResult res;
    runner_erms(&cfg, 200.0, 2.0, 0, NULL, &res);
    assert(res.feasible == 1);
    printf("PASS\n");
}

static void test_warm_start(void) {
    printf("test_warm_start... ");
    StudyConfig cfg = make_test_cfg(10, 4, 66666);
    SliceResult res1;
    runner_erms(&cfg, 80.0, 1.5, 0, NULL, &res1);

    SliceResult res2;
    runner_erms(&cfg, 80.0, 1.5, 1, &res1.best_state, &res2);
    assert(res2.has_state);
    assert(!isnan(res2.min_energy));
    printf("PASS (cold=%.4e warm=%.4e)\n", res1.min_energy, res2.min_energy);
}

static void test_no_deadlock_stress(void) {
    printf("test_no_deadlock_stress... ");
    StudyConfig cfg = make_test_cfg(10, 8, 77777);
    SliceResult res;
    runner_erms(&cfg, 50.0, 1.0, 0, NULL, &res);
    assert(res.has_state);
    printf("PASS (resample_events=%d)\n", res.resample_events);
}

int main(void) {
    printf("=== ER-MS Resample Tests (N=10) ===\n");
    test_no_crash_r1();
    test_no_crash_r8();
    test_resample_occurs();
    test_clone_rng_diverges();
    test_easy_feasibility();
    test_warm_start();
    test_no_deadlock_stress();
    printf("=== All ER-MS tests passed ===\n");
    return 0;
}
