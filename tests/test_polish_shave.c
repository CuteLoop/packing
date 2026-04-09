#include "../include/polish.h"
#include "../include/utils.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static StudyConfig make_test_cfg(int N, uint64_t seed) {
    StudyConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.N = N;
    cfg.seed = seed;
    cfg.run_id = 0;
    cfg.eps_feas = 1e-6;
    cfg.weights.alpha_L = 0.0;
    cfg.weights.lambda_ov = 1.0;
    cfg.weights.mu_out = 1.0;
    strncpy(cfg.method, "erms", sizeof(cfg.method) - 1);
    return cfg;
}

static void test_shave_triggers_and_restores(void) {
    printf("test_shave_triggers_and_restores... ");
    StudyConfig cfg = make_test_cfg(10, 99999);
    SliceResult res;

    /* Budget: 3.0s, Stall threshold: 0.5s, Shave budget: 0.3s
     * This guarantees the shave triggers at least once. */
    runner_polish(&cfg, 50.0, 3.0, 0.5, 0.3, 0, NULL, &res);

    /* Assert the weight was strictly restored */
    assert(cfg.weights.mu_out == 1.0);
    /* Assert we got some valid state back */
    assert(res.has_state);

    printf("PASS (final_energy=%.4e feasible=%d)\n", res.min_energy, res.feasible);
}

static void test_polish_preserves_feasible(void) {
    printf("test_polish_preserves_feasible... ");
    StudyConfig cfg = make_test_cfg(10, 88888);
    SliceResult res;

    /* Large L makes feasibility easy; polish should preserve it */
    runner_polish(&cfg, 200.0, 2.0, 10.0, 0.5, 0, NULL, &res);

    assert(res.has_state);
    assert(res.feasible == 1);
    assert(!isnan(res.min_energy));

    printf("PASS (energy=%.4e)\n", res.min_energy);
}

static void test_polish_r_restored(void) {
    printf("test_polish_r_restored... ");
    StudyConfig cfg = make_test_cfg(10, 77777);
    cfg.R = 4;
    SliceResult res;

    runner_polish(&cfg, 50.0, 1.0, 10.0, 0.5, 0, NULL, &res);

    /* R must be restored to caller's value, not left at 20 */
    assert(cfg.R == 4);
    assert(res.has_state);

    printf("PASS\n");
}

int main(void) {
    printf("=== Polish & Shave Tests ===\n");
    test_shave_triggers_and_restores();
    test_polish_preserves_feasible();
    test_polish_r_restored();
    printf("=== All Polish tests passed ===\n");
    return 0;
}
