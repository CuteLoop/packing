// PT method targeted tests
// Only tests PT swap logic and reporting, not full packing correctness
// To be compiled with the rest of the test suite

#include "../include/methods.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// 1. PT swap acceptance counter sanity
typedef struct {
    double Ei, Ej, Ti, Tj;
    double rng_val;
    int expect_accept;
} SwapTestCase;

void test_pt_swap_acceptance_counter() {
    SwapTestCase cases[] = {
        {1.0, 0.0, 1.0, 2.0, 0.0, 1}, // acc=1, rng=0, accept
        {1.0, 0.0, 1.0, 2.0, 0.99, 1}, // acc=1, rng=0.99, accept
        {1.0, 2.0, 1.0, 2.0, 0.5, 0}, // acc<1, rng>acc, reject
    };
    for (int i = 0; i < 3; ++i) {
        double acc = pt_swap_accept_prob(cases[i].Ei, cases[i].Ej, cases[i].Ti, cases[i].Tj);
        int accepted = (cases[i].rng_val < acc) ? 1 : 0;
        assert((accepted == cases[i].expect_accept) && "Swap accept logic failed");
    }
    printf("test_pt_swap_acceptance_counter passed\n");
}

// 2. PT rejection test
void test_pt_rejection() {
    double Ei = 10.0, Ej = -10.0, Ti = 1.0, Tj = 100.0;
    double acc = pt_swap_accept_prob(Ei, Ej, Ti, Tj);
    assert(acc < 0.1 && "Acceptance should be small");
    printf("test_pt_rejection passed\n");
}

// 3. PT guaranteed-accept case
void test_pt_guaranteed_accept() {
    double Ei = 0.0, Ej = 0.0, Ti = 1.0, Tj = 2.0;
    double acc = pt_swap_accept_prob(Ei, Ej, Ti, Tj);
    assert(fabs(acc - 1.0) < 1e-12 && "Acceptance should be 1");
    printf("test_pt_guaranteed_accept passed\n");
}

// 4. Temperature ladder test
void test_temp_ladder() {
    double temps[8];
    build_temp_ladder(temps, 8, 1.0, 10.0);
    for (int i = 1; i < 8; ++i) {
        assert(temps[i] > temps[i-1] && "Ladder not strictly monotone");
    }
    for (int i = 0; i < 7; ++i) {
        assert(fabs(temps[i] - temps[i+1]) > 1e-8 && "Duplicate temperatures");
    }
    printf("test_temp_ladder passed\n");
}

// 5. Study-mode PT integration smoke test
// This is a shell-out test, not C. See scripts/test_pt_study_smoke.sh

int main() {
    test_pt_swap_acceptance_counter();
    test_pt_rejection();
    test_pt_guaranteed_accept();
    test_temp_ladder();
    printf("All PT unit tests passed.\n");
    return 0;
}
