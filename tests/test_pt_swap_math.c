#include "../include/methods.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

static void test_swap_accept_basic(void) {
    double Ei = 10.0;
    double Ej = 12.0;
    double Ti = 1.0;
    double Tj = 2.0;

    double acc = pt_swap_accept_prob(Ei, Ej, Ti, Tj);
    double expected = exp((1.0 / Ti - 1.0 / Tj) * (Ej - Ei));
    if (expected > 1.0) expected = 1.0;
    assert(fabs(acc - expected) < 1e-12);
}

static void test_swap_accept_always(void) {
    double Ei = 10.0;
    double Ej = 10.0;
    double Ti = 1.0;
    double Tj = 2.0;
    double acc = pt_swap_accept_prob(Ei, Ej, Ti, Tj);
    assert(fabs(acc - 1.0) < 1e-12);
}

int main(void) {
    printf("=== PT Swap Math Tests ===\n");
    test_swap_accept_basic();
    test_swap_accept_always();
    printf("=== All PT Swap Math tests passed ===\n");
    return 0;
}
