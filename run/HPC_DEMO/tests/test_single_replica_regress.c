#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/annealing.h"
#include "../include/common.h"
#include "../include/geometry.h"
#include "../include/spatial_hash.h"
#include "../include/utils.h"
#include "../include/physics.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void init_state(State *s, int N, double L) {
    s->N = N;
    s->L = L;
    s->cx = malloc((size_t)N * sizeof(double));
    s->cy = malloc((size_t)N * sizeof(double));
    s->th = malloc((size_t)N * sizeof(double));
    s->world = malloc((size_t)N * (size_t)NV * sizeof(Vec2));
    s->aabb = malloc((size_t)N * sizeof(AABB));
    s->tri_aabb = malloc((size_t)N * (size_t)NTRI * sizeof(AABB));
    if (!s->cx || !s->cy || !s->th || !s->world || !s->aabb || !s->tri_aabb) {
        fprintf(stderr, "alloc failed\n");
        exit(1);
    }
    s->br = base_bounding_radius();
    grid_init(&s->grid, N, L, s->br * 2.0);
}

static void free_state(State *s) {
    grid_free(&s->grid);
    free(s->cx);
    free(s->cy);
    free(s->th);
    free(s->world);
    free(s->aabb);
    free(s->tri_aabb);
}

static void randomize_pose(State *s, RNG *rng) {
    for (int i = 0; i < s->N; i++) {
        s->cx[i] = rng_uniform(rng, -0.5 * s->L, 0.5 * s->L);
        s->cy[i] = rng_uniform(rng, -0.5 * s->L, 0.5 * s->L);
        s->th[i] = wrap_angle_0_2pi(rng_uniform(rng, 0.0, 2.0 * M_PI));
        update_instance(s, i);
    }
    grid_rebuild(&s->grid, s->N, s->L, s->grid.cell, s->cx, s->cy);
}

static void copy_pose(State *dst, const State *src) {
    for (int i = 0; i < src->N; i++) {
        dst->cx[i] = src->cx[i];
        dst->cy[i] = src->cy[i];
        dst->th[i] = src->th[i];
        update_instance(dst, i);
    }
    grid_rebuild(&dst->grid, dst->N, dst->L, dst->grid.cell, dst->cx, dst->cy);
}

int main(void) {
    const int N = 5;
    const double L = 3.0;
    const uint64_t seed = 12345ULL;
    const uint64_t run_id = 7ULL;

    State s0;
    State s_old;
    State s_new;
    init_state(&s0, N, L);
    init_state(&s_old, N, L);
    init_state(&s_new, N, L);

    RNG rng_init;
    rng_seed(&rng_init, 9999ULL);
    randomize_pose(&s0, &rng_init);

    copy_pose(&s_old, &s0);
    copy_pose(&s_new, &s0);

    PhaseParams pp = {
        .iters = 5000,
        .T_start = 1.0, .T_end = 1e-5,
        .step_xy_start = 0.05, .step_th_start = 0.5,
        .adapt_window = 2000, .acc_low = 0.4, .acc_high = 0.6,
        .step_shrink = 0.95, .step_grow = 1.05,
        .step_xy_min = 1e-5, .step_xy_max = 2.0,
        .step_th_min = 1e-4, .step_th_max = M_PI,
        .lambda_start = 1.0, .lambda_max = 1e6,
        .mu_start = 1.0, .mu_max = 1e6,
        .ramp_every = 5000, .ramp_factor = 2.0,
        .log_every = 0
    };

    RNG rng_old;
    rng_seed(&rng_old, seed);
    (void)try_pack_at_current_L_old(&s_old, &rng_old, &pp, &pp, 1, seed, run_id, NULL, NULL, NULL, 0);

    RNG rng_new;
    rng_seed(&rng_new, seed);
    (void)try_pack_at_current_L(&s_new, &rng_new, &pp, &pp, 1, seed, run_id, NULL, NULL, NULL, 0);

    Totals t_old = compute_totals_full_grid(&s_old);
    Totals t_new = compute_totals_full_grid(&s_new);

    double tol = 1e-12;
    assert(isfinite(t_old.overlap_total) && isfinite(t_old.out_total));
    assert(isfinite(t_new.overlap_total) && isfinite(t_new.out_total));
    assert(fabs(t_old.overlap_total - t_new.overlap_total) <= tol);
    assert(fabs(t_old.out_total - t_new.out_total) <= tol);

    free_state(&s0);
    free_state(&s_old);
    free_state(&s_new);

    printf("test_single_replica_regress ok\n");
    return 0;
}
