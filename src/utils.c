
#include "../include/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>
#include <errno.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void rng_seed(RNG *rng, uint64_t seed) {
    uint64_t x = seed;
    rng->s = splitmix64(&x);
    if (rng->s == 0) rng->s = 0xdeadbeefULL;
}

static uint64_t xorshift64star(RNG *rng) {
    uint64_t x = rng->s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

double rng_u01(RNG *rng) {
    return (double)(xorshift64star(rng) >> 11) * (1.0 / 9007199254740992.0);
}

double rng_uniform(RNG *rng, double a, double b) {
    return a + (b - a) * rng_u01(rng);
}

double now_seconds(void) { return (double)time(NULL); }

int file_exists(const char *path) {
    FILE *f = fopen(path, "r"); if (f) { fclose(f); return 1; } return 0;
}

void ensure_dir(const char *name) {
    if (mkdir(name, 0755) == 0) return;
    if (errno == EEXIST) return;
}

static int streq_impl(const char *a, const char *b) { return strcmp(a, b) == 0; }

int streq_wrapper(const char *a, const char *b) { return streq_impl(a,b); }

uint64_t make_trial_seed(uint64_t base_seed, uint64_t run_id, uint64_t trial_id) {
    uint64_t x = base_seed;
    x ^= 0xD1B54A32D192ED03ULL * (run_id + 0x9e3779b97f4a7c15ULL);
    x ^= 0x94D049BB133111EBULL * (trial_id + 0xBF58476D1CE4E5B9ULL);
    return splitmix64(&x);
}

double wrap_angle_0_2pi(double th) {
    double two = 2.0 * M_PI;
    th = fmod(th, two);
    if (th < 0.0) th += two;
    return th;
}

