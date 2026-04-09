#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>

typedef struct { uint64_t s; } RNG;

void rng_seed(RNG *rng, uint64_t seed);
double rng_u01(RNG *rng);

double rng_uniform(RNG *rng, double a, double b);
double now_seconds(void);
int file_exists(const char *path);
void ensure_dir(const char *name);

// helper wrappers exported from utils.c
uint64_t make_trial_seed(uint64_t base_seed, uint64_t run_id, uint64_t trial_id);
int streq_wrapper(const char *a, const char *b);

#define streq(a,b) streq_wrapper((a),(b))

double wrap_angle_0_2pi(double th);

#endif // UTILS_H
