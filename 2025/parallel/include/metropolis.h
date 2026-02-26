#ifndef METROPOLIS_H
#define METROPOLIS_H

#include "rng.h"

/* Metropolis criterion:
   accept always if dE <= 0, else accept with prob exp(-dE/T). */
int metropolis_accept(double dE, double T, RNG* rng);

#endif
