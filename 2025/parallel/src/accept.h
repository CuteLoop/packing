#ifndef ACCEPT_H
#define ACCEPT_H

#include <stdbool.h>

// Metropolis-Hastings acceptance criterion.
// deltaE: E_new - E_old
// beta: Inverse temperature (1/T). High beta = greedy. Low beta = random.
// Returns true if move should be accepted.
bool accept_proposal(double deltaE, double beta);

#endif
