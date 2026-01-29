#include "accept.h"
#include <stdlib.h>
#include <math.h>

bool accept_proposal(double deltaE, double beta) {
    if (deltaE <= 0.0) return true;
    double p = exp(-beta * deltaE);
    double r = (double)rand() / (double)RAND_MAX;
    return (r < p);
}
