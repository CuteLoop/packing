#include "anneal_joint.h"
#include "rng.h"
#include "metropolis.h"
#include "energy.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void* xcalloc(size_t n, size_t sz) {
    return calloc(n, sz);
}

static double energy_total_joint(const Vec2* X, size_t N, double r, double L, double alpha) {
    return energy_pair(X, N, r) + energy_wall(X, N, r, L) + alpha * L;
}

JointAnnealResult anneal_joint_run(
    const Vec2* X0, size_t N,
    double L0, double r, double alpha,
    double T0, double gamma, size_t n_steps,
    double sigma_x, double sigma_L,
    double p_move_x,
    unsigned long seed
) {
    JointAnnealResult out;
    memset(&out, 0, sizeof(out));
    out.N = N;

    if (N == 0 || !X0) return out;
    if (p_move_x < 0.0) p_move_x = 0.0;
    if (p_move_x > 1.0) p_move_x = 1.0;
    if (L0 < 2.0 * r) L0 = 2.0 * r;

    Vec2* X = (Vec2*)xcalloc(N, sizeof(Vec2));
    Vec2* Xbest = (Vec2*)xcalloc(N, sizeof(Vec2));
    if (!X || !Xbest) { free(X); free(Xbest); return out; }
    memcpy(X, X0, N * sizeof(Vec2));
    memcpy(Xbest, X0, N * sizeof(Vec2));

    JointTrace tr;
    memset(&tr, 0, sizeof(tr));
    tr.n_steps = n_steps;
    tr.E = (double*)xcalloc(n_steps, sizeof(double));
    tr.L = (double*)xcalloc(n_steps, sizeof(double));
    tr.T = (double*)xcalloc(n_steps, sizeof(double));
    tr.accepted = (int*)xcalloc(n_steps, sizeof(int));
    tr.move_type = (int*)xcalloc(n_steps, sizeof(int));
    tr.moved_idx = (int*)xcalloc(n_steps, sizeof(int));
    if (!tr.E || !tr.L || !tr.T || !tr.accepted || !tr.move_type || !tr.moved_idx) {
        free(X); free(Xbest);
        free(tr.E); free(tr.L); free(tr.T);
        free(tr.accepted); free(tr.move_type); free(tr.moved_idx);
        return out;
    }

    RNG rng;
    rng_seed(&rng, seed);

    double L = L0;
    double T = T0;
    double E = energy_total_joint(X, N, r, L, alpha);

    double Ebest = E;
    double Lbest = L;

    size_t n_accept = 0;

    for (size_t t = 0; t < n_steps; ++t) {
        tr.T[t] = T;
        tr.E[t] = E;
        tr.L[t] = L;
        tr.accepted[t] = 0;
        tr.moved_idx[t] = -1;

        const double u = rng_u01(&rng);

        if (u < p_move_x) {
            tr.move_type[t] = 0;

            const size_t idx = (size_t)(rng_u01(&rng) * (double)N);
            Vec2 newp = X[idx];
            newp.x += sigma_x * rng_normal(&rng);
            newp.y += sigma_x * rng_normal(&rng);

            if (newp.x < 0.0) newp.x = 0.0;
            if (newp.x > L)   newp.x = L;
            if (newp.y < 0.0) newp.y = 0.0;
            if (newp.y > L)   newp.y = L;

            const double dE = delta_energy_move_one(X, N, idx, newp, r, L);
            const int acc = metropolis_accept(dE, T, &rng);

            tr.moved_idx[t] = (int)idx;

            if (acc) {
                X[idx] = newp;
                E += dE;
                tr.accepted[t] = 1;
                n_accept++;

                if (E < Ebest) {
                    Ebest = E;
                    Lbest = L;
                    memcpy(Xbest, X, N * sizeof(Vec2));
                }
            }
        } else {
            tr.move_type[t] = 1;

            double Lp = L + sigma_L * rng_normal(&rng);
            if (Lp < 2.0 * r) Lp = 2.0 * r;

            const double Ew = energy_wall(X, N, r, L);
            const double Ewp = energy_wall(X, N, r, Lp);
            const double dE = (Ewp - Ew) + alpha * (Lp - L);

            const int acc = metropolis_accept(dE, T, &rng);
            if (acc) {
                L = Lp;
                E += dE;
                tr.accepted[t] = 1;
                n_accept++;

                if (E < Ebest) {
                    Ebest = E;
                    Lbest = L;
                    memcpy(Xbest, X, N * sizeof(Vec2));
                }
            }
        }

        T *= gamma;
    }

    out.X_best = Xbest;
    out.L_best = Lbest;
    out.E_best = Ebest;
    out.accept_rate = (n_steps > 0) ? ((double)n_accept / (double)n_steps) : 0.0;
    out.trace = tr;

    free(X);
    return out;
}

void anneal_joint_free(JointAnnealResult* res) {
    if (!res) return;
    free(res->X_best);
    res->X_best = NULL;

    free(res->trace.E);
    free(res->trace.L);
    free(res->trace.T);
    free(res->trace.accepted);
    free(res->trace.move_type);
    free(res->trace.moved_idx);

    memset(&res->trace, 0, sizeof(res->trace));
}
