#include "anneal.h"
#include "energy.h"
#include "rng.h"
#include "io.h"

#include <stdio.h>
#include <stdlib.h>

static void usage(const char* prog)
{
    fprintf(stderr,
        "Usage:\n"
        "  %s N L r alpha T0 gamma n_steps sigma seed out_prefix\n\n"
        "Example:\n"
        "  %s 10 10 1 0 1 0.999 200000 0.25 123 out/smoke\n",
        prog, prog);
}

int main(int argc, char** argv)
{
    /* Parameters (10 args after program name) */
    if (argc != 11) {
        usage(argv[0]);
        return 2;
    }

    size_t N       = (size_t)strtoull(argv[1], NULL, 10);
    double L       = atof(argv[2]);
    double r       = atof(argv[3]);
    double alpha   = atof(argv[4]);
    double T0      = atof(argv[5]);
    double gamma   = atof(argv[6]);
    size_t n_steps = (size_t)strtoull(argv[7], NULL, 10);
    double sigma   = atof(argv[8]);
    uint64_t seed  = (uint64_t)strtoull(argv[9], NULL, 10);
    const char* prefix = argv[10];

    if (N == 0 || L <= 0.0 || r < 0.0 || gamma <= 0.0 || gamma >= 1.0 || n_steps == 0 || sigma < 0.0) {
        fprintf(stderr, "Invalid parameters.\n");
        return 2;
    }

    /* Initialize X0 (better baseline): uniform in [r, L-r]^2 if possible,
       otherwise fall back to [0, L]^2. */
    Vec2* X0 = (Vec2*)malloc(N * sizeof(Vec2));
    if (!X0) { perror("malloc X0"); return 1; }

    RNG rng;
    rng_seed(&rng, seed);

    double lo = 0.0, span = L;
    if (L - 2.0 * r > 0.0) {
        lo = r;
        span = L - 2.0 * r;
    }

    for (size_t i = 0; i < N; ++i) {
        X0[i].x = lo + span * rng_u01(&rng);
        X0[i].y = lo + span * rng_u01(&rng);
    }

    /* Run SA */
    AnnealResult res = anneal_run(X0, N, L, r, alpha, T0, gamma, n_steps, sigma, seed);

    /* Output paths */
    char trace_path[512], best_csv_path[512], best_svg_path[512];
    snprintf(trace_path, sizeof(trace_path), "%s_trace.csv", prefix);
    snprintf(best_csv_path, sizeof(best_csv_path), "%s_best.csv", prefix);
    snprintf(best_svg_path, sizeof(best_svg_path), "%s_best.svg", prefix);

    if (write_trace_csv(trace_path, &res.trace) != 0) {
        fprintf(stderr, "Failed to write %s\n", trace_path);
    }
    if (write_centers_csv(best_csv_path, res.X_best, N) != 0) {
        fprintf(stderr, "Failed to write %s\n", best_csv_path);
    }
    if (write_best_svg(best_svg_path, res.X_best, N, L, r) != 0) {
        fprintf(stderr, "Failed to write %s\n", best_svg_path);
    }

    printf("Done.\n");
    printf("  E_best = %.17g\n", res.E_best);
    printf("  accept_rate = %.6f\n", res.accept_rate);
    printf("  wrote: %s\n", trace_path);
    printf("  wrote: %s\n", best_csv_path);
    printf("  wrote: %s\n", best_svg_path);

    anneal_free_result(&res);
    free(X0);
    return 0;
}
