/*
 * main.c — Collision benchmark driver.
 *
 * Usage:
 *   ./bench [reps] [out.csv]
 *
 * Runs the full scaling study:
 *   N ∈ {10, 20, 50, 100, 200}  ×  regime ∈ {sparse, dense, jammed}
 *   ×  method ∈ {A, B, C}
 *
 * Outputs:
 *   - out.csv (default: bench_results.csv) — all metrics for every run
 *   - stdout summary table
 *
 * reps = number of full pair-loop repetitions per (N, regime, method).
 * Default reps is chosen so each cell takes at least ~0.5 s.
 */

#include "bench.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Scaling study parameters ────────────────────────────────── */
static int N_VALUES[]  = { 10, 20, 50, 100, 200 };
static int N_COUNT     = 5;

static PlaceMode MODES[]       = { PLACE_SPARSE, PLACE_DENSE, PLACE_JAMMED };
static const char *MODE_NAMES[]= { "sparse",     "dense",     "jammed"     };
static int MODE_COUNT          = 3;

/* Default repetitions per cell (override on command line) */
#define DEFAULT_REPS 4

/* ── CSV ─────────────────────────────────────────────────────── */
void csv_write_header(FILE *f) {
    fprintf(f,
        "N,regime,method,"
        "pairs_total,broadphase_pass,narrow_calls,collisions,"
        "wall_time_s,mpairs_per_s,"
        /* Method C extras (0 for A and B) */
        "part_pairs,part_aabb_pass,"
        "hint_tests,hint_rejects,"
        "sat_full_calls,sat_hits,sat_axes_tested,avg_axes_per_sat\n"
    );
}

void csv_write_row(FILE *f, int N, const char *regime, const char *method,
                  const BenchStats *s, const CachedStats *cs)
{
    double avg_axes = (cs && cs->sat_full_calls > 0)
        ? (double)cs->sat_axes_tested / cs->sat_full_calls : 0.0;
    fprintf(f,
        "%d,%s,%s,"
        "%lld,%lld,%lld,%lld,"
        "%.6f,%.4f,"
        "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.2f\n",
        N, regime, method,
        s->pairs_total, s->broadphase_pass, s->narrow_calls, s->collisions,
        s->wall_time_s, s->mpairs_per_s,
        cs ? cs->part_pairs    : 0LL,
        cs ? cs->part_aabb_pass: 0LL,
        cs ? cs->hint_tests    : 0LL,
        cs ? cs->hint_rejects  : 0LL,
        cs ? cs->sat_full_calls: 0LL,
        cs ? cs->sat_hits      : 0LL,
        cs ? cs->sat_axes_tested:0LL,
        avg_axes
    );
}

/* ── pretty-print one row ────────────────────────────────────── */
static void print_row(int N, const char *regime, const char *method,
                      const BenchStats *s, const CachedStats *cs)
{
    double bp_pct = (s->pairs_total > 0)
        ? 100.0 * s->broadphase_pass / s->pairs_total : 0.0;
    double col_pct = (s->broadphase_pass > 0)
        ? 100.0 * s->collisions / s->broadphase_pass : 0.0;

    printf("  N=%3d  %-7s  %s  | pairs=%6lld  bp_pass=%5.1f%%  "
           "narrow=%6lld  coll=%5.1f%%  | t=%.3fs  %.2f Mpairs/s",
           N, regime, method,
           s->pairs_total, bp_pct,
           s->narrow_calls, col_pct,
           s->wall_time_s, s->mpairs_per_s);

    if (cs && cs->hint_tests > 0) {
        double hr_pct = 100.0 * cs->hint_rejects / cs->hint_tests;
        double avg    = (cs->sat_full_calls > 0)
            ? (double)cs->sat_axes_tested / cs->sat_full_calls : 0.0;
        printf("  | hint_rej=%4.1f%%  avg_axes=%.1f", hr_pct, avg);
    }
    printf("\n");
}

/* ── dynamic reps selection (target ~0.5 s per cell) ─────────── */
static int choose_reps(int N, int base_reps) {
    /* Fewer reps for large N (cost is O(N^2)) */
    if (N >= 200) return base_reps > 1 ? 1 : base_reps;
    if (N >= 100) return base_reps > 2 ? 2 : base_reps;
    return base_reps;
}

/* ── main ───────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    int   base_reps = DEFAULT_REPS;
    const char *csv_path = "bench_results.csv";

    if (argc >= 2) base_reps = atoi(argv[1]);
    if (argc >= 3) csv_path  = argv[2];

    printf("=== Collision Benchmark ===\n");
    printf("reps (base)=%d  output=%s\n\n", base_reps, csv_path);

    /* Build polygon */
    PolyDef poly;
    poly_build_tree(&poly);

    /* Open CSV */
    FILE *csv = fopen(csv_path, "w");
    if (!csv) { fprintf(stderr, "Cannot open %s\n", csv_path); return 1; }
    csv_write_header(csv);

    /* Pose buffer (max N = 512) */
    Pose *poses = (Pose *)calloc(MAX_N, sizeof(Pose));

    /* ── Scaling study ─────────────────────────────────────────── */
    for (int ni = 0; ni < N_COUNT; ni++) {
        int N = N_VALUES[ni];
        int reps = choose_reps(N, base_reps);
        printf("─── N = %d  (reps=%d) ───────────────────────────────\n",
               N, reps);

        for (int mi = 0; mi < MODE_COUNT; mi++) {
            PlaceMode mode = MODES[mi];
            const char *mname = MODE_NAMES[mi];
            RNG rng = { 0xDEADBEEF ^ (uint64_t)N ^ ((uint64_t)mi << 32) };
            double L;
            placement_generate(poses, N, &poly, &rng, mode, &L);
            printf("  [%s]  L=%.3f\n", mname, L);

            /* Method A */
            {
                BenchStats s = bench_method_a(&poly, poses, N, reps);
                print_row(N, mname, "A", &s, NULL);
                csv_write_row(csv, N, mname, "A", &s, NULL);
                fflush(csv);
            }
            /* Method B */
            {
                BenchStats s = bench_method_b(&poly, poses, N, reps);
                print_row(N, mname, "B", &s, NULL);
                csv_write_row(csv, N, mname, "B", &s, NULL);
                fflush(csv);
            }
            /* Method C */
            {
                CachedStats cs = {0};
                BenchStats s = bench_method_c(&poly, poses, N, reps, &cs);
                print_row(N, mname, "C", &s, &cs);
                csv_write_row(csv, N, mname, "C", &s, &cs);
                fflush(csv);
            }
            printf("\n");
        }
    }

    /* ── Speedup summary ──────────────────────────────────────── */
    printf("=== Speedup summary (C vs B, dense regime) ===\n");
    printf("Rerun with --speedup flag or read bench_results.csv "
           "for per-cell ratios.\n\n");
    printf("See analyze.py for automated plots and tables.\n");

    fclose(csv);
    free(poses);
    return 0;
}
