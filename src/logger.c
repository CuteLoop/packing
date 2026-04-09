#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/logger.h"
#include "../include/utils.h"

static FILE *g_csv = NULL;
static char g_prefix[256] = "run";

int logger_init(const char *prefix) {
    if (prefix) strncpy(g_prefix, prefix, sizeof(g_prefix)-1);
    char path[512];
    snprintf(path, sizeof(path), "csv/%s_log.csv", g_prefix);
    g_csv = fopen(path, "a");
    if (!g_csv) return 0;
    fprintf(g_csv, "time,trial,L,N,feas\n");
    fflush(g_csv);
    return 1;
}

void logger_log_trial(int trial_id, const State *s, double feas) {
    if (!g_csv) return;
    time_t now = time(NULL);
    char tbuf[64];
    strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
    fprintf(g_csv, "%s,%d,%.10f,%d,%.17g\n", tbuf, trial_id, s->L, s->N, feas);
}

int logger_write_snapshot(const char *path, const State *s, double feas) {
    // reuse main's write_csv/svg if available; fallback to CSV only
    FILE *f = fopen(path, "w");
    if (!f) return 0;
    fprintf(f, "# L=%.17g best_feas=%.17g N=%d\n", s->L, feas, s->N);
    fprintf(f, "i,cx,cy,theta_rad\n");
    for (int i = 0; i < s->N; i++) {
        fprintf(f, "%d,%.17g,%.17g,%.17g\n", i, s->cx[i], s->cy[i], s->th[i]);
    }
    fclose(f);
    return 1;
}

void logger_flush(void) {
    if (g_csv) fflush(g_csv);
}

void logger_close(void) {
    if (g_csv) fclose(g_csv);
    g_csv = NULL;
}
