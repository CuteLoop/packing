#ifndef LOGGER_H
#define LOGGER_H

#include "common.h"

// Simple logger API: initialize with prefix, log per-trial CSV rows, snapshots, and close.
int logger_init(const char *prefix);
void logger_log_trial(int trial_id, const State *s, double feas);
int logger_write_snapshot(const char *path, const State *s, double feas);
void logger_flush(void);
void logger_close(void);

#endif // LOGGER_H
