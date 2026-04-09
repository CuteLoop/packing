#ifndef POLISH_H
#define POLISH_H

#include "methods.h"

/* Runs MS-polish at R=20 on a fixed L_best.
 * Includes a stochastic shave if no improvement is seen for stall_threshold_sec.
 *
 * cfg              – study config (R will be forced to 20 internally)
 * L_best           – fixed side length to polish at
 * total_budget_sec – wall-clock budget for the entire polish run
 * stall_threshold_sec – seconds without improvement before triggering shave
 * shave_budget_sec – duration of a single shave slice (mu_out = 0)
 * use_warm_start   – if nonzero, seed replicas from init_state
 * init_state       – warm-start state (may be NULL if use_warm_start == 0)
 * out_res          – populated with the best strictly feasible state found
 */
void runner_polish(StudyConfig *cfg, double L_best, double total_budget_sec,
                   double stall_threshold_sec, double shave_budget_sec,
                   int use_warm_start, const ReplicaState *init_state,
                   SliceResult *out_res);

#endif
