#ifndef CONFIG_H
#define CONFIG_H

typedef struct {
    int iters;
    double T_start;
    double T_end;
    int adapt_window;
    double acc_low;
    double acc_high;
    double step_xy_start;
    double step_th_start;
    double step_shrink;
    double step_grow;
    double step_xy_min;
    double step_xy_max;
    double step_th_min;
    double step_th_max;
    double lambda_start;
    double mu_start;
    int ramp_every;
    double ramp_factor;
    double lambda_max;
    double mu_max;
    double p_reinsert;
    double p_rotmix;
    int log_every;
} PhaseParams;

#endif // CONFIG_H
