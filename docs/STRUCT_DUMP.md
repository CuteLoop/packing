# Phase 0A.5 — Struct Dump (verbatim)

Source: run/HPC_DEMO/include/common.h

```c
// Constants for the base polygon
#define NV 15
#define NTRI 13

typedef struct { double x, y; } Vec2;

typedef struct { double minx, miny, maxx, maxy; } AABB;

typedef struct { int a, b, c; } Tri;

// Grid structure used by spatial hash (also exposed via State.grid in some translations)
typedef struct {
	double L;
	double cell;
	int nx, ny;
	double half;

	int *head;
	int *next;
	int *prev;
	int *cell_id;
	int N;
} Grid;

// Totals returned by energy computations
typedef struct {
	double overlap_total;
	double out_total;
} Totals;

typedef struct {
	double alpha_L;
	double lambda_ov;
	double mu_out;
} Weights;

// Shared packing `State` and helpers (moved here so modules compile)
typedef struct {
	int N;
	double L;

	double *cx;
	double *cy;
	double *th;

	Vec2  *world;     // N*NV
	AABB  *aabb;      // N
	AABB  *tri_aabb;  // N*NTRI

	double br;

	// Grid (declared here so spatial module and monolith share the same type)
	Grid grid;
} State;
```

Source: run/HPC_DEMO/include/utils.h

```c
typedef struct { uint64_t s; } RNG;
```

Source: run/HPC_DEMO/include/config.h

```c
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
```

Source: run/HPC_DEMO/src/annealing.c

```c
typedef struct {
    int k;
    double old_cx, old_cy, old_th;
    double dE;
    double d_ov;
    double d_out;
} Move;
```
