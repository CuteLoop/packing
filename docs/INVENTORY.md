# HPC_DEMO Inventory (Phase 0A)

Scope
-----
Files scanned:
- run/HPC_DEMO/include/*.h
- run/HPC_DEMO/src/*.c

Types and structs
-----------------
From headers:
- run/HPC_DEMO/include/common.h:
  - Vec2
  - AABB
  - Tri
  - Grid
  - Totals
  - Weights
  - State
- run/HPC_DEMO/include/config.h:
  - PhaseParams
- run/HPC_DEMO/include/utils.h:
  - RNG
- run/HPC_DEMO/include/spatial_hash.h:
  - cell_visit_fn (function pointer)

From source-only (file-local):
- run/HPC_DEMO/src/annealing.c:
  - Move (file-local struct)

Globals
-------
- run/HPC_DEMO/src/base_geometry.c:
  - const Tri TRIS[NTRI]
  - const Vec2 BASE_V[NV]
- run/HPC_DEMO/src/logger.c:
  - static FILE *g_csv
  - static char g_prefix[256]
- run/HPC_DEMO/src/main.c:
  - volatile sig_atomic_t g_stop_requested
- run/HPC_DEMO/src/trace.c:
  - static __thread int g_depth
  - static int g_enabled
  - static int g_max_depth
  - static int g_show_unknown

Functions (signature + file + globals read/write)
-------------------------------------------------
run/HPC_DEMO/src/main.c
- void handle_sigterm(int sig)
  - writes: g_stop_requested
- void write_svg(const char *path, State *s, double feas)
  - reads: s
- void write_csv(const char *path, const char *prefix, uint64_t run_id, uint64_t seed, State *s, double feas)
  - reads: s
- void usage(void)
- int main(int argc, char **argv)
  - reads: g_stop_requested

run/HPC_DEMO/src/annealing.c
- static Move propose_move(State *s, const Weights *w, RNG *rng, double step_xy, double step_th)
- static void undo_move(State *s, const Move *m)
- static void run_phase(State *s, Totals *t, Weights *w, RNG *rng, const PhaseParams *pp, double *step_xy, double *step_th)
- double try_pack_at_current_L(State *s, RNG *rng, const PhaseParams *A, const PhaseParams *B, int trials,
  uint64_t seed, uint64_t run_id, double *out_cx, double *out_cy, double *out_th, int verbose)

run/HPC_DEMO/src/physics.c
- int tri_sat_penetration_idx(const Vec2 *wi, const Vec2 *wj, int ai0, int ai1, int ai2, int bj0, int bj1, int bj2, double *depth_out)
  - reads: TRIS (via indices), BASE_V indirectly through State world
- int bounding_circle_reject(const State *s, int i, int j)
- double overlap_pair_penalty(const State *s, int i, int j)
  - reads: TRIS
- double overlap_sum_for_k_grid(const State *s, int k)
- double outside_for_k(const State *s, int k)
- Totals compute_totals_full_grid(const State *s)
- double energy_from_totals(const State *s, const Weights *w, const Totals *t)
- double feasibility_metric(const Totals *t)

run/HPC_DEMO/src/base_geometry.c
- void build_world_verts(Vec2 *world, double cx, double cy, double theta)
  - reads: BASE_V
- double base_polygon_area(void)
  - reads: BASE_V
- double base_bounding_radius(void)
  - reads: BASE_V
- void update_instance(State *s, int i)
  - reads: BASE_V, TRIS

run/HPC_DEMO/src/spatial_hash.c
- int grid_index(const Grid *g, int ix, int iy)
- void grid_cell_xy(const Grid *g, double x, double y, int *ix, int *iy)
- void grid_init(Grid *g, int N, double L, double cell)
- void grid_free(Grid *g)
- void grid_insert(Grid *g, int i, double x, double y)
- void grid_remove(Grid *g, int i)
- void grid_update(Grid *g, int i, double x, double y)
- void grid_rebuild(Grid *g, int N, double L, double cell, const double *cx, const double *cy)
- int grid_R_cells(const State *s)
- void aabb_to_cell_range(const Grid *g, const AABB *b, int *ix0, int *iy0, int *ix1, int *iy1)
- void grid_query_neighbors(const Grid *g, const State *s, int k, cell_visit_fn visit, void *ctx)

run/HPC_DEMO/src/utils.c
- void rng_seed(RNG *rng, uint64_t seed)
- double rng_u01(RNG *rng)
- double rng_uniform(RNG *rng, double a, double b)
- double now_seconds(void)
- int file_exists(const char *path)
- void ensure_dir(const char *name)
- int streq_wrapper(const char *a, const char *b)
- uint64_t make_trial_seed(uint64_t base_seed, uint64_t run_id, uint64_t trial_id)
- double wrap_angle_0_2pi(double th)

run/HPC_DEMO/src/logger.c
- int logger_init(const char *prefix)
  - writes: g_prefix, g_csv
- void logger_log_trial(int trial_id, const State *s, double feas)
  - reads: g_csv
- int logger_write_snapshot(const char *path, const State *s, double feas)
- void logger_flush(void)
  - reads: g_csv
- void logger_close(void)
  - writes: g_csv

run/HPC_DEMO/src/trace.c
- static int trace_enabled(void)
  - reads/writes: g_enabled, g_max_depth, g_show_unknown
- static void trace_print(const char *event, void *func)
  - reads: g_depth, g_show_unknown
- void __cyg_profile_func_enter(void *func, void *caller)
  - writes: g_depth
- void __cyg_profile_func_exit(void *func, void *caller)
  - writes: g_depth

run/HPC_DEMO/src/geometry.c
- (no functions; includes geometry.h)

Gate 0A note
------------
Mutable state in annealing.c, physics.c, spatial_hash.c is tracked through State and Grid fields and the globals listed above.
