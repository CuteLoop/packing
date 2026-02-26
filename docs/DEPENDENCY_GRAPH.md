# HPC_DEMO Dependency Graph (Phase 0A)

File-level call graph (best-effort)
-----------------------------------
- main.c
  - calls: write_svg, write_csv, try_pack_at_current_L
  - uses: base_polygon_area, base_bounding_radius, build_world_verts, update_instance
  - uses: grid_init, grid_free, grid_rebuild
  - uses: rng_seed, rng_uniform, wrap_angle_0_2pi, ensure_dir
  - uses: logger_init, logger_log_trial, logger_close
- annealing.c
  - calls: overlap_sum_for_k_grid, outside_penalty_aabb, update_instance, grid_update
  - calls: compute_totals_full_grid, feasibility_metric
  - calls: grid_rebuild
  - calls: rng_u01, rng_uniform, rng_seed, make_trial_seed, wrap_angle_0_2pi
- physics.c
  - calls: grid_R_cells, grid_index
  - uses: aabb_overlap, outside_penalty_aabb (from common.h)
  - uses: TRIS, BASE_V (from base_geometry.c via extern)
- base_geometry.c
  - defines: BASE_V, TRIS
  - provides: build_world_verts, base_polygon_area, base_bounding_radius, update_instance
- spatial_hash.c
  - provides: grid_* functions
- utils.c
  - provides: RNG, file and math helpers
- logger.c
  - provides: log CSV output helpers
- trace.c
  - provides: call-tracing hooks (optional build)

Module dependencies (by feature area)
-------------------------------------
- Geometry module
  - base_geometry.c provides BASE_V/TRIS and geometry transforms.
  - geometry.h is consumed by main.c, annealing.c, physics.c.
- Spatial hash module
  - spatial_hash.c consumed by main.c, annealing.c, physics.c.
- Physics module
  - physics.c consumed by annealing.c and main.c.

High-level dependency chain
---------------------------
- main.c -> annealing.c -> physics.c -> spatial_hash.c
- main.c -> geometry.c/base_geometry.c
- annealing.c -> geometry.c/base_geometry.c
- physics.c -> geometry.c/base_geometry.c
- all modules use types from common.h

Notes
-----
- base_geometry.c is the only source of global constants BASE_V and TRIS.
- logger.c and trace.c are self-contained utilities and do not affect solver physics.
