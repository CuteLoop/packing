Here is the updated Work Queue, strictly aligned with the deliverables and acceptance criteria from the revised Engineering Plan.

Key updates include:

    P2: Added explicit offset arrays to the InstanceCache struct.

    P5: Clarified the "Repair" phase after squeezing.

    P7: Added the "Restart near elite" logic (perturbation) to the migration task.

Work Queue — Chimera Nonconvex Packing Solver
P0: Stabilize benchmark (Phase 0)

    [ ] Add CLI parsing to tools/collision_benchmark.c (--instances, --pairs, --seed, --mode, --check_agreement K).

    [ ] Implement timing instrumentation: separate cache_build, broadphase, narrowphase.

    [ ] Critical: Implement DEBUG_ORACLE mode that exists non-zero on first SAT/Tri mismatch.

    [ ] Create make bench and make bench-debug targets.

    [ ] Verify check_agreement passes for K=5000 random poses.

P1: Refactor into modules (Phase 1)

    [ ] Create src/geom_vec2.h, src/aabb.h (move inline ops here).

    [ ] Create src/shape_tree.c/.h (make_tree_poly_local + explicit free).

    [ ] Create src/triangulate_earclip.c/.h (ear clipping + helper memory management).

    [ ] Create src/convex_decomp.c/.h (tri merge, local axes build).

    [ ] Create src/instance_cache.c/.h (define InstanceCache struct + build_all + free).

    [ ] Create src/collide_sat.c/.h (cached SAT + baseline SAT).

    [ ] Create src/collide_tri_oracle.c/.h (guarded by #ifdef DEBUG_ORACLE).

    [ ] Refactor tools/collision_benchmark.c to use new modules.

P2: Incremental cache + per-part AABBs (Phase 2)

    [ ] Extend InstanceCache: add partAabb, instAabb, plus vertOffset and axisOffset arrays.

    [ ] Implement cache_build_all: compute all AABBs and offsets.

    [ ] Implement cache_update_one(D, C, i, newPose): update world verts/axes and AABBs for instance i.

    [ ] Test: Implement test_cache_update_one_matches_rebuild() (correctness check).

    [ ] Update collide_cached_convex: prune sub-parts using partAabb overlap.

P3: Grid hash broadphase (Phase 3)

    [ ] Implement src/grid_hash.c/.h: grid_init, grid_build_all, grid_update_one.

    [ ] Implement grid_query_candidates: must return sorted unique IDs (for determinism).

    [ ] Test: Add brute-force check test_grid_no_false_negatives_smallN.

P4: Energy + deltaE for move-one (Phase 4)

    [ ] Implement src/energy.c/.h: outside_penalty_aabb, energy_full (debug).

    [ ] Implement delta_energy_move_one: calculate ΔE using neighbor candidates.

    [ ] Implement src/propose.c/.h: translate/rotate proposals with sigma schedule.

    [ ] Implement src/accept.c/.h: greedy + Metropolis (beta schedule).

    [ ] Test: Add test_deltaE_matches_full_smallN (verify incremental energy accuracy).

P5: Island solver loop (Relax/Squeeze/Kick) (Phase 5)

    [ ] Implement src/solver_island.c/.h: relax_steps, try_squeeze, kick, best_L tracking.

    [ ] Implement "Repair" phase: short relaxation run immediately after try_squeeze.

    [ ] Add CSV logger per rank: time, iter, L, E, best_L, accept_rate.

    [ ] Create CLI binary chimera_island (single-rank entry point).

P6: OpenMP batch proposals (Phase 6)

    [ ] Implement src/proposal_batch.c/.h: Batch size B, parallel eval, commit-one policy.

    [ ] Ensure tie-breaking is deterministic (e.g., lowest index wins ties).

    [ ] Benchmark: Measure proposals/sec and verify speedup vs. serial loop.

P7: MPI elite migration (Phase 7)

    [ ] Implement src/mpi_migration.c/.h: MPI_Allgather for best_L comparison.

    [ ] Implement elite broadcast: winner sends pose array to all ranks.

    [ ] Implement restart_near_elite: losers reset state to elite + small perturbation.

    [ ] Integrate into chimera_mpi main with --mig_seconds.

    [ ] Create Slurm scripts (single-node and multi-node).

P8: GPU (Conditional Phase 8)

    [ ] Add profiler counters: measure % time in collide_sat vs rest.

    [ ] If justified: Implement CUDA kernel for batch ΔE evaluation.