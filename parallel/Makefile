CC=gcc
CFLAGS=-O2 -Wall -Wextra -pedantic -Iinclude
LDFLAGS=-lm

NVCC=nvcc
NVFLAGS=-O3 -Isrc -arch=sm_60

SRC_BASE=src/rng.c src/energy.c src/propose.c src/metropolis.c src/anneal.c

# extra sources for solvers
SRC_EXTRA=src/bounds.c src/solve_feasible.c src/anneal_joint.c

SRC=$(SRC_BASE) $(SRC_EXTRA)

TESTS=tests/test_rng tests/test_energy tests/test_propose tests/test_metropolis tests/test_anneal tests/test_bounds tests/test_feasible tests/test_joint

all: test



bench: collision_benchmark.c src/shape_tree.c src/triangulate_earclip.c src/convex_decomp.c src/collide_tri_oracle.c src/collide_sat.c src/instance_cache.c src/grid_hash.c src/energy.c src/propose.c src/accept.c
	$(CC) -O2 -std=c11 collision_benchmark.c src/shape_tree.c src/triangulate_earclip.c src/convex_decomp.c src/collide_tri_oracle.c src/collide_sat.c src/instance_cache.c src/grid_hash.c src/energy.c src/propose.c src/accept.c -lm -o bench



bench-debug: CFLAGS += -DDEBUG_ORACLE -g
bench-debug: collision_benchmark.c src/shape_tree.c src/triangulate_earclip.c src/convex_decomp.c src/collide_tri_oracle.c src/collide_sat.c src/instance_cache.c src/grid_hash.c src/energy.c src/propose.c src/accept.c
	$(CC) $(CFLAGS) -std=c11 collision_benchmark.c src/shape_tree.c src/triangulate_earclip.c src/convex_decomp.c src/collide_tri_oracle.c src/collide_sat.c src/instance_cache.c src/grid_hash.c src/energy.c src/propose.c src/accept.c -lm -o bench-debug

# Build runnable SA runner
sa_run: $(SRC) src/io.c src/main.c
	$(CC) $(CFLAGS) $(SRC) src/io.c src/main.c -o sa_run $(LDFLAGS)

tests/test_rng: $(SRC) tests/test_rng.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

tests/test_energy: $(SRC) tests/test_energy.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

tests/test_propose: $(SRC) tests/test_propose.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

tests/test_metropolis: $(SRC) tests/test_metropolis.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

tests/test_anneal: $(SRC) tests/test_anneal.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

tests/test_bounds: $(SRC) tests/test_bounds.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

tests/test_feasible: $(SRC) tests/test_feasible.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

tests/test_joint: $(SRC) tests/test_joint.c
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

test: $(TESTS)
	@set -e; \
	for t in $(TESTS); do \
	  echo "RUN $$t"; \
	  ./$$t; \
	done; \
	echo "ALL TESTS PASSED"

# ---------------- GPU build (Chimera) ----------------
GPU_OBJS=src/main.o src/gpu_memory.o src/gpu_rng.o src/solver_kernel.o src/gpu_geometry.o \
	 src/geometry_bake.o src/convex_decomp.o src/shape_tree.o src/triangulate_earclip.o src/rng.o

chimera_run: $(GPU_OBJS)
	$(NVCC) $(NVFLAGS) -o chimera_run $(GPU_OBJS) -lm

src/main.o: src/main.c
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/%.o: src/%.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TESTS) src/*.o chimera_run
