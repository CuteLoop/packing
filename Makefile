CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -pedantic
LDFLAGS = -lm
BIN_DIR = bin

SRCS_SA = sa_pack_shrink/sa_pack_shrink.c
SRCS_SA_POLY = sa_pack_poly_shrink/sa_pack_shrink_poly.c
SRCS_HPC = HPC/hpc_parallel.c
SRCS_CMAES = cmaes/cmaes_pack_poly.c
SRCS_VANILLA = vanilla_sa/sa_pack.c

ALL_TARGETS = \
	$(BIN_DIR)/sa_pack_shrink \
	$(BIN_DIR)/sa_pack_shrink_poly \
	$(BIN_DIR)/hpc_parallel \
	$(BIN_DIR)/cmaes_pack_poly \
	$(BIN_DIR)/sa_pack

.PHONY: all clean run help

all: $(BIN_DIR) $(ALL_TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/sa_pack_shrink: $(SRCS_SA) | $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/sa_pack_shrink_poly: $(SRCS_SA_POLY) | $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/hpc_parallel: $(SRCS_HPC) | $(BIN_DIR)
	$(CC) -O3 -march=native $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/cmaes_pack_poly: $(SRCS_CMAES) | $(BIN_DIR)
	$(CC) -O3 -march=native $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BIN_DIR)/sa_pack: $(SRCS_VANILLA) | $(BIN_DIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

run: all
	@echo "Running demo: ./bin/sa_pack_shrink"
	./bin/sa_pack_shrink

clean:
	rm -rf $(BIN_DIR)

help:
	@echo "Makefile targets:"
	@echo "  make         -> build all binaries into ./bin/" 
	@echo "  make run     -> build and run ./bin/sa_pack_shrink (default demo)"
	@echo "  make clean   -> remove ./bin/"

# Test targets
TEST_DIR = tests

.PHONY: test

test: CFLAGS_TEST = -Irun/HPC_DEMO/include
test: CFLAGS_TEST = -Irun/HPC_DEMO/include
test: $(TEST_DIR)/test_utils $(TEST_DIR)/test_geometry $(TEST_DIR)/test_spatial_hash $(TEST_DIR)/test_aabb $(TEST_DIR)/test_io

# Coverage build / report
.PHONY: coverage coverage-clean

coverage-clean:
	rm -rf coverage.info out-coverage

coverage: coverage-clean
	@echo "Building tests with coverage flags and running them..."
	$(MAKE) clean
	$(MAKE) test CFLAGS="$(CFLAGS) -O0 -g -fprofile-arcs -ftest-coverage" CFLAGS_TEST="$(CFLAGS_TEST)"
	# Run all tests to generate .gcda files
	./tests/test_geometry || true
	./tests/test_utils || true
	./tests/test_spatial_hash || true
	./tests/test_aabb || true
	./tests/test_io || true
	# Capture coverage (requires lcov/genhtml)
	@if command -v lcov >/dev/null 2>&1; then \
		lcov --capture --directory . --output-file coverage.info || true; \
	else \
		echo "lcov not installed; skipping capture. Install lcov to generate coverage reports."; \
		exit 0; \
	fi
	@if command -v genhtml >/dev/null 2>&1; then \
		genhtml coverage.info --output-directory out-coverage || true; \
		echo "Coverage report generated at out-coverage/index.html"; \
	else \
		echo "genhtml not installed; coverage.info produced. Install genhtml to generate HTML report."; \
		exit 0; \
	fi


$(TEST_DIR)/test_utils: $(TEST_DIR)/test_utils.c run/HPC_DEMO/src/utils.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CFLAGS_TEST) $< run/HPC_DEMO/src/utils.c -o $@ $(LDFLAGS)

$(TEST_DIR)/test_geometry: $(TEST_DIR)/test_geometry.c run/HPC_DEMO/src/base_geometry.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CFLAGS_TEST) $< run/HPC_DEMO/src/base_geometry.c -o $@ $(LDFLAGS)

$(TEST_DIR)/test_spatial_hash: $(TEST_DIR)/test_spatial_hash.c run/HPC_DEMO/src/spatial_hash.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CFLAGS_TEST) $< run/HPC_DEMO/src/spatial_hash.c -o $@ $(LDFLAGS)

$(TEST_DIR)/test_aabb: $(TEST_DIR)/test_aabb.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CFLAGS_TEST) $< -o $@ $(LDFLAGS)

$(TEST_DIR)/test_io: $(TEST_DIR)/test_io.c run/HPC_DEMO/src/utils.c | $(BIN_DIR)
	$(CC) $(CFLAGS) $(CFLAGS_TEST) $< run/HPC_DEMO/src/utils.c -o $@ $(LDFLAGS)


