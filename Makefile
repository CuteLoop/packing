CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -pedantic
LDFLAGS = -lm
BIN_DIR = bin

# Legacy solver targets removed (2025/ solvers deleted).
# This Makefile is a transitional stub; it will be replaced by
# run/HPC_DEMO/Makefile when HPC_DEMO is promoted to root.

.PHONY: all clean help

all:
	@echo "Legacy targets removed. Use 'make' from run/HPC_DEMO/ to build the solver."

clean:
	rm -rf $(BIN_DIR)

help:
	@echo "Makefile targets:"
	@echo "  make         -> n/a (legacy targets removed; see run/HPC_DEMO/)"
	@echo "  make test    -> run legacy unit tests"
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


