CC = gcc
CFLAGS = -O3 -std=c11 -Wall -Wextra -Iinclude -fopenmp
LDFLAGS = -lm -fopenmp -ldl

SRCDIR = src
TESTDIR = tests
BINDIR = bin


TRACE ?= 0

SOURCES = $(wildcard $(SRCDIR)/*.c)
EXTRA_SRCS = $(SRCDIR)/bisection.c $(SRCDIR)/warmstart.c $(SRCDIR)/method_ms_stub.c $(SRCDIR)/method_ms.c $(SRCDIR)/method_erms.c $(SRCDIR)/method_pt.c $(SRCDIR)/polish.c
SRCS_REFACT = $(filter-out $(SRCDIR)/HPC_parallel.c $(SRCDIR)/HPC_parallel_monolith.c,$(SOURCES)) $(EXTRA_SRCS)
SRCS_REFACT := $(sort $(SRCS_REFACT))

ifeq ($(TRACE),1)
	CFLAGS += -finstrument-functions
	LDFLAGS += -ldl -rdynamic
endif

.PHONY: all clean test run

all: $(BINDIR)/solver

$(BINDIR):
	mkdir -p $(BINDIR)

$(BINDIR)/solver: | $(BINDIR)
	@echo "Compiling refactored solver from $(SRCS_REFACT)"
	$(CC) $(CFLAGS) $(SRCS_REFACT) -o $@ $(LDFLAGS)

run: all
	@echo "Run: ./bin/solver N trials"
	@echo "Example: ./bin/solver 3 5"

test: $(BINDIR)/test_replica_swap $(BINDIR)/test_rebuild_derived $(BINDIR)/test_single_replica_regress $(BINDIR)/test_ms_parallel $(BINDIR)/test_erms_resample $(BINDIR)/test_pt_swap_math $(BINDIR)/test_pt_quench $(BINDIR)/test_polish_shave
	@./$(BINDIR)/test_replica_swap
	@./$(BINDIR)/test_rebuild_derived
	@./$(BINDIR)/test_single_replica_regress
	@./$(BINDIR)/test_ms_parallel
	@./$(BINDIR)/test_erms_resample
	@./$(BINDIR)/test_pt_swap_math
	@./$(BINDIR)/test_pt_quench
	@./$(BINDIR)/test_polish_shave


$(BINDIR)/test_replica_swap: $(TESTDIR)/test_replica_swap.c $(SRCDIR)/replica.c $(SRCDIR)/base_geometry.c $(SRCDIR)/spatial_hash.c $(SRCDIR)/physics.c $(SRCDIR)/utils.c | $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BINDIR)/test_rebuild_derived: $(TESTDIR)/test_rebuild_derived.c $(SRCDIR)/replica.c $(SRCDIR)/base_geometry.c $(SRCDIR)/spatial_hash.c $(SRCDIR)/physics.c $(SRCDIR)/utils.c | $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BINDIR)/test_single_replica_regress: $(TESTDIR)/test_single_replica_regress.c $(SRCDIR)/annealing.c $(SRCDIR)/replica.c $(SRCDIR)/base_geometry.c $(SRCDIR)/spatial_hash.c $(SRCDIR)/physics.c $(SRCDIR)/utils.c | $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BINDIR)/test_ms_parallel: $(TESTDIR)/test_ms_parallel.c $(SRCDIR)/method_ms.c $(SRCDIR)/annealing.c $(SRCDIR)/replica.c $(SRCDIR)/base_geometry.c $(SRCDIR)/spatial_hash.c $(SRCDIR)/physics.c $(SRCDIR)/utils.c | $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BINDIR)/test_erms_resample: $(TESTDIR)/test_erms_resample.c $(SRCDIR)/method_erms.c $(SRCDIR)/annealing.c $(SRCDIR)/replica.c $(SRCDIR)/base_geometry.c $(SRCDIR)/spatial_hash.c $(SRCDIR)/physics.c $(SRCDIR)/utils.c | $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)


$(BINDIR)/test_pt_swap_math: $(TESTDIR)/test_pt_swap_math.c $(SRCDIR)/method_pt.c $(SRCDIR)/annealing.c $(SRCDIR)/replica.c $(SRCDIR)/base_geometry.c $(SRCDIR)/spatial_hash.c $(SRCDIR)/physics.c $(SRCDIR)/utils.c | $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BINDIR)/test_pt_quench: $(TESTDIR)/test_pt_quench.c $(SRCDIR)/method_pt.c $(SRCDIR)/annealing.c $(SRCDIR)/replica.c $(SRCDIR)/base_geometry.c $(SRCDIR)/spatial_hash.c $(SRCDIR)/physics.c $(SRCDIR)/utils.c | $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(BINDIR)/test_polish_shave: $(TESTDIR)/test_polish_shave.c $(SRCDIR)/polish.c $(SRCDIR)/method_erms.c $(SRCDIR)/annealing.c $(SRCDIR)/replica.c $(SRCDIR)/base_geometry.c $(SRCDIR)/spatial_hash.c $(SRCDIR)/physics.c $(SRCDIR)/utils.c | $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(BINDIR)
