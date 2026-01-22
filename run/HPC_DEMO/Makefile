CC = gcc
CFLAGS = -O3 -std=c11 -Wall -Wextra -Iinclude
LDFLAGS = -lm

SRCDIR = src
BINDIR = bin


SOURCES = $(wildcard $(SRCDIR)/*.c)
SRCS_REFACT = $(filter-out $(SRCDIR)/HPC_parallel.c $(SRCDIR)/HPC_parallel_monolith.c,$(SOURCES))

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

# Simple test target (builds a SAT unit test if present)
test: $(BINDIR)/test_sat

$(BINDIR)/test_sat: | $(BINDIR)
	@echo "No tests provided yet. Create tests/test_sat.c to enable this target."
	@false

clean:
	rm -rf $(BINDIR)
