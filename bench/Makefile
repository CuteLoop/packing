# Makefile — Collision benchmark
#
# Targets:
#   make          → build optimised binary ./bench
#   make debug    → build with sanitizers
#   make run      → build and run full study
#   make quick    → build and run with reps=1 (smoke test)
#   make clean

CC      = gcc
CFLAGS_BASE = -std=c99 -Wall -Wextra -Wpedantic
CFLAGS_OPT  = -O3 -march=native -ffast-math
CFLAGS_DBG  = -O0 -g -fsanitize=address,undefined

SRCS = poly.c placement.c methods.c main.c
OBJS = $(SRCS:.c=.o)

LDFLAGS = -lm

# ── Default: optimised ────────────────────────────────────────
.PHONY: all
all: bench

bench: $(SRCS) bench.h
	$(CC) $(CFLAGS_BASE) $(CFLAGS_OPT) -o $@ $(SRCS) $(LDFLAGS)
	@echo "Built ./bench (optimised)"

# ── Debug build ───────────────────────────────────────────────
.PHONY: debug
debug: CFLAGS = $(CFLAGS_BASE) $(CFLAGS_DBG)
debug: bench_dbg

bench_dbg: $(SRCS) bench.h
	$(CC) $(CFLAGS_BASE) $(CFLAGS_DBG) -o $@ $(SRCS) $(LDFLAGS)
	@echo "Built ./bench_dbg (debug + sanitizers)"

# ── Run targets ───────────────────────────────────────────────
.PHONY: run quick

run: bench
	./bench 4 bench_results.csv

quick: bench
	./bench 1 bench_quick.csv

# ── Analysis ──────────────────────────────────────────────────
.PHONY: plot
plot:
	python3 analyze.py bench_results.csv

# ── Clean ─────────────────────────────────────────────────────
.PHONY: clean
clean:
	rm -f bench bench_dbg *.o bench_results.csv bench_quick.csv
