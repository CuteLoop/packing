CC=gcc
CFLAGS=-O2 -Wall -Wextra -pedantic -Iinclude
LDFLAGS=-lm

SRC=src/rng.c src/energy.c src/propose.c src/metropolis.c src/anneal.c

TESTS=tests/test_rng tests/test_energy tests/test_propose tests/test_metropolis tests/test_anneal

all: test

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

test: $(TESTS)
	@set -e; \
	for t in $(TESTS); do \
	  echo "RUN $$t"; \
	  ./$$t; \
	done; \
	echo "ALL TESTS PASSED"

clean:
	rm -f $(TESTS)
