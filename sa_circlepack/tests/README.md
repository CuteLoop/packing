
---

## `tests/README.md`

```md
# `tests/` — Test Guide (TDD for Simulated Annealing Circle Packing)

This directory contains unit tests designed to enforce a strict workflow:

**Theory → Algorithmic Invariant → Unit Test → Code**

Each test is tagged implicitly by its intent:
- **Mathematical invariant**: derived from the model definition (energy, constraints, Metropolis rule).
- **Statistical/probabilistic check**: validates Monte Carlo behavior approximately, with tolerances.
- **Engineering/regression**: reproducibility, deterministic traces, memory ownership, API contracts.

All tests are deterministic: they use explicit seeds and fixed parameters.

## How to run

From repository root:

```bash
make test
