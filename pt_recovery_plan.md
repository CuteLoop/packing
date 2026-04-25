# Parallel Tempering (PT) Recovery Plan

## 1. Study-mode Dispatch Path
- Confirmed: `--study --method pt` dispatches to `runner_pt` via function pointer in main.c.
- PT is actually running its intended code path in study mode.

## 2. PT Initialization
- Replicas are allocated as an array (no aliasing).
- Each replica is initialized with a unique seed.
- Temperatures are set by `build_temp_ladder` (monotone, geometric ladder).
- Each replica gets its own temperature and is initialized independently.

## 3. Swap Logic
- Swaps are attempted in a single-threaded OpenMP block, alternating parity.
- Acceptance probability is computed as `pt_swap_accept_prob(Ei, Ej, Ti, Tj)` (mathematically correct).
- Random test is `if (rng_u01(&swap_rng) < acc)` (correct).
- `swap_attempts++` for every attempted swap.
- `swap_accepts++` only for accepted swaps.

### Potential Bugs to Check
- If all temperatures are equal, swaps always accepted.
- If all energies are equal, swaps always accepted.
- If temperature ladder is degenerate, swaps always accepted.
- If state evolution is broken and all replicas are identical, swaps always accepted.

## 4. Synchronization
- Swaps are performed in a single-threaded OpenMP block (no race conditions).
- Parity alternates each swap phase (adjacent pairs swapped in alternating order).

## 5. Debug Instrumentation Plan
- Add `#ifdef PT_DEBUG` block inside swap loop to print detailed info for the first swap of each probe.
- Add per-probe summary at the end of each probe under `#ifdef PT_DEBUG`.

## 6. Next Steps
- Add `#define PT_DEBUG 1` at the top of `method_pt.c` (or use `-DPT_DEBUG` in the Makefile for temporary debugging).
- Print all relevant swap variables for the first swap in each probe.
- Print per-probe summary: attempts, accepts, acceptance rate, min/max/mean energies, number of distinct energies, coldest replica energy, and feasibility.

## Deliverables
- Fix only clearly identified PT bugs.
- Keep changes isolated to PT.
- Do not touch MS or ER-MS behavior.
- Provide a concise audit summary in comments or a short markdown note.
