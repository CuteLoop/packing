    # Phase 7 Draft: Chimera Sweep Edition (P100, Brute Force)

    ## Goal
    Generate a full **packing density curve** by sweeping $N=1..200$ on a Tesla P100 using the **brute‑force $O(n^2)$ kernel**, which is stable and fast enough at this scale. Spatial hashing is intentionally **not used** in this phase.

    ## Core Architecture
    - **Physics:** brute‑force collision/overlap checks (GPU).
    - **Logic:** Parallel Tempering (Replica Exchange) to escape local minima.
    - **Memory:** Hall of Fame (HOF) to archive top solutions.
    - **Stability:** Smart shrink with a **theoretical minimum** (area‑based) to avoid crashes.

    ## Phase Breakdown

    ### Phase 7.0 — Sweep Orchestrator
    **Objective:** Run a sweep from `N=start` to `N=end` and log `(N, final L, density)`.

    **Tasks**
    - Add a sweep runner that calls a single‑run function for each `N`.
    - Log results to `chimera_sweep.csv`.

    **Acceptance Criteria**
    - A CSV with `N,Final_L,Density` for all `N` in range.

    ---

    ### Phase 7.1 — Theoretical Floor (Safety)
    **Objective:** Prevent the box size from shrinking below physically feasible limits.

    **Logic**
    - Compute total area $A = N \cdot A_{poly}$ (use `poly_area(tree)` when available).
    - Set $L_{min} = \max(\sqrt{A}, L_{geom})$.
    - Only shrink when best energy indicates near‑feasible configuration.

    **Acceptance Criteria**
    - No crashes at small `L`.
    - Box size never drops below $L_{min}$.

    ---

    ### Phase 7.2 — Parallel Tempering (Replica Exchange)
    **Objective:** Swap temperatures across chains to escape local minima.

    **Tasks**
    - Maintain a temperature ladder `T[i]` on host.
    - Every 10 epochs, perform greedy swap: if hot chain energy is better, swap temps.

    **Acceptance Criteria**
    - Observed temperature swaps; energy trajectory shows step‑wise improvements.

    ---

    ### Phase 7.3 — Hall of Fame (HOF)
    **Objective:** Preserve top solutions and re‑inject them later.

    **Tasks**
    - Track `HOF_SIZE` solutions by energy + fingerprint uniqueness.
    - Every 100 epochs, inject a random HOF solution into a random chain.

    **Acceptance Criteria**
    - HOF updates logged.
    - Lazarus injections observed in logs.

    ---

    ## Implementation Notes

    ### Sweep Runner (Pseudo)
    ```
    for N in [start..end]:
        run_sweep_step(N, epochs)
        log N, final_L, density
    ```

    ### Density
    ```
    density = total_area / (L * L)
    ```

    ---

    ## Risks & Mitigations
    - **Risk:** Sweep runtime too long.
    - **Mitigation:** Use smaller `epochs` for larger `N` or early stop if no progress.

    - **Risk:** Over‑aggressive shrinking.
    - **Mitigation:** Only shrink when best energy is near‑feasible; enforce $L_{min}$.

    ---

    ## Deliverables
    - `chimera_sweep.csv` with full curve.
    - Stable sweep runner with logging.

    ---

    ## Acceptance Criteria (Summary)
    - Sweep completes for the chosen range.
    - No crashes as `L` shrinks.
    - Density curve produced for analysis.
