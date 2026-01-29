# Autonomous Agent Loop Contract

You are authorized to work autonomously on this repository under the following rules:

## Loop
Repeat the following cycle until the current task’s Definition of Done is satisfied:

1. Read the current task description.
2. Make the minimal code changes required.
3. Build the project.
4. Run all relevant tests.
5. If tests fail:
   - Diagnose the failure.
   - Fix the code.
   - Repeat from step 3.
6. If tests pass:
   - Verify acceptance criteria.
   - Mark the task complete.
   - Proceed to the next task in ISSUES.md.

## Constraints
- Do not delete functionality unless explicitly instructed.
- Do not introduce GPU code unless Phase 8 is reached.
- Preserve deterministic behavior when `--deterministic` is enabled.
- Prefer correctness over performance until acceptance criteria are met.

## Logging
- Write a short summary of each completed task in `AGENT_LOG.md`.
- Note any assumptions or uncertainties.

You may continue this loop without further user confirmation.
