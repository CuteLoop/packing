# LLM Context Pack: Packing Project Forensic Audit (Artifact-Only)

Date: 2026-04-27
Workspace root: C:/Users/ual-laptop/Documents/packing
Scope: strict artifact-only reconstruction of what was run, what data produced reported results, and which claims are supported.

## 1) What this project is

This repository contains:

- A C solver stack under src/ and include/ for polygon packing.
- Study methods including ms, erms, and pt.
- Legacy monolithic HPC artifacts under 2025/old.
- Current SLURM study scripts under scripts/.
- Report assets and paper text under 0report/ and report/.

## 2) Session objective history (high-level)

The work evolved through four phases:

1. Build publication-grade collision pipeline figures.
2. Verify paper wording against implementation behavior.
3. Build a complete claims inventory for reproducibility checks.
4. Run a strict forensic audit of experiment provenance and claim validity.

## 3) Key files touched or used in this session

Figure/report workflow:

- 0report/make_collision_pipeline_fig.py
- 0report/make_report_figs.py
- 0report/0rep.tex
- claims.md

Forensic evidence sources:

- scripts/run_graph_suite.slurm
- scripts/sweep_n100.slurm
- 2025/old/five_node_parallel.slurm
- 2025/old/five_node_worker.sh
- 2025/old/smoke_n005_5nodes_3m.slurm
- 2025/old/smoke_n010_5nodes_3m.slurm
- 2025/old/n020_5nodes_4h.slurm
- 2025/old/n050_5nodes_4h.slurm
- 2025/old/n100_5nodes_4h.slurm
- 2025/old/n200_5nodes_8h.slurm
- 2025/old/old_n25_4h.slurm
- 2025/old/old_n50_4h.slurm
- 2025/old/old_n100_4h.slurm
- 2025/old/old_n200_8h.slurm
- 2025/old/analysis/feasible_run_statistics.csv
- 2025/old/analysis/feasible_n_statistics.csv
- 2025/old/logs/*_history_log.csv
- out/*/*_bisection.csv

## 4) Pipeline classification (what was actually configured)

### A) Current graph suite pipeline

Source: scripts/run_graph_suite.slurm

- Methods: ms, erms, pt
- Ns: 5, 10, 20, 50, 100
- Per-run budget: time_budget_sec=3600
- Seeds: 1000 and 2000
- Two runs per method/N launched concurrently
- R=10 (thread setting in this script)

### B) Current isolated N=100 pipeline (script is corrupted)

Source: scripts/sweep_n100.slurm

- Intended method set: ms, erms, pt
- N=100 only
- Per-run budget: 3600 seconds
- Seeds 1000 and 2000 concurrently
- Script includes duplicated/broken lines, so treat as potentially corrupted launcher text

### C) Legacy node-labeled launcher lineage (wrapper files named for five nodes)

Sources:

- 2025/old/five_node_parallel.slurm
- 2025/old/five_node_worker.sh
- 2025/old/smoke_n005_5nodes_3m.slurm
- 2025/old/smoke_n010_5nodes_3m.slurm
- 2025/old/n020_5nodes_4h.slurm
- 2025/old/n050_5nodes_4h.slurm
- 2025/old/n100_5nodes_4h.slurm
- 2025/old/n200_5nodes_8h.slurm

Behavior:

- Runs legacy monolithic executable (HPC_parallel_old) via the `five_node_parallel.slurm` / `five_node_worker.sh` launcher path.
- Smoke jobs (N=5,10): TIME_LIMIT=180 sec, RUNS_PER_NODE default 8.
- Main jobs (N=20,50,100): TIME_LIMIT=14100 sec, RUNS_PER_NODE default 64.
- Main N=200: TIME_LIMIT=28500 sec, RUNS_PER_NODE default 64.

Important correction:

- The wrapper filenames and SBATCH headers say `5nodes`, but that should not be treated as proof that the audited N=5/10 results were physically run on five nodes.
- User correction for this audit: the executed run for the small-N lineage was accidentally a one-node run, and the context pack should preserve that correction.
- Therefore the safe provenance label is `node-labeled legacy launcher lineage`, not `confirmed five-node execution`.

### D) Legacy array scripts (task-style lineage)

Sources:

- 2025/old/old_n25_4h.slurm
- 2025/old/old_n50_4h.slurm
- 2025/old/old_n100_4h.slurm
- 2025/old/old_n200_8h.slurm

Behavior:

- Also runs legacy HPC_parallel_old.
- Array sizes and task concurrency:
  - N25: array 1-8%2
  - N50: array 1-6%2
  - N100: array 1-4%2
  - N200: array 1-2%2
- Uses TIME_LIMIT with walltime cushion logic.

## 5) Provenance of best-of-best rows by N

Derived from:

- 2025/old/analysis/feasible_n_statistics.csv
- 2025/old/analysis/feasible_run_statistics.csv

Resolved best-source lineage:

- N=5 -> best log name contains node (node-labeled legacy launcher lineage; user says actual run was one-node, not true five-node)
- N=10 -> best log name contains node (node-labeled legacy launcher lineage; user says actual run was one-node, not true five-node)
- N=25 -> best log name contains task (task-style lineage)
- N=50 -> best log name contains task (task-style lineage)
- N=100 -> best log name contains task (task-style lineage)
- N=200 -> best log name contains task (task-style lineage)

Important: N=5,10 are not from the same pipeline lineage as N>=25. Also, the N=5,10 lineage should not be described as confirmed five-node execution.

## 6) Quantitative run-scale extraction

Computed from artifacts on disk:

Total history logs by N:

- 5: 15
- 10: 15
- 25: 6
- 50: 4
- 100: 3
- 200: 2

Feasible runs by N:

- 5: 15
- 10: 15
- 25: 6
- 50: 4
- 100: 3
- 200: 1

Average first-feasible time (seconds):

- 5: 6.667
- 10: 30.867
- 25: 15.833
- 50: 44.25
- 100: 54.0
- 200: 19.0

Wall-time spread (seconds):

- N5: min 152, median 205, max 233
- N10: min 147, median 241, max 251
- N25: min 13955, median 14012, max 14123
- N50: min 13757, median 13806.5, max 13850
- N100: min 13683, median 13688, max 13688
- N200: min/median/max 28579

## 7) Claim checks requested in audit

### Claim: fixed iteration budget

Verdict: unsupported/false for executed artifact set.

Reason:

- Scripts and logs indicate time-bounded execution (time_budget_sec or time_limit), not a universal fixed iteration count in run artifacts.

### Claim: K=100,000

Verdict: unknown under strict artifact-only run-output constraint.

Reason:

- Not directly encoded in run history/bisection artifacts used for provenance reconstruction.
- May appear in code/docs, but not proven as executed budget in the audited run outputs.

### Claim: M=105,000

Verdict: unknown under strict artifact-only run-output constraint.

Reason:

- Appears in some narrative/guide text, but not established from run artifacts as the governing executed budget.

### Claim: 26 bisection steps

Verdict: not universally supported.

Reason:

- Example bisection traces show different probe counts; at least one graph bisection file has probe_idx 0..9 (10 probes).

### Claim: sub-nanometer precision

Verdict: unsupported.

Reason:

- No artifact-level unit calibration proving nanometer interpretation was found in run logs/CSVs.

### Claim: polish shrinks L

Verdict: supported.

Reason:

- Multiple history logs show repeated polish_improve stages with decreasing L over time.

## 8) Reconstructed actual behavior (artifact-driven)

- Outer execution is budgeted by wall-clock limits.
- Bracketing/bisection occurs, but step counts vary by run and configuration.
- Adaptive polish often contributes substantial post-bisection L improvements.
- Large-N runs often consume near-full allotted time budgets.

## 9) Mixed-provenance risk statement

Final N-series evidence mixes two run lineages:

- Node-labeled legacy launcher lineage for N=5 and N=10. User correction: these were accidentally run as one-node, not true five-node executions.
- Task-style lineage for N=25,50,100,200.

Risk:

- Cross-N comparisons can be confounded by changes in orchestration lineage and run generation context, not just by N scaling.
- Effective sample size decreases sharply at larger N, further weakening comparability.

## 10) Safe vs unsafe narrative claims for paper text

Safer claims to keep:

- Feasible-run counts drop with larger N in audited artifacts.
- Large-N runs are strongly budget/time limited.
- Polish phase contributes meaningful post-feasibility improvements in many runs.

Claims to avoid or rewrite unless newly proven:

- Single homogeneous pipeline provenance across all Ns.
- Any statement that N=5 or N=10 were confirmed five-node executions.
- Universal 26-step bisection claim.
- Universal fixed-iteration-budget claim for audited experiments.
- Unit-specific precision claims (for example, sub-nanometer) without unit calibration evidence.

## 11) Practical handoff guidance for other LLMs

When using this context with another model, ask it to:

1. Keep artifact-only constraints explicit.
2. Separate what is proven by logs/CSVs from what is inferred from code/docs.
3. Treat N=5,10 and N>=25 as mixed provenance unless re-audited and harmonized.
4. Propose exact paper wording that reflects mixed lineage and sample-size limitations.

## 12) Environment note from prior debugging memory

On this Windows machine, .venv_plot originated from Linux and may contain non-executable bin/python stubs.
Use py -3 for local analysis scripts unless a native Windows venv is recreated.
