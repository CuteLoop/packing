# Debug Story: 5-node Old Solver Launch + Feasibility/History Hardening

This document captures the key debug phases, symptoms, root causes, code changes, and verification steps for the old solver pipeline in `2025/old`.

## Scope

- Solver: `hpc_parallel.c`
- Launchers:
  - `five_node_parallel.slurm`
  - `five_node_worker.sh`
  - `one_node_parallel.slurm`
  - wrappers: `n020_5nodes_4h.slurm`, `n050_5nodes_4h.slurm`, `n100_5nodes_4h.slurm`, `n200_5nodes_8h.slurm`, `smoke_n005_5nodes_3m.slurm`, `smoke_n010_5nodes_3m.slurm`
  - post-run tooling: `summarize_old_runs.py`

---

## Phase 1: Compile failures in old solver (`hpc_parallel.c`)

### Symptom

Compile error on HPC:

```text
error: 'g_logger' undeclared (first use in this function)
error: static declaration of 'logger_close' follows non-static declaration
```

### Root cause

- `maybe_stop_and_flush()` referenced `g_logger` before it was declared.
- Inside `maybe_stop_and_flush()` there was `extern void logger_close(void);` which conflicted with later `static void logger_close(void)`.

### Fix

- Moved `Logger` typedef and `g_logger` declaration above `maybe_stop_and_flush()`.
- Removed `extern logger_close` pattern and replaced with direct close/flush of logger handles in signal path.

### Status

- Fixed in `hpc_parallel.c`.

---

## Phase 2: Wrong "best" outputs could include overlap/infeasible states

### Symptom

Smoke test indicated saved outputs could still contain overlap/outside penalties.

### Root cause

Several write paths and snapshot paths updated/wrote state without strict feasibility gate in every location.

### Fixes added in `hpc_parallel.c`

1. Added timestamped history snapshots with labels:
   - stage labels (`initial`, `bisect_feas`, `checkpoint`, `polish_improve`, `final`, etc.)
   - feasibility labels in filename (`noov|ov`, `feas|infeas`)
2. Added event log:
   - `logs/<prefix>_history_log.csv`
3. Guarded canonical best writes so infeasible states do not overwrite canonical best.
4. Added good-hit timing summary fields:
   - `good_hits`, `first_good_t`, `last_good_t`

### New outputs

- `csv/history/...`
- `img/history/...`
- `logs/<prefix>_history_log.csv`

---

## Phase 3: Multi-node smoke jobs failed immediately (`bash` not found)

### Symptom

`logs/old_smoke_*.err` showed:

```text
error: execve(): bash: No such file or directory
```

### Root cause

`bash` used by name in `srun`/`xargs` on compute nodes; PATH/exec resolution did not find it in that context.

### Fix

- Switched all launcher invocations to `/bin/bash` explicitly.

Files updated:

- `five_node_parallel.slurm`
- `one_node_parallel.slurm`
- all wrapper scripts calling launcher now use `/bin/bash ...`

---

## Phase 4: Inline `srun bash -lc '...'` block executed incorrectly

### Symptom

`*.out` showed huge shell/environment dumps (`BASH_ARGV`, env vars) instead of expected node work logs.

### Root cause

Large inline shell payload in `srun ... bash -lc '...'` was brittle and got mangled by nested quoting/expansion.

### Fix

- Extracted node runtime logic into separate script: `five_node_worker.sh`
- Simplified launcher to:

```bash
srun --ntasks="${SLURM_JOB_NUM_NODES}" --ntasks-per-node=1 --cpu-bind=none /bin/bash "$WORKDIR/five_node_worker.sh"
```

- Added `--cpu-bind=none` to avoid strict 1-core pin behavior for the controller task.

---

## Phase 5: Worker failed with `WORKDIR: unbound variable`

### Symptom

`logs/old_smoke_*.err` showed:

```text
five_node_worker.sh: line 4: WORKDIR: unbound variable
```

### Root cause

Cluster environment had:

```text
SLURM_EXPORT_ENV=NONE
```

So exported variables from parent script were not propagated into `srun` task environments.

### Fix

- Pass required values as positional args from launcher to worker script.
- Worker script now parses args first, validates required params, then runs.

New launcher call pattern:

```bash
srun ... /bin/bash "$WORKDIR/five_node_worker.sh" \
  "$WORKDIR" "$N" "$RUNS_PER_NODE" "$BASE_SEED" "$TIME_LIMIT" "$CHECKPOINT_EVERY" "$RESERVE_CPUS" "$OUT_BASE" "$RUN_TAG"
```

Worker arg parse pattern:

```bash
WORKDIR="${1:-${WORKDIR:-}}"
N="${2:-${N:-}}"
RUNS_PER_NODE="${3:-${RUNS_PER_NODE:-}}"
BASE_SEED="${4:-${BASE_SEED:-}}"
TIME_LIMIT="${5:-${TIME_LIMIT:-}}"
CHECKPOINT_EVERY="${6:-${CHECKPOINT_EVERY:-}}"
RESERVE_CPUS="${7:-${RESERVE_CPUS:-2}}"
OUT_BASE="${8:-${OUT_BASE:-}}"
RUN_TAG="${9:-${RUN_TAG:-}}"
```

---

## Final launcher design (current)

### `five_node_parallel.slurm`

- Requests 5 nodes:
  - `--nodes=5`
  - `--ntasks=5`
  - `--ntasks-per-node=1`
- Uses exclusive nodes.
- Compiles once on submit host context (shared FS binary) then runs worker on each node.
- Creates per-submission run namespace:
  - `RUN_TAG=<timestamp>_job<jobid>`
  - `OUT_BASE=out/N###/<RUN_TAG>`
- Writes a run manifest in `logs/old_five_node_<jobid>_manifest.txt`.
- Prints an output sanity block at the end (node dir count, file count, history count, best count).

### `five_node_worker.sh`

- One controller per node.
- Creates node-separated output tree:
  - `out/N###/<RUN_TAG>/node_00`, `out/N###/<RUN_TAG>/node_01`, ...
- Computes worker fanout as:

```bash
DETECTED_CPUS=${SLURM_CPUS_ON_NODE:-$(nproc)}
WORKERS=$((DETECTED_CPUS - RESERVE_CPUS))
```

- Launches independent seeds with `xargs -P $WORKERS`.

### Prefix format

Per-run prefix includes job and node identity:

```text
N{N}_{RUN_TAG}_{node_tag}_{hostname}_w{local_worker}
```

This enables per-node and per-run timeseries grouping.

---

## Production wrappers created

- `n020_5nodes_4h.slurm`
- `n050_5nodes_4h.slurm`
- `n100_5nodes_4h.slurm`
- `n200_5nodes_8h.slurm`
- smoke:
  - `smoke_n005_5nodes_3m.slurm`
  - `smoke_n010_5nodes_3m.slurm`

Budgets:

- N20/N50/N100: wall 4h, solver `TIME_LIMIT=14100`
- N200: wall 8h, solver `TIME_LIMIT=28500`
- smoke N5/N10: wrapper filenames still contain `3m`, but scheduler wall is now 4m to reduce timeout risk; solver `TIME_LIMIT=180`.

---

## Phase 6: Time-limit cancellations on smoke runs

### Symptom

`logs/old_smoke_*.err` may show:

```text
*** JOB <id> ... CANCELLED ... DUE TO TIME LIMIT ***
```

### Root cause

- Tight smoke walltime budget can be exhausted by compile + launch + node startup jitter.
- This is independent of solver correctness and independent of `.gitignore`.

### Mitigations

- Increased smoke walltime to 4 minutes in smoke wrappers.
- Kept solver internal `TIME_LIMIT=180` for smoke behavior consistency.
- If needed, submit with lower fanout for smoke validation:

```bash
sbatch --export=ALL,RUNS_PER_NODE=4 smoke_n005_5nodes_3m.slurm
sbatch --export=ALL,RUNS_PER_NODE=4 smoke_n010_5nodes_3m.slurm
```

---

## Artifact hygiene (`.gitignore`)

Ignored runtime outputs now include:

- `2025/old/out/`
- `2025/old/logs/smoke_test_telemetry.csv`
- `2025/old/logs/smoke_test_snapstats.csv`
- `2025/old/logs/old_five_node_*_manifest.txt`

Note: `.gitignore` only affects untracked files. If any artifact was already tracked, remove it from index once with `git rm --cached <path>`.

---

## Known non-fatal warning

`logs/*.err` may still show:

```text
srun: warning: switches lack access to 1 nodes: i17n12
```

This warning is not the root fatal path observed so far.

---

## Verification checklist for next run

1. Submit smoke:

```bash
sbatch smoke_n005_5nodes_3m.slurm
sbatch smoke_n010_5nodes_3m.slurm
```

Optional lower fanout smoke:

```bash
sbatch --export=ALL,RUNS_PER_NODE=4 smoke_n005_5nodes_3m.slurm
sbatch --export=ALL,RUNS_PER_NODE=4 smoke_n010_5nodes_3m.slurm
```

2. Verify no old fatal signatures:

```bash
grep -R "execve(): bash" logs/old_smoke_*.err
grep -R "unbound variable" logs/old_smoke_*.err
```

3. Verify node controllers launched:

```bash
grep -R "===== NODE LAUNCH =====" logs/old_smoke_*.out | wc -l
```

Expected: around 5 per job.

4. Verify per-node outputs:

```bash
find out -maxdepth 4 -type d | sort
```

Expect `out/N###/<RUN_TAG>/node_00` ... `node_04`.

5. Verify solver artifacts appear under each node folder:

```bash
find out -path "*/logs/*_history_log.csv" | head
find out -path "*/csv/*_best_polys_N*.csv" | head
```

---

## Notes for future LLM revisions

- Avoid large inline `srun bash -lc '...'` payloads; prefer dedicated worker scripts.
- Assume `SLURM_EXPORT_ENV=NONE` may be set; pass critical values as args.
- Keep `/bin/bash` explicit on compute-node invocations.
- Keep canonical best writes feasibility-gated; use history files for broader event capture.
- Preserve node identity in run prefix to enable per-node timeseries aggregation.
