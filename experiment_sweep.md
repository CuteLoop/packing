# Experiment Sweep and Analysis Report

This document summarizes what was run, what the analysis scripts produce, and the concrete results currently present in this workspace.

## 1. Data Snapshot

- Graph-suite output directories currently present: `out/N005`, `out/N010`, `out/N020`, `out/N050`, `out/N100`
- Total bisection CSV files under `out/`: 53
- Graph-suite bisection CSVs matched by `out/N*/graph_*/*_bisection.csv`: 28
- Legacy feasible logs in `2025/old/logs`: includes N=5,10,25,50,100,200 history logs

## 2. Main Sweep Jobs (Current Pipeline)

Primary sweep jobs:

- `scripts/run_graph_suite.slurm` for N=5,10,20,50,100
- `scripts/sweep_n100.slurm` for focused N=100 sweeps

Typical per-run artifacts:

- `*_bisection.csv`
- `*_log.csv`
- `*_best_state.csv`
- `*_best_state.svg`

## 3. Graph-Suite Results from `out/N*/graph_*/*_bisection.csv`

The table below is aggregated from the existing CSVs in this workspace. Medians are across seeds for each `(N, method)` pair.

| N | method | runs | median L_best | best L_best | median probes | median wall_sec | median final bracket | median first feasible t |
|---:|:------|----:|--------------:|------------:|--------------:|----------------:|---------------------:|------------------------:|
| 5 | erms | 2 | 1.886118 | 1.665342 | 10.0 | 37.0 | 0.001299 | 6.0 |
| 5 | ms | 2 | 1.616642 | 1.601707 | 10.0 | 65.5 | 0.001299 | 0.0 |
| 5 | pt | 2 | NaN | NaN | 157.0 | 3600.0 | 0.000000 | NaN |
| 10 | erms | 2 | 2.538811 | 2.503915 | 10.0 | 73.0 | 0.001837 | 4.5 |
| 10 | ms | 2 | 2.539729 | 2.511262 | 10.0 | 27.5 | 0.001837 | 6.0 |
| 10 | pt | 2 | NaN | NaN | 157.0 | 3600.0 | 0.000000 | NaN |
| 20 | erms | 2 | 4.022881 | 3.993012 | 10.0 | 68.5 | 0.002597 | 10.5 |
| 20 | ms | 2 | 4.008596 | 4.003401 | 10.0 | 66.0 | 0.002597 | 7.0 |
| 20 | pt | 2 | NaN | NaN | 157.0 | 3600.0 | 0.000000 | NaN |
| 50 | erms | 2 | 8.941852 | 8.941852 | 10.0 | 65.0 | 0.005476 | 64.5 |
| 50 | ms | 2 | 8.147873 | 8.010980 | 10.0 | 96.0 | 0.005476 | 25.5 |
| 50 | pt | 2 | NaN | NaN | 71.0 | 2590.0 | 0.000000 | NaN |
| 100 | erms | 2 | 11.456042 | 11.451202 | 10.0 | 195.5 | 0.009680 | 22.5 |
| 100 | ms | 2 | 14.819771 | 14.790732 | 10.0 | 209.0 | 0.009680 | 149.0 |

Notes:

- `NaN` for some PT entries means no feasible bisection row in the matched files.
- `median final bracket` is taken from `bracket_width` if present, otherwise `L_hi - L_lo` on the final row.

## 4. Legacy Feasible Analysis Results (`2025/old`)

Derived from:

- `2025/old/analysis/feasible_run_statistics.csv`
- `2025/old/analysis/feasible_n_statistics.csv`

Per-N feasible summary:

| N | runs with feasible points | median first feasible t (s) | median time to best (s) | best-of-best L | best-of-best density | median best density |
|---:|--------------------------:|----------------------------:|------------------------:|---------------:|---------------------:|--------------------:|
| 5 | 15 | 5.0 | 110.0 | 1.472296904 | 0.566567651 | 0.558101006 |
| 10 | 15 | 6.0 | 148.0 | 2.055599851 | 0.581293427 | 0.564563142 |
| 25 | 6 | 8.0 | 205.5 | 3.315186124 | 0.558723251 | 0.539594628 |
| 50 | 4 | 46.0 | 3656.5 | 4.652290343 | 0.567425633 | 0.560395616 |
| 100 | 3 | 53.0 | 3254.0 | 6.785398411 | 0.533484569 | 0.526799685 |
| 200 | 1 | 19.0 | 449.0 | 10.104171651 | 0.481172870 | 0.481172870 |

## 5. What Each `plot_*` and `make_*` Script Does

### `2025/old/plot_feasible_timeseries.py`

Purpose:

- Builds publication-style feasible `L vs elapsed time` panels for selected N values
- Right-hand panel shows best packing preview (from SVG when available, else reconstructed from CSV)

Key inputs:

- `--logs-dir` (default `logs`)
- `--n-values` (default `5,10,25,50,100,200`)
- `--output` PNG path; PDF with same stem is also written

Outputs generated in this workspace:

- `2025/old/img/feasible_L_timeseries_N5_N10_N25_N50_N100_N200.png`
- `2025/old/img/feasible_L_timeseries_N5_N10_N25_N50_N100_N200.pdf`
- `2025/old/img/feasible_L_timeseries_N5_N10_N20_N50_N100_N200.png` (older run including N=20)
- `2025/old/img/feasible_L_timeseries_N5_N10_N20_N50_N100_N200.pdf`

### `2025/old/plot_feasible_progress.py`

Purpose:

- Plot 1: best-so-far density trajectories over time, with median curve per N
- Plot 2: time-to-final-best distribution by N

Key inputs:

- `--logs-dir`
- `--n-values`
- `--output-dir` (defaults to sibling `img` directory)

Outputs generated:

- `2025/old/img/feasible_best_density_timeseries.png`
- `2025/old/img/feasible_best_density_timeseries.pdf`
- `2025/old/img/feasible_time_to_best_by_n.png`
- `2025/old/img/feasible_time_to_best_by_n.pdf`

### `2025/old/make_feasible_timeseries.py`

Purpose:

- Converts feasible rows from `*_history_log.csv` into tabular timeseries and run summary CSVs
- Optional static plot generation

Default outputs (if run with defaults):

- `<output-dir>/feasible_L_timeseries.csv`
- `<output-dir>/feasible_L_run_summary.csv`
- `<output-dir>/<plot name>`

### `2025/old/make_feasible_best_gifs.py`

Purpose:

- Selects best run per N and creates a frame-by-frame GIF of best-so-far feasible improvements

Key inputs:

- `--logs-dir`
- `--n-values`
- `--output-dir` (defaults to `img/gifs`)
- `--duration-ms`

Output generated:

- `2025/old/img/gifs/best_run_progress_N010.gif`

### `2025/old/plot_packing.py`

Purpose:

- Render one packing CSV (`*_best_polys_N###.csv`) into a static image (`png` or `svg`)

Usage pattern:

- `python 2025/old/plot_packing.py <csv> <out> [--size ...] [--margin ...]`

## 6. Additional Legacy Analysis Scripts Used

### `2025/old/summarize_feasible_stats.py`

Purpose:

- Produces run-level and N-level feasible statistics

Outputs generated:

- `2025/old/analysis/feasible_run_statistics.csv`
- `2025/old/analysis/feasible_n_statistics.csv`

## 7. Legacy Slurm Job Families in `2025/old`

### A) Five-node embarrassingly parallel controller/worker

Main controller:

- `2025/old/five_node_parallel.slurm`

Worker launcher:

- `2025/old/five_node_worker.sh`

N-specific wrappers:

- `2025/old/n020_5nodes_4h.slurm`
- `2025/old/n050_5nodes_4h.slurm`
- `2025/old/n100_5nodes_4h.slurm`
- `2025/old/n200_5nodes_8h.slurm`

Configuration highlights:

- `--nodes=5`, `--ntasks=5`, `--ntasks-per-node=1`, `--exclusive`
- Defaults to `RUNS_PER_NODE=64` (about 320 seeds total per job)
- `TIME_LIMIT` set close to walltime minus post-processing cushion
- Per-node workers run with `xargs -P <workers>` where `workers = detected_cpus - RESERVE_CPUS`

### B) Single-node Slurm array jobs (older baseline)

- `2025/old/old_n25_4h.slurm` with `--array=1-8%2`
- `2025/old/old_n50_4h.slurm` with `--array=1-6%2`
- `2025/old/old_n100_4h.slurm` with `--array=1-4%2`
- `2025/old/old_n200_8h.slurm` with `--array=1-2%2`

Configuration highlights:

- One task per array element, each running one seed (`SEED = 12345 + SLURM_ARRAY_TASK_ID`)
- Compiles `hpc_parallel.c` on-node and runs `./HPC_parallel_old`
- Uses N-specific trial schedules (`trials_bracket`, `trials_bisect`, `trials_polish`)
- Logs/artifacts are written under `csv/`, `img/`, and `logs/`

## 8. Generated Visual Artifact List (Current Workspace)

- `2025/old/img/feasible_best_density_timeseries.pdf`
- `2025/old/img/feasible_best_density_timeseries.png`
- `2025/old/img/feasible_L_timeseries_N5_N10_N20_N50_N100_N200.pdf`
- `2025/old/img/feasible_L_timeseries_N5_N10_N20_N50_N100_N200.png`
- `2025/old/img/feasible_L_timeseries_N5_N10_N25_N50_N100_N200.pdf`
- `2025/old/img/feasible_L_timeseries_N5_N10_N25_N50_N100_N200.png`
- `2025/old/img/feasible_time_to_best_by_n.pdf`
- `2025/old/img/feasible_time_to_best_by_n.png`
- `2025/old/img/gifs/best_run_progress_N010.gif`
- `2025/old/img/history/file.gif`
- `2025/old/img/smoke_feasible_L_timeseries_N5_N10.pdf`
- `2025/old/img/smoke_feasible_L_timeseries_N5_N10.png`

## 9. Repro Commands (Python 3 on Windows)

```powershell
py -3 .\2025\old\summarize_feasible_stats.py --logs-dir .\2025\old\logs --n-values "5,10,25,50,100,200"
py -3 .\2025\old\plot_feasible_progress.py --logs-dir .\2025\old\logs --n-values "5,10,25,50,100,200"
py -3 .\2025\old\plot_feasible_timeseries.py --logs-dir .\2025\old\logs --n-values "5,10,25,50,100,200" --output .\2025\old\img\feasible_L_timeseries_N5_N10_N25_N50_N100_N200.png
py -3 .\2025\old\make_feasible_best_gifs.py --logs-dir .\2025\old\logs --n-values "10"
```

## 10. References

- Engineering spec: `docs/DOCS.md`
- Experiment plan: `docs/Experiments.md`
- Output schema: `docs/CSV_SCHEMA.md`
- Main analysis scripts: `scripts/analyze_comparison.py`, `scripts/analyze_sweep.py`, `scripts/build_submission.py`
