# Experiment Report

Date: 2026-04-25

This document consolidates the experiment work in this repository, from Slurm job setup to generated results, plots, and analysis outputs.

## 1) Scope and Data Sources

This report covers two pipelines:

1. Current graph-suite study pipeline in [scripts](../scripts):
- [scripts/run_graph_suite.slurm](../scripts/run_graph_suite.slurm)
- [scripts/sweep_n100.slurm](../scripts/sweep_n100.slurm)
- outputs in [out](../out)

2. Legacy naive-SA embarrassingly parallel baseline in [2025/old](../2025/old):
- Slurm launchers in [2025/old](../2025/old)
- logs in [2025/old/logs](../2025/old/logs)
- analysis tables in [2025/old/analysis](../2025/old/analysis)
- figures in [2025/old/img](../2025/old/img)

## 2) Experiment Execution Configurations

Intent note:
- The intended legacy campaign was the five-node launcher family.
- The legacy results currently present in this workspace are mixed provenance: five-node style for N=5,10 and single-node array style for N=25,50,100,200.

### 2.1 Graph suite job (current pipeline)

File: [scripts/run_graph_suite.slurm](../scripts/run_graph_suite.slurm)

Configured resources:
- 1 node, 1 task, 20 CPUs per task, 8 GB RAM, 4 hours
- methods: ms, erms, pt
- N values: 5, 10, 20, 50, 100
- seeds per method/N: 1000 and 2000
- per-run budget: 3600 s
- per-seed thread count: R=10 (two seeds launched concurrently)

Output logs:
- [logs](../logs) files named graph_<jobid>.out/.err

### 2.2 Isolated N=100 sweep (current pipeline)

File: [scripts/sweep_n100.slurm](../scripts/sweep_n100.slurm)

Configured resources:
- 1 node, 1 task, 20 CPUs per task, 8 GB RAM, 4 hours
- methods: ms, erms, pt
- fixed N=100
- seeds: 1000 and 2000
- per-run budget: 3600 s

Output logs:
- [logs](../logs) files named n100_sweep_<jobid>.out/.err

### 2.3 Legacy five-node embarrassingly parallel family

Controller and worker:
- [2025/old/five_node_parallel.slurm](../2025/old/five_node_parallel.slurm)
- [2025/old/five_node_worker.sh](../2025/old/five_node_worker.sh)

N-specific wrappers:
- [2025/old/n020_5nodes_4h.slurm](../2025/old/n020_5nodes_4h.slurm)
- [2025/old/n050_5nodes_4h.slurm](../2025/old/n050_5nodes_4h.slurm)
- [2025/old/n100_5nodes_4h.slurm](../2025/old/n100_5nodes_4h.slurm)
- [2025/old/n200_5nodes_8h.slurm](../2025/old/n200_5nodes_8h.slurm)
- smoke launchers: [2025/old/smoke_n005_5nodes_3m.slurm](../2025/old/smoke_n005_5nodes_3m.slurm), [2025/old/smoke_n010_5nodes_3m.slurm](../2025/old/smoke_n010_5nodes_3m.slurm)

Key behavior:
- 5 exclusive nodes, 1 controller task per node
- default RUNS_PER_NODE=64 (up to ~320 seeds/job)
- workers per node are auto-derived from available CPUs minus RESERVE_CPUS
- deterministic seeds from BASE_SEED + global index
- each run writes independent history and best-geometry artifacts

Observed in current artifacts:
- Used for N=5 and N=10 cohorts only (filename pattern contains `_node_..._w...`).

### 2.4 Legacy single-node array family

Files:
- [2025/old/old_n25_4h.slurm](../2025/old/old_n25_4h.slurm)
- [2025/old/old_n50_4h.slurm](../2025/old/old_n50_4h.slurm)
- [2025/old/old_n100_4h.slurm](../2025/old/old_n100_4h.slurm)
- [2025/old/old_n200_8h.slurm](../2025/old/old_n200_8h.slurm)

Array sizes and limits:
- N=25: array 1-8%2
- N=50: array 1-6%2
- N=100: array 1-4%2
- N=200: array 1-2%2

Each array task runs one independent seed with task-specific trial schedules.

Observed in current artifacts:
- This is the run family that produced the N=25,50,100,200 cohorts used in the legacy summary tables and plots (filename pattern contains `_task<id>`).

## 3) Run Inventory in Workspace

Current graph-suite output folders present:
- [out/N005](../out/N005)
- [out/N010](../out/N010)
- [out/N020](../out/N020)
- [out/N050](../out/N050)
- [out/N100](../out/N100)

Observed counts:
- total bisection CSVs in [out](../out): 53
- graph-suite subset matched by out/N*/graph_*/*_bisection.csv: 28 files

Legacy feasible history logs are present for:
- N = 5, 10, 25, 50, 100, 200 in [2025/old/logs](../2025/old/logs)

Provenance by naming pattern in `*_history_log.csv`:
- N=5: node=15, task=0
- N=10: node=15, task=0
- N=25: node=0, task=6
- N=50: node=0, task=4
- N=100: node=0, task=3
- N=200: node=0, task=2

## 4) Quantitative Results

### 4.1 Graph-suite aggregation (from out/N*/graph_*/*_bisection.csv)

Medians are across available seeds per (N, method).

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
- NaN rows indicate no feasible points in the matched graph files for that method/N.
- Final bracket uses bracket_width when available, else computed from L_hi-L_lo.

### 4.2 Legacy feasible summary (from 2025/old analysis)

Source tables:
- [2025/old/analysis/feasible_run_statistics.csv](../2025/old/analysis/feasible_run_statistics.csv)
- [2025/old/analysis/feasible_n_statistics.csv](../2025/old/analysis/feasible_n_statistics.csv)

Run-family note for this summary:
- N=5 and N=10 rows come from five-node style runs.
- N=25,50,100,200 rows come from single-node array runs.

Per-N summary:

| N | runs with feasible points | median first feasible t (s) | median time to best (s) | best-of-best L | best-of-best density | median best density |
|---:|--------------------------:|----------------------------:|------------------------:|---------------:|---------------------:|--------------------:|
| 5 | 15 | 5.0 | 110.0 | 1.472296904 | 0.566567651 | 0.558101006 |
| 10 | 15 | 6.0 | 148.0 | 2.055599851 | 0.581293427 | 0.564563142 |
| 25 | 6 | 8.0 | 205.5 | 3.315186124 | 0.558723251 | 0.539594628 |
| 50 | 4 | 46.0 | 3656.5 | 4.652290343 | 0.567425633 | 0.560395616 |
| 100 | 3 | 53.0 | 3254.0 | 6.785398411 | 0.533484569 | 0.526799685 |
| 200 | 1 | 19.0 | 449.0 | 10.104171651 | 0.481172870 | 0.481172870 |

## 5) Plots and Visual Outputs Generated

### 5.1 Legacy feasible analysis plots

From [2025/old/img](../2025/old/img):
- [2025/old/img/feasible_L_timeseries_N5_N10_N25_N50_N100_N200.png](../2025/old/img/feasible_L_timeseries_N5_N10_N25_N50_N100_N200.png)
- [2025/old/img/feasible_L_timeseries_N5_N10_N25_N50_N100_N200.pdf](../2025/old/img/feasible_L_timeseries_N5_N10_N25_N50_N100_N200.pdf)
- [2025/old/img/feasible_best_density_timeseries.png](../2025/old/img/feasible_best_density_timeseries.png)
- [2025/old/img/feasible_best_density_timeseries.pdf](../2025/old/img/feasible_best_density_timeseries.pdf)
- [2025/old/img/feasible_time_to_best_by_n.png](../2025/old/img/feasible_time_to_best_by_n.png)
- [2025/old/img/feasible_time_to_best_by_n.pdf](../2025/old/img/feasible_time_to_best_by_n.pdf)
- [2025/old/img/gifs/best_run_progress_N010.gif](../2025/old/img/gifs/best_run_progress_N010.gif)

Additional historical/smoke visuals:
- [2025/old/img/smoke_feasible_L_timeseries_N5_N10.png](../2025/old/img/smoke_feasible_L_timeseries_N5_N10.png)
- [2025/old/img/smoke_feasible_L_timeseries_N5_N10.pdf](../2025/old/img/smoke_feasible_L_timeseries_N5_N10.pdf)
- [2025/old/img/history/file.gif](../2025/old/img/history/file.gif)

### 5.2 Top-level analysis plots

From [analysis/plots](../analysis/plots):
- [analysis/plots/density_vs_N.png](../analysis/plots/density_vs_N.png)
- [analysis/plots/density_vs_inv_sqrtN.png](../analysis/plots/density_vs_inv_sqrtN.png)
- [analysis/plots/area_gap_vs_N.png](../analysis/plots/area_gap_vs_N.png)
- [analysis/plots/boundary_fraction_vs_N.png](../analysis/plots/boundary_fraction_vs_N.png)
- [analysis/plots/nn_mean_vs_N.png](../analysis/plots/nn_mean_vs_N.png)
- [analysis/plots/orientation_entropy_vs_N.png](../analysis/plots/orientation_entropy_vs_N.png)

Supporting table:
- [analysis/summary.csv](../analysis/summary.csv)

## 6) Analysis Scripts and What They Produced

### 6.1 Legacy scripts in 2025/old

- [2025/old/summarize_feasible_stats.py](../2025/old/summarize_feasible_stats.py)
  - produced [2025/old/analysis/feasible_run_statistics.csv](../2025/old/analysis/feasible_run_statistics.csv)
  - produced [2025/old/analysis/feasible_n_statistics.csv](../2025/old/analysis/feasible_n_statistics.csv)

- [2025/old/plot_feasible_timeseries.py](../2025/old/plot_feasible_timeseries.py)
  - produced multi-panel feasible L-over-time figures and PDFs

- [2025/old/plot_feasible_progress.py](../2025/old/plot_feasible_progress.py)
  - produced density-over-time and time-to-best plots

- [2025/old/make_feasible_best_gifs.py](../2025/old/make_feasible_best_gifs.py)
  - produced best-run GIF animation

- [2025/old/make_feasible_timeseries.py](../2025/old/make_feasible_timeseries.py)
  - writes feasible time-series CSV and run-summary CSV for history logs
  - optional quick plot

- [2025/old/plot_packing.py](../2025/old/plot_packing.py)
  - renders one geometry CSV to a static image

### 6.2 Top-level scripts in scripts/

- [scripts/analyze_sweep.py](../scripts/analyze_sweep.py)
  - reads csv/*_best_polys_N*.csv
  - writes [analysis/summary.csv](../analysis/summary.csv)
  - writes plot set in [analysis/plots](../analysis/plots)

- [scripts/analyze_comparison.py](../scripts/analyze_comparison.py)
  - parses sweep/graph bisection CSVs
  - designed to output comparison summary + method-vs-N plots

- [scripts/build_submission.py](../scripts/build_submission.py)
  - builds [submission.csv](../submission.csv) from best/checkpoint files

### 6.3 Validation

- [analysis/validate_schema.py](../analysis/validate_schema.py)
  - validates study CSV schema and basic consistency

## 7) Reproducibility Commands Used

Windows Python commands used for the legacy feasible analysis:

- py -3 .\2025\old\summarize_feasible_stats.py --logs-dir .\2025\old\logs --n-values "5,10,25,50,100,200"
- py -3 .\2025\old\plot_feasible_progress.py --logs-dir .\2025\old\logs --n-values "5,10,25,50,100,200"
- py -3 .\2025\old\plot_feasible_timeseries.py --logs-dir .\2025\old\logs --n-values "5,10,25,50,100,200" --output .\2025\old\img\feasible_L_timeseries_N5_N10_N25_N50_N100_N200.png
- py -3 .\2025\old\make_feasible_best_gifs.py --logs-dir .\2025\old\logs --n-values "10"

Graph summary extraction (defensive parsing for blank numeric fields) was run over out/N*/graph_*/*_bisection.csv and used to generate Section 4.1.

## 8) Key Observations

1. In available graph-suite outputs, ms and erms have feasible solutions across N=5..100, while pt entries in matched graph files frequently contain no feasible rows under these budgets.
2. In legacy naive-SA baseline logs, N=5 and N=10 have many feasible runs and quick first-feasible times; larger N values are feasible but generally take much longer to reach final best configurations.
3. Legacy feasible density peaks are around 0.56-0.58 for N up to 50 in the best runs present here.
4. There is no N=20 legacy history-log cohort in 2025/old; legacy large-N series uses N=25 instead.
5. The legacy large-N cohorts present here (N=25,50,100,200) were produced by the single-node array scripts (`old_n*_*.slurm`), not by the five-node launcher family.

## 9) Referenced Documentation

- [docs/Experiments.md](../docs/Experiments.md)
- [docs/CSV_SCHEMA.md](../docs/CSV_SCHEMA.md)
- [docs/DOCS.md](../docs/DOCS.md)
- [README.md](../README.md)
