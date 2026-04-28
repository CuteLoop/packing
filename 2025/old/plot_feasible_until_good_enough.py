#!/usr/bin/env python3
"""Plot feasible L-vs-time traces until a good-enough packing is reached.

Outputs:
1) A single combined plot containing all selected N time series.
2) One rendered best configuration image per N.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from feasible_analysis_common import (
    parse_n_from_name,
    parse_n_values,
    read_history_rows,
    render_config_from_csv,
    style_matplotlib,
)

RunSeries = List[Tuple[float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all feasible L-vs-time traces until good-enough packing"
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory containing *_history_log.csv files (default: logs)",
    )
    parser.add_argument(
        "--n-values",
        default="5,10,25,50,100,200",
        help="Comma-separated N values to include",
    )
    parser.add_argument(
        "--good-enough-rel-gap",
        type=float,
        default=0.02,
        help="Relative gap above best L per N used as good-enough threshold, e.g. 0.02 => 2%%",
    )
    parser.add_argument(
        "--output-timeseries",
        default="img/feasible_L_all_timeseries_until_good_enough.png",
        help="Combined output PNG for all time series",
    )
    parser.add_argument(
        "--output-config-dir",
        default="img/per_n_configs",
        help="Output directory for per-N configuration images",
    )
    return parser.parse_args()


def collect_feasible_by_n(logs_dir: Path, n_values: List[int]) -> Dict[int, List[List[Dict[str, object]]]]:
    by_n: Dict[int, List[List[Dict[str, object]]]] = {n: [] for n in n_values}
    for path in sorted(logs_dir.glob("*_history_log.csv")):
        n = parse_n_from_name(path)
        if n not in by_n:
            continue

        rows = read_history_rows(path)
        feasible = [row for row in rows if int(row["feasible"]) == 1]
        if feasible:
            by_n[n].append(feasible)
    return by_n


def best_L_per_n(by_n: Dict[int, List[List[Dict[str, object]]]]) -> Dict[int, float]:
    best: Dict[int, float] = {}
    for n, runs in by_n.items():
        min_L = None
        for run in runs:
            for row in run:
                L = float(row["L"])
                if min_L is None or L < min_L:
                    min_L = L
        if min_L is not None:
            best[n] = min_L
    return best


def truncate_run_until_good_enough(run: List[Dict[str, object]], threshold_L: float) -> RunSeries:
    out: RunSeries = []
    for row in run:
        t = float(row["elapsed_sec"])
        L = float(row["L"])
        out.append((t, L))
        if L <= threshold_L:
            break
    return out


def build_truncated_runs(
    by_n: Dict[int, List[List[Dict[str, object]]]], best_by_n: Dict[int, float], rel_gap: float
) -> Dict[int, List[RunSeries]]:
    truncated: Dict[int, List[RunSeries]] = {n: [] for n in by_n}
    for n, runs in by_n.items():
        if n not in best_by_n:
            continue
        threshold = best_by_n[n] * (1.0 + rel_gap)
        for run in runs:
            series = truncate_run_until_good_enough(run, threshold)
            if series:
                truncated[n].append(series)
    return truncated


def choose_best_config_rows(by_n: Dict[int, List[List[Dict[str, object]]]]) -> Dict[int, Dict[str, object]]:
    picked: Dict[int, Dict[str, object]] = {}
    for n, runs in by_n.items():
        best_row = None
        for run in runs:
            for row in run:
                if best_row is None or float(row["L"]) < float(best_row["L"]):
                    best_row = row
        if best_row is not None:
            picked[n] = best_row
    return picked


def plot_all_timeseries(
    truncated: Dict[int, List[RunSeries]],
    best_by_n: Dict[int, float],
    rel_gap: float,
    out_png: Path,
) -> None:
    style_matplotlib()
    fig, ax = plt.subplots(figsize=(13.0, 7.2), constrained_layout=True)

    cmap = plt.get_cmap("tab10")
    n_order = [n for n in sorted(truncated) if truncated[n]]
    for i, n in enumerate(n_order):
        color = cmap(i % 10)
        for series in truncated[n]:
            xs = [x for x, _ in series]
            ys = [y for _, y in series]
            ax.plot(xs, ys, color=color, alpha=0.33, linewidth=0.95)

        threshold = best_by_n[n] * (1.0 + rel_gap)
        ax.axhline(threshold, color=color, alpha=0.22, linewidth=0.9, linestyle="--")

    if n_order:
        legend_handles = []
        for i, n in enumerate(n_order):
            color = cmap(i % 10)
            legend_handles.append(plt.Line2D([0], [0], color=color, lw=2.0, label=f"N={n}"))
        ax.legend(handles=legend_handles, title="Instance size", loc="upper right", frameon=True)

    ax.set_title("Feasible square length over time until good-enough packing")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Square length L")
    ax.grid(alpha=0.22, linewidth=0.55)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def render_per_n_configs(best_row_by_n: Dict[int, Dict[str, object]], logs_dir: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = logs_dir.parent
    written: List[Path] = []

    for n in sorted(best_row_by_n):
        row = best_row_by_n[n]
        csv_rel = str(row.get("csv_path", ""))
        image = render_config_from_csv(csv_rel, workspace_dir, figsize=(4.2, 4.2), dpi=220)
        if image is None:
            continue

        out_path = out_dir / f"N{n:03d}_best_config.png"
        plt.imsave(out_path, image)
        written.append(out_path)

    return written


def main() -> int:
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    n_values = parse_n_values(args.n_values)

    if not logs_dir.exists():
        raise SystemExit(f"Logs directory not found: {logs_dir}")

    by_n = collect_feasible_by_n(logs_dir, n_values)
    best_by_n = best_L_per_n(by_n)
    truncated = build_truncated_runs(by_n, best_by_n, args.good_enough_rel_gap)
    best_row_by_n = choose_best_config_rows(by_n)

    out_timeseries = Path(args.output_timeseries)
    plot_all_timeseries(truncated, best_by_n, args.good_enough_rel_gap, out_timeseries)
    written_configs = render_per_n_configs(best_row_by_n, logs_dir, Path(args.output_config_dir))

    print(f"Saved: {out_timeseries}")
    print(f"Saved: {out_timeseries.with_suffix('.pdf')}")
    for n in sorted(n_values):
        runs = len(by_n.get(n, []))
        truncated_runs = len(truncated.get(n, []))
        if n in best_by_n:
            threshold = best_by_n[n] * (1.0 + args.good_enough_rel_gap)
            print(
                f"N={n}: runs={runs}, truncated_runs={truncated_runs}, best_L={best_by_n[n]:.9f}, good_enough_L={threshold:.9f}"
            )
        else:
            print(f"N={n}: no feasible runs")

    if written_configs:
        print("Per-N configuration renders:")
        for path in written_configs:
            print(f"  {path}")
    else:
        print("No per-N configuration images were generated.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
