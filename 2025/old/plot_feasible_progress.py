#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from feasible_analysis_common import (
    best_so_far_feasible_rows,
    density_from_L,
    parse_n_from_name,
    parse_n_values,
    read_history_rows,
    style_matplotlib,
)

RunCurve = Tuple[str, List[float], List[float], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot best-so-far density and time-to-best summaries")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing *_history_log.csv files")
    parser.add_argument("--n-values", default="5,10,25,50,100,200", help="Comma-separated N values to include")
    parser.add_argument("--output-dir", default=None, help="Directory for generated plots")
    return parser.parse_args()


def default_output_dir(logs_dir: Path) -> Path:
    return logs_dir.parent / "img"


def collect_curves(logs_dir: Path, n_values: List[int]) -> Dict[int, List[RunCurve]]:
    by_n: Dict[int, List[RunCurve]] = {n: [] for n in n_values}
    for path in sorted(logs_dir.glob("*_history_log.csv")):
        n = parse_n_from_name(path)
        if n not in by_n:
            continue
        rows = read_history_rows(path)
        best_rows = best_so_far_feasible_rows(rows)
        if not best_rows:
            continue
        xs = [float(row["elapsed_sec"]) for row in best_rows]
        ys = [density_from_L(n, float(row["L"])) for row in best_rows]
        by_n[n].append((path.stem, xs, ys, xs[-1]))
    return by_n


def stepped_values(xs: List[float], ys: List[float], grid: np.ndarray, run_end: float) -> np.ndarray:
    values = np.full(grid.shape, np.nan, dtype=float)
    if not xs:
        return values
    j = 0
    current = np.nan
    for i, t in enumerate(grid):
        if t > run_end:
            break
        while j < len(xs) and xs[j] <= t:
            current = ys[j]
            j += 1
        values[i] = current
    return values


def save_dual(path_base: Path, fig: plt.Figure) -> None:
    fig.savefig(path_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")


def plot_density_curves(by_n: Dict[int, List[RunCurve]], output_dir: Path) -> Path:
    nonempty = [n for n, runs in by_n.items() if runs]
    if not nonempty:
        raise SystemExit("No feasible runs found for density plot")

    style_matplotlib()
    fig, axes = plt.subplots(len(nonempty), 1, figsize=(10.5, max(3.0, 2.5 * len(nonempty))), squeeze=False)
    axes_flat = axes[:, 0]
    cmap = plt.get_cmap("cividis")

    for ax, n in zip(axes_flat, nonempty):
        runs = by_n[n]
        max_t = max(run_end for _name, _xs, _ys, run_end in runs)
        grid = np.linspace(0.0, max_t, 250)
        samples = []
        for index, (_name, xs, ys, run_end) in enumerate(runs):
            ax.step(xs, ys, where="post", linewidth=0.95, alpha=0.35, color=cmap((index % 10) / 9.0))
            samples.append(stepped_values(xs, ys, grid, run_end))

        sample_matrix = np.vstack(samples)
        valid_mask = np.isfinite(sample_matrix)
        median_curve = np.full(grid.shape, np.nan, dtype=float)
        for column in range(sample_matrix.shape[1]):
            column_values = sample_matrix[:, column][valid_mask[:, column]]
            if column_values.size:
                median_curve[column] = float(np.median(column_values))
        ax.plot(grid, median_curve, color="black", linewidth=2.1, label="Median best-so-far density")
        ax.set_title(f"N={n}: best-so-far density over time ({len(runs)} runs)")
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel(r"Density $\eta = N A_p / L^2$")
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", frameon=True)

    fig.tight_layout()
    out_base = output_dir / "feasible_best_density_timeseries"
    save_dual(out_base, fig)
    plt.close(fig)
    return out_base.with_suffix(".png")


def plot_time_to_best(by_n: Dict[int, List[RunCurve]], output_dir: Path) -> Path:
    style_matplotlib()
    ordered = [(n, runs) for n, runs in by_n.items() if runs]
    if not ordered:
        raise SystemExit("No feasible runs found for time-to-best plot")

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    positions = np.arange(1, len(ordered) + 1)
    data = [[xs[-1] for _name, xs, _ys, _run_end in runs] for _n, runs in ordered]
    ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True, boxprops={"facecolor": "#cbd5e1"})

    for pos, values in zip(positions, data):
        jitter = np.linspace(-0.12, 0.12, num=len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(np.full(len(values), pos) + jitter, values, color="#0f172a", alpha=0.78, s=28, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(n) for n, _runs in ordered])
    ax.set_xlabel("N")
    ax.set_ylabel("Time to final best (s)")
    ax.set_title("Distribution of time to final best feasible configuration by N")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_base = output_dir / "feasible_time_to_best_by_n"
    save_dual(out_base, fig)
    plt.close(fig)
    return out_base.with_suffix(".png")


def main() -> int:
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    n_values = parse_n_values(args.n_values)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(logs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_n = collect_curves(logs_dir, n_values)
    density_path = plot_density_curves(by_n, output_dir)
    time_path = plot_time_to_best(by_n, output_dir)

    print(f"Wrote density plot: {density_path}")
    print(f"Wrote time-to-best plot: {time_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
