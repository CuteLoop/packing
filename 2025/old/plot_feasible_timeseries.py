#!/usr/bin/env python3
"""Publication-style feasible L-vs-time plots for selected run sizes.

Layout:
- Left column: time series (all runs + aggregate best-so-far envelope)
- Right column: best configuration image (from SVG) + concise summary
"""

from __future__ import annotations

import argparse
import string
import csv
import io
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

Point = Tuple[float, float, str, str]
RunSeries = Tuple[str, List[Point]]

# Canonical base polygon used by the solver.
BASE_V = [
    (0.0, 0.8),
    (0.125, 0.5),
    (0.0625, 0.5),
    (0.2, 0.25),
    (0.1, 0.25),
    (0.35, 0.0),
    (0.075, 0.0),
    (0.075, -0.2),
    (-0.075, -0.2),
    (-0.075, 0.0),
    (-0.35, 0.0),
    (-0.1, 0.25),
    (-0.2, 0.25),
    (-0.0625, 0.5),
    (-0.125, 0.5),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot feasible time series from history logs")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing *_history_log.csv files")
    parser.add_argument(
        "--n-values",
        default="5,10,25,50,100,200",
        help="Comma-separated N values to include (default: 5,10,25,50,100,200)",
    )
    parser.add_argument(
        "--output",
        default="img/feasible_L_timeseries_selected_N.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--time-series-only",
        action="store_true",
        help="Render only time-series rows (no per-N configuration preview column)",
    )
    return parser.parse_args()


def parse_n_values(raw: str) -> List[int]:
    n_values: List[int] = []
    seen = set()
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        n = int(value)
        if n in seen:
            continue
        seen.add(n)
        n_values.append(n)
    if not n_values:
        raise ValueError("No N values specified")
    return n_values


def parse_n_from_name(path: Path) -> int | None:
    m = re.match(r"^N(\d+)_", path.name)
    return int(m.group(1)) if m else None


def read_feasible_points(csv_path: Path) -> List[Point]:
    points: List[Point] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if int(row.get("feasible", "0")) != 1:
                    continue
                elapsed = float(row["elapsed_sec"])
                L = float(row["L"])
                svg_path = row.get("svg_path", "")
                csv_geom = row.get("csv_path", "")
            except (TypeError, ValueError, KeyError):
                continue
            points.append((elapsed, L, svg_path, csv_geom))
    points.sort(key=lambda x: x[0])
    return points


def collect_runs(logs_dir: Path, n_values: List[int]) -> Dict[int, List[RunSeries]]:
    by_n: Dict[int, List[RunSeries]] = {n: [] for n in n_values}
    for path in sorted(logs_dir.glob("*_history_log.csv")):
        n = parse_n_from_name(path)
        if n not in by_n:
            continue
        pts = read_feasible_points(path)
        if pts:
            by_n[n].append((path.stem, pts))
    return by_n


def get_best_event(runs: List[RunSeries]) -> Tuple[str, float, float, str, str] | None:
    best: Tuple[str, float, float, str, str] | None = None
    for run_name, pts in runs:
        for elapsed, L, svg_path, csv_geom in pts:
            if best is None or L < best[2]:
                best = (run_name, elapsed, L, svg_path, csv_geom)
    return best


def parse_header_L(path: Path) -> float:
    try:
        with path.open("r", encoding="utf-8") as f:
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                if line.startswith("#") and " L=" in line:
                    return float(line.split(" L=")[1].split()[0])
    except Exception:
        pass
    return 0.0


def read_config_rows(path: Path) -> List[Tuple[float, float, float]]:
    rows: List[Tuple[float, float, float]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                if line.strip().lower().startswith("i,cx,cy,theta_rad"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                rows.append((float(parts[1]), float(parts[2]), float(parts[3])))
    except Exception:
        return []
    return rows


def transform_poly(cx: float, cy: float, th: float) -> List[Tuple[float, float]]:
    c = math.cos(th)
    s = math.sin(th)
    return [(c * x - s * y + cx, s * x + c * y + cy) for x, y in BASE_V]


def render_config_from_csv(csv_rel_path: str, workspace_dir: Path) -> np.ndarray | None:
    if not csv_rel_path:
        return None
    csv_path = workspace_dir / csv_rel_path
    if not csv_path.exists():
        return None

    rows = read_config_rows(csv_path)
    if not rows:
        return None

    L = parse_header_L(csv_path)
    fig = Figure(figsize=(3.0, 3.0), dpi=170)
    canvas = FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if L > 0:
        half = 0.5 * L
        ax.add_patch(Rectangle((-half, -half), L, L, fill=False, linewidth=1.6, edgecolor="black"))
    else:
        half = None

    for cx, cy, th in rows:
        poly = transform_poly(cx, cy, th)
        ax.add_patch(
            Polygon(poly, closed=True, facecolor="#6f8aa6", edgecolor="#0f172a", linewidth=0.45, alpha=0.38)
        )

    if half is not None:
        m = max(0.02, 0.02 * L)
        ax.set_xlim(-half - m, half + m)
        ax.set_ylim(-half - m, half + m)
    else:
        xs = [p[0] for cx, cy, th in rows for p in transform_poly(cx, cy, th)]
        ys = [p[1] for cx, cy, th in rows for p in transform_poly(cx, cy, th)]
        m = 0.03
        ax.set_xlim(min(xs) - m, max(xs) + m)
        ax.set_ylim(min(ys) - m, max(ys) + m)

    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
    return buf.copy()


def load_svg_preview(svg_rel_path: str, workspace_dir: Path) -> np.ndarray | None:
    if not svg_rel_path:
        return None
    svg_path = workspace_dir / svg_rel_path
    if not svg_path.exists():
        return None

    # First choice: CairoSVG conversion.
    try:
        import cairosvg  # type: ignore

        png_bytes = cairosvg.svg2png(url=str(svg_path))
        with Image.open(io.BytesIO(png_bytes)) as im:
            return np.asarray(im.convert("RGBA"))
    except Exception:
        pass

    # Fallback: svglib + reportlab renderPM backend.
    try:
        from reportlab.graphics import renderPM  # type: ignore
        from svglib.svglib import svg2rlg  # type: ignore

        drawing = svg2rlg(str(svg_path))
        if drawing is not None:
            pil_im = renderPM.drawToPIL(drawing, dpi=180)
            return np.asarray(pil_im.convert("RGBA"))
    except Exception:
        pass

    return None


def load_best_preview(svg_rel_path: str, csv_rel_path: str, workspace_dir: Path) -> np.ndarray | None:
    preview = load_svg_preview(svg_rel_path, workspace_dir)
    if preview is not None:
        return preview
    return render_config_from_csv(csv_rel_path, workspace_dir)


def aggregate_best_envelope(runs: List[RunSeries]) -> Tuple[List[float], List[float]]:
    events: List[Tuple[float, float]] = []
    for _, pts in runs:
        for t, L, _, _ in pts:
            events.append((t, L))
    if not events:
        return [], []

    events.sort(key=lambda x: x[0])
    xs: List[float] = []
    ys: List[float] = []
    current = float("inf")
    for t, L in events:
        if L < current:
            current = L
        xs.append(t)
        ys.append(current)
    return xs, ys


def style_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#1f2937",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "figure.dpi": 200,
        }
    )


def plot_group(
    ax_ts: plt.Axes,
    ax_info: plt.Axes | None,
    n: int,
    runs: List[RunSeries],
    workspace_dir: Path,
) -> None:
    if not runs:
        ax_ts.text(0.5, 0.5, f"No feasible points for N={n}", ha="center", va="center", transform=ax_ts.transAxes)
        ax_ts.set_title(f"N={n}")
        ax_ts.set_xlabel("Elapsed time (s)")
        ax_ts.set_ylabel("Square length L")
        if ax_info is not None:
            ax_info.axis("off")
        return

    cmap = plt.get_cmap("cividis")
    for i, (_run_name, pts) in enumerate(runs):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax_ts.plot(xs, ys, linewidth=0.95, alpha=0.58, color=cmap((i % 10) / 9.0))

    env_x, env_y = aggregate_best_envelope(runs)
    if env_x:
        ax_ts.plot(env_x, env_y, color="black", linewidth=2.2, label="Aggregate best-so-far")

    all_x = [x for _, pts in runs for x, _, _, _ in pts]
    all_y = [y for _, pts in runs for _, y, _, _ in pts]
    ax_ts.set_xlim(0, max(all_x) * 1.03)
    ymin, ymax = min(all_y), max(all_y)
    pad = 0.05 * (ymax - ymin) if ymax > ymin else 0.02 * max(1.0, ymax)
    ax_ts.set_ylim(ymin - pad, ymax + pad)

    ax_ts.set_title(f"N={n}: feasible configurations ({len(runs)} runs)")
    ax_ts.set_xlabel("Elapsed time (s)")
    ax_ts.set_ylabel("Square length L")
    ax_ts.margins(x=0.01)
    if env_x:
        handles = [
            Line2D([0], [0], color="#475569", lw=1.0, alpha=0.75, label="Feasible per-run trace"),
            Line2D([0], [0], color="#111827", lw=2.3, label="Aggregate best-so-far"),
            Line2D(
                [0],
                [0],
                marker="*",
                markersize=10,
                markerfacecolor="gold",
                markeredgecolor="black",
                lw=0,
                label="Best feasible point",
            ),
        ]
        ax_ts.legend(handles=handles, loc="upper right", frameon=True, facecolor="white", edgecolor="#d1d5db")

    best = get_best_event(runs)
    if ax_info is not None:
        ax_info.axis("off")
        ax_info.text(0.02, 0.98, "", va="top", fontsize=10, fontweight="bold", transform=ax_info.transAxes)

    if best is None:
        ax_info.text(0.02, 0.86, "No feasible best found", va="top", fontsize=9, transform=ax_info.transAxes)
        return

    _run_name, elapsed, best_L, svg_path, csv_geom = best
    ax_ts.scatter([elapsed], [best_L], marker="*", s=145, zorder=6, edgecolors="black", linewidths=0.7, color="gold")

    if ax_info is None:
        return

    preview = load_best_preview(svg_path, csv_geom, workspace_dir)
    img_ax = ax_info.inset_axes([0.02, 0.16, 0.96, 0.82])
    if preview is not None:
        img_ax.imshow(preview)
        img_ax.set_xticks([])
        img_ax.set_yticks([])
    else:
        img_ax.set_xticks([])
        img_ax.set_yticks([])
        img_ax.text(0.5, 0.5, "Preview unavailable", ha="center", va="center", fontsize=8, transform=img_ax.transAxes)

    for spine in img_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_edgecolor("#b7c0c9")

    summary_text = f"Best configuration for N={n}\nL = {best_L:.9f}   |   t = {elapsed:.1f} s"
    ax_info.text(
        0.5,
        0.07,
        summary_text,
        ha="center",
        va="center",
        fontsize=9,
        transform=ax_info.transAxes,
    )


def main() -> int:
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    output_png = Path(args.output)
    output_pdf = output_png.with_suffix(".pdf")
    n_values = parse_n_values(args.n_values)

    if not logs_dir.exists():
        raise SystemExit(f"Logs directory not found: {logs_dir}")

    style_matplotlib()
    by_n = collect_runs(logs_dir, n_values)

    fig_height = max(2.8 * len(n_values), 8.0)
    if args.time_series_only:
        fig = plt.figure(figsize=(12.5, fig_height), constrained_layout=True)
        gs = fig.add_gridspec(len(n_values), 1, height_ratios=[1] * len(n_values))
    else:
        fig = plt.figure(figsize=(16.0, fig_height), constrained_layout=True)
        gs = fig.add_gridspec(len(n_values), 2, width_ratios=[4.0, 2.5], height_ratios=[1] * len(n_values))

    workspace_dir = logs_dir.parent
    for row, n in enumerate(n_values):
        if args.time_series_only:
            ax_ts = fig.add_subplot(gs[row, 0])
            ax_info = None
        else:
            ax_ts = fig.add_subplot(gs[row, 0])
            ax_info = fig.add_subplot(gs[row, 1])
        plot_group(ax_ts, ax_info, n, by_n[n], workspace_dir)
        panel_label = string.ascii_lowercase[row % len(string.ascii_lowercase)]
        ax_ts.text(-0.08, 1.03, f"({panel_label})", transform=ax_ts.transAxes, fontsize=11, fontweight="bold")

    fig.suptitle("Feasible Square Length Over Time", fontsize=14, fontweight="bold")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_pdf)
    print(f"Saved: {output_png}")
    print(f"Saved: {output_pdf}")

    for n in n_values:
        n_pts = sum(len(pts) for _, pts in by_n[n])
        print(f"N={n} runs: {len(by_n[n])}, feasible points: {n_pts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
