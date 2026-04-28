#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from feasible_analysis_common import (
    best_so_far_feasible_rows,
    parse_n_from_name,
    parse_n_values,
    read_history_rows,
    render_config_from_csv,
    style_matplotlib,
)

FrameEvent = Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build best-run GIFs from feasible improvement snapshots")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing *_history_log.csv files")
    parser.add_argument("--n-values", default="5,10,25,50,100,200", help="Comma-separated N values to include")
    parser.add_argument("--output-dir", default=None, help="Directory for generated GIFs")
    parser.add_argument("--duration-ms", type=int, default=550, help="Frame duration in milliseconds")
    return parser.parse_args()


def default_output_dir(logs_dir: Path) -> Path:
    return logs_dir.parent / "img" / "gifs"


def choose_best_runs(logs_dir: Path, n_values: List[int]) -> Dict[int, Tuple[str, Path, List[FrameEvent]]]:
    best_by_n: Dict[int, Tuple[str, Path, List[FrameEvent], float]] = {}
    for path in sorted(logs_dir.glob("*_history_log.csv")):
        n = parse_n_from_name(path)
        if n not in n_values:
            continue
        rows = read_history_rows(path)
        best_rows = best_so_far_feasible_rows(rows)
        if not best_rows:
            continue
        best_L = float(best_rows[-1]["L"])
        current = best_by_n.get(n)
        if current is None or best_L < current[3]:
            best_by_n[n] = (path.stem, path, best_rows, best_L)
    return {n: (name, path, events) for n, (name, path, events, _best_L) in best_by_n.items()}


def render_frame(n: int, run_name: str, event: FrameEvent, events: List[FrameEvent], workspace_dir: Path) -> Image.Image:
    style_matplotlib()
    fig, (ax_ts, ax_img) = plt.subplots(1, 2, figsize=(9.6, 4.8), gridspec_kw={"width_ratios": [1.25, 1.0]})

    xs = [float(row["elapsed_sec"]) for row in events]
    ys = [float(row["L"]) for row in events]
    current_t = float(event["elapsed_sec"])
    current_L = float(event["L"])
    upto = [index for index, value in enumerate(xs) if value <= current_t]
    last_index = upto[-1] if upto else 0

    ax_ts.step(xs, ys, where="post", color="#64748b", linewidth=1.6, alpha=0.75)
    ax_ts.step(xs[: last_index + 1], ys[: last_index + 1], where="post", color="#111827", linewidth=2.4)
    ax_ts.scatter([current_t], [current_L], s=85, color="gold", edgecolors="black", zorder=5)
    ax_ts.set_title(f"N={n}: {run_name}")
    ax_ts.set_xlabel("Elapsed time (s)")
    ax_ts.set_ylabel("Best-so-far L")
    ax_ts.grid(alpha=0.25)

    preview = render_config_from_csv(str(event["csv_path"]), workspace_dir, figsize=(4.2, 4.2), dpi=170)
    ax_img.axis("off")
    if preview is not None:
        ax_img.imshow(preview)
    else:
        ax_img.text(0.5, 0.5, "Preview unavailable", ha="center", va="center", transform=ax_img.transAxes)

    ax_img.set_title(
        f"t = {current_t:.1f} s\nL = {current_L:.9f}\nstage = {event['stage']}",
        fontsize=10,
    )
    fig.tight_layout()

    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()) if fig.canvas else np.zeros((1, 1, 4), dtype=np.uint8))
    if image.size == (1, 1):
        fig.canvas.draw()
        image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()))
    plt.close(fig)
    return image.convert("P", palette=Image.ADAPTIVE)


def write_gif(path: Path, frames: List[Image.Image], duration_ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def main() -> int:
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    n_values = parse_n_values(args.n_values)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(logs_dir)
    workspace_dir = logs_dir.parent

    best_runs = choose_best_runs(logs_dir, n_values)
    if not best_runs:
        raise SystemExit("No feasible best-run animations could be built")

    for n in sorted(best_runs):
        run_name, _path, events = best_runs[n]
        frames = [render_frame(n, run_name, event, events, workspace_dir) for event in events]
        out_path = output_dir / f"best_run_progress_N{n:03d}.gif"
        write_gif(out_path, frames, args.duration_ms)
        print(f"Wrote GIF: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())