#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

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


def polygon_area(vertices: Sequence[Tuple[float, float]]) -> float:
    area2 = 0.0
    for i, (x0, y0) in enumerate(vertices):
        x1, y1 = vertices[(i + 1) % len(vertices)]
        area2 += x0 * y1 - x1 * y0
    return 0.5 * abs(area2)


POLY_AREA = polygon_area(BASE_V)


def parse_n_values(raw: str) -> List[int]:
    values: List[int] = []
    seen = set()
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        value = int(text)
        if value in seen:
            continue
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("No N values specified")
    return values


def parse_n_from_name(path: Path) -> int | None:
    match = re.match(r"^N(\d+)_", path.name)
    return int(match.group(1)) if match else None


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
            "figure.dpi": 180,
        }
    )


def read_history_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            try:
                elapsed = float(raw["elapsed_sec"])
                L = float(raw["L"])
            except (KeyError, TypeError, ValueError):
                continue

            rows.append(
                {
                    "event_idx": int(raw.get("event_idx", len(rows))),
                    "elapsed_sec": elapsed,
                    "L": L,
                    "stage": raw.get("stage", ""),
                    "feasible": int(raw.get("feasible", "0") or 0),
                    "csv_path": raw.get("csv_path", ""),
                    "svg_path": raw.get("svg_path", ""),
                    "overlap": float(raw.get("overlap", "0") or 0.0),
                    "outside": float(raw.get("outside", "0") or 0.0),
                }
            )
    rows.sort(key=lambda row: (float(row["elapsed_sec"]), int(row["event_idx"])))
    return rows


def feasible_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    return [row for row in rows if int(row["feasible"]) == 1]


def best_so_far_feasible_rows(rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    best_rows: List[Dict[str, object]] = []
    current = float("inf")
    for row in feasible_rows(rows):
        L = float(row["L"])
        if L < current:
            current = L
            best_rows.append(dict(row))
    return best_rows


def density_from_L(n: int, L: float) -> float:
    return (n * POLY_AREA) / (L * L)


def parse_header_L(path: Path) -> float:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for _ in range(5):
                line = handle.readline()
                if not line:
                    break
                if line.startswith("#") and " L=" in line:
                    return float(line.split(" L=")[1].split()[0])
    except Exception:
        return 0.0
    return 0.0


def read_config_rows(path: Path) -> List[Tuple[float, float, float]]:
    rows: List[Tuple[float, float, float]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                if line.strip().lower().startswith("i,cx,cy,theta_rad"):
                    continue
                parts = [part.strip() for part in line.split(",")]
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


def figure_to_rgba(fig: Figure) -> np.ndarray:
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    buffer = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))
    return buffer.copy()


def render_config_from_csv(csv_rel_path: str, workspace_dir: Path, figsize: Tuple[float, float] = (3.0, 3.0), dpi: int = 170) -> np.ndarray | None:
    if not csv_rel_path:
        return None
    csv_path = workspace_dir / csv_rel_path
    if not csv_path.exists():
        return None

    rows = read_config_rows(csv_path)
    if not rows:
        return None

    L = parse_header_L(csv_path)
    fig = Figure(figsize=figsize, dpi=dpi)
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
        margin = max(0.02, 0.02 * L)
        ax.set_xlim(-half - margin, half + margin)
        ax.set_ylim(-half - margin, half + margin)
    else:
        xs = [point[0] for cx, cy, th in rows for point in transform_poly(cx, cy, th)]
        ys = [point[1] for cx, cy, th in rows for point in transform_poly(cx, cy, th)]
        margin = 0.03
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

    image = figure_to_rgba(fig)
    plt.close(fig)
    return image
