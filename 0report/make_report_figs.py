from __future__ import annotations

import csv
import math
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "0report" / "img"
POLY_AREA = 0.245625

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

BEST_CONFIGS = {
    20: ROOT / "out" / "N020" / "graph_erms_1h" / "N020_erms_s1000_r0_best_state.csv",
    50: ROOT / "out" / "N050" / "graph_ms_1h" / "N050_ms_s2000_r1_best_state.csv",
    100: ROOT / "out" / "N100" / "graph_erms_1h" / "N100_erms_s2000_r1_best_state.csv",
    200: ROOT / "2025" / "old" / "csv" / "N200_job5490050_task2_best_polys_N200.csv",
}

TIMESERIES_SOURCES = {
    5: ("history_glob", ROOT / "2025" / "old" / "logs" / "N5_*_history_log.csv"),
    10: ("history_glob", ROOT / "2025" / "old" / "logs" / "N10_*_history_log.csv"),
    20: ("bisection", ROOT / "out" / "N020" / "graph_erms_1h" / "N020_erms_s1000_r0_bisection.csv"),
    25: ("history_glob", ROOT / "2025" / "old" / "logs" / "N25_*_history_log.csv"),
    50: ("history_glob", ROOT / "2025" / "old" / "logs" / "N50_*_history_log.csv"),
    100: ("history_glob", ROOT / "2025" / "old" / "logs" / "N100_*_history_log.csv"),
    200: ("history_glob", ROOT / "2025" / "old" / "logs" / "N200_*_history_log.csv"),
}

REPORT_FIGURES = {
    "feasible_L_all_timeseries_until_good_enough.png": ROOT / "2025" / "old" / "img" / "feasible_L_all_timeseries_until_good_enough.png",
    "feasible_L_timeseries_N5_N10_N25_N50_N100_N200.png": ROOT / "2025" / "old" / "img" / "feasible_L_timeseries_N5_N10_N25_N50_N100_N200.png",
    "feasible_best_density_timeseries.png": ROOT / "2025" / "old" / "img" / "feasible_best_density_timeseries.png",
    "feasible_time_to_best_by_n.png": ROOT / "2025" / "old" / "img" / "feasible_time_to_best_by_n.png",
    "per_n_configs/N005_best_config.png": ROOT / "2025" / "old" / "img" / "per_n_configs" / "N005_best_config.png",
    "per_n_configs/N010_best_config.png": ROOT / "2025" / "old" / "img" / "per_n_configs" / "N010_best_config.png",
    "per_n_configs/N025_best_config.png": ROOT / "2025" / "old" / "img" / "per_n_configs" / "N025_best_config.png",
    "per_n_configs/N050_best_config.png": ROOT / "2025" / "old" / "img" / "per_n_configs" / "N050_best_config.png",
    "per_n_configs/N100_best_config.png": ROOT / "2025" / "old" / "img" / "per_n_configs" / "N100_best_config.png",
    "per_n_configs/N200_best_config.png": ROOT / "2025" / "old" / "img" / "per_n_configs" / "N200_best_config.png",
}


def read_old_density_points() -> Dict[int, float]:
    path = ROOT / "2025" / "old" / "analysis" / "feasible_n_statistics.csv"
    out: Dict[int, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            n = int(row["N"])
            out[n] = float(row["best_of_best_density"])
    return out


HEADER_RE = re.compile(
    r"L=(?P<L>[-+eE0-9\.]+)\s+best_feas=(?P<best_feas>[-+eE0-9\.]+)\s+N=(?P<N>\d+)"
)


def read_dense_density_points() -> Dict[int, float]:
    out: Dict[int, float] = {}

    for path in (ROOT / "2025" / "csv").glob("*_best_polys_N*.csv"):
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline().strip()
        match = HEADER_RE.search(header)
        if not match:
            continue

        n = int(match.group("N"))
        l_val = float(match.group("L"))
        best_feas = float(match.group("best_feas"))
        if abs(best_feas) > 1e-12 or l_val <= 0.0:
            continue

        density = (n * POLY_AREA) / (l_val * l_val)
        if n not in out or density > out[n]:
            out[n] = density

    for n, path in BEST_CONFIGS.items():
        l_val = parse_header_L(path)
        density = (float(n) * POLY_AREA) / (l_val * l_val)
        if n not in out or density > out[n]:
            out[n] = density

    return out


def parse_header_L(path: Path) -> float:
    with path.open("r", encoding="utf-8") as handle:
        for _ in range(5):
            line = handle.readline()
            if not line:
                break
            if line.startswith("#") and " L=" in line:
                return float(line.split(" L=")[1].split()[0])
    raise ValueError(f"Could not parse L from {path}")


def read_config_rows(path: Path):
    rows = []
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
    return rows


def transform_poly(cx: float, cy: float, theta: float):
    c = math.cos(theta)
    s = math.sin(theta)
    return [(c * x - s * y + cx, s * x + c * y + cy) for x, y in BASE_V]


def render_packing(csv_path: Path, out_path: Path) -> None:
    rows = read_config_rows(csv_path)
    L = parse_header_L(csv_path)

    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=220)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    half = 0.5 * L
    ax.add_patch(Rectangle((-half, -half), L, L, fill=False, linewidth=1.8, edgecolor="black"))

    for cx, cy, theta in rows:
        poly = transform_poly(cx, cy, theta)
        ax.add_patch(
            Polygon(poly, closed=True, facecolor="#94a3b8", edgecolor="#0f172a", linewidth=0.45, alpha=0.55)
        )

    margin = max(0.03, 0.02 * L)
    ax.set_xlim(-half - margin, half + margin)
    ax.set_ylim(-half - margin, half + margin)
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def read_bisection_curve(path: Path):
    xs = []
    ys = []
    best = float("inf")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            t = float(row["wall_sec_end"])
            lbest_raw = (row.get("L_best") or "").strip()
            lmid_raw = (row.get("L_mid") or "").strip()
            feasible = (row.get("feasible") or "").strip() in {"1", "1.0", "true", "True"}
            candidate = None
            if lbest_raw:
                candidate = float(lbest_raw)
            elif feasible and lmid_raw:
                candidate = float(lmid_raw)
            if candidate is None:
                continue
            best = min(best, candidate)
            xs.append(t)
            ys.append(best)
    return xs, ys


def read_history_curve(path: Path):
    xs = []
    ys = []
    best = float("inf")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            feasible = (row.get("feasible") or "").strip() == "1"
            if not feasible:
                continue
            t = float(row["elapsed_sec"])
            candidate = float(row["L"])
            best = min(best, candidate)
            xs.append(t)
            ys.append(best)
    return xs, ys


def read_history_best_envelope(paths: List[Path]) -> Tuple[List[float], List[float]]:
    events: List[Tuple[float, float]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                feasible = (row.get("feasible") or "").strip() == "1"
                if not feasible:
                    continue
                events.append((float(row["elapsed_sec"]), float(row["L"])))

    if not events:
        return [], []

    events.sort(key=lambda item: item[0])
    xs: List[float] = []
    ys: List[float] = []
    best = float("inf")
    for t, l in events:
        if l < best:
            best = l
        xs.append(t)
        ys.append(best)
    return xs, ys


def make_timeseries(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=220)
    colors = {
        5: "#0f766e",
        10: "#0891b2",
        20: "#1d4ed8",
        25: "#3b82f6",
        50: "#059669",
        100: "#dc2626",
        200: "#7c3aed",
    }

    for n, (kind, path) in sorted(TIMESERIES_SOURCES.items()):
        if kind == "bisection":
            xs, ys = read_bisection_curve(path)
        elif kind == "history":
            xs, ys = read_history_curve(path)
        else:
            xs, ys = read_history_best_envelope(sorted(path.parent.glob(path.name)))
        if not xs:
            continue
        ax.step(xs, ys, where="post", linewidth=2.0, color=colors[n], label=f"N={n}")

    ax.set_xlabel("Wall time (s)")
    ax.set_ylabel("Best feasible L")
    ax.set_title("Best feasible L vs wall time (one time series per N)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_density_scaling_plots(out_dir: Path) -> None:
    density = read_dense_density_points()

    n_vals = sorted(density)
    y = [density[n] for n in n_vals]
    x_inv = [1.0 / math.sqrt(float(n)) for n in n_vals]

    fig, ax = plt.subplots(figsize=(6.2, 4.3), dpi=220)
    ax.plot(n_vals, y, marker="o", markersize=2.6, linewidth=1.0)
    ax.set_xlabel("N")
    ax.set_ylabel("Density ρ = N·A(P)/L²")
    ax.set_title("Packing density vs N")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "density_vs_N.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.3), dpi=220)
    ax.plot(x_inv, y, marker="o", markersize=2.6, linewidth=1.0)
    ax.set_xlabel("1/√N")
    ax.set_ylabel("Density ρ")
    ax.set_title("Density vs 1/√N")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "density_vs_inv_sqrtN.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for out_name, src_path in REPORT_FIGURES.items():
        out_path = OUT_DIR / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, out_path)
    for n, path in BEST_CONFIGS.items():
        render_packing(path, OUT_DIR / f"best_N{n:03d}.png")
    make_timeseries(OUT_DIR / "timeseries_bestL.png")
    make_density_scaling_plots(OUT_DIR)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
