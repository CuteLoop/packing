#!/usr/bin/env python3
"""
analyze_sweep.py

Parse HPC_parallel outputs and compute packing-quality metrics across N.

Expected inputs (default):
  csv/*_best_polys_N###.csv

Outputs:
  analysis/summary.csv
  analysis/plots/*.png

Usage examples:
  python3 scripts/analyze_sweep.py
  python3 scripts/analyze_sweep.py --csv_glob "csv/*_checkpoint_N*.csv"
  python3 scripts/analyze_sweep.py --outdir analysis --bins 36 --boundary_k 2.0

Notes:
- Reads L and best_feas from the header line written by write_polys_csv():
    # prefix=... run_id=... seed=... L=... best_feas=... N=...
- Assumes the polygon is exactly the BASE_V used in HPC_parallel.c (hardcoded below).
"""

import argparse
import glob
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --- Polygon definition: must match HPC_parallel.c BASE_V ---
BASE_V = np.array([
    [  0.0,     0.8  ],  # 0 tip
    [  0.125,   0.5  ],  # 1
    [  0.0625,  0.5  ],  # 2
    [  0.2,     0.25 ],  # 3
    [  0.1,     0.25 ],  # 4
    [  0.35,    0.0  ],  # 5
    [  0.075,   0.0  ],  # 6
    [  0.075,  -0.2  ],  # 7
    [ -0.075,  -0.2  ],  # 8
    [ -0.075,   0.0  ],  # 9
    [ -0.35,    0.0  ],  # 10
    [ -0.1,     0.25 ],  # 11
    [ -0.2,     0.25 ],  # 12
    [ -0.0625,  0.5  ],  # 13
    [ -0.125,   0.5  ],  # 14
], dtype=float)


def polygon_area(verts: np.ndarray) -> float:
    """Shoelace formula (absolute area)."""
    x = verts[:, 0]
    y = verts[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    return 0.5 * abs(np.sum(x * y2 - x2 * y))


def bounding_radius(verts: np.ndarray) -> float:
    """Max distance from origin among vertices."""
    return float(np.sqrt(np.max(np.sum(verts**2, axis=1))))


POLY_AREA = polygon_area(BASE_V)
BR = bounding_radius(BASE_V)


@dataclass
class RunRecord:
    path: str
    prefix: str
    run_id: Optional[int]
    seed: Optional[int]
    N: int
    L: float
    best_feas: Optional[float]
    density: float
    area_gap: float
    area_gap_frac: float
    boundary_fraction: float
    nn_mean: float
    nn_std: float
    orient_entropy: float


HEADER_RE = re.compile(
    r"#\s*prefix=(?P<prefix>\S+)\s+run_id=(?P<run_id>\d+)\s+seed=(?P<seed>\d+)\s+"
    r"L=(?P<L>[-+eE0-9\.]+)\s+best_feas=(?P<best_feas>[-+eE0-9\.]+)\s+N=(?P<N>\d+)"
)


def parse_header(lines: List[str]) -> Dict[str, Optional[str]]:
    """Extract header fields; return dict with keys or None."""
    for line in lines:
        if not line.startswith("#"):
            break
        m = HEADER_RE.search(line.strip())
        if m:
            return m.groupdict()
    return {}


def load_positions_csv(path: str) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    """Load csv rows i,cx,cy,theta_rad plus parsed header metadata."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    meta = parse_header(lines)

    # Pandas: skip comment lines beginning with '#'
    df = pd.read_csv(path, comment="#")
    # Expected columns: i,cx,cy,theta_rad
    required = {"i", "cx", "cy", "theta_rad"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    return df, meta


def nearest_neighbor_stats(xy: np.ndarray) -> Tuple[float, float]:
    """
    Compute nearest-neighbor distances.
    O(N^2) but N<=200 so it's fine.
    """
    n = xy.shape[0]
    if n <= 1:
        return 0.0, 0.0

    # Pairwise squared distances
    d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(np.min(d2, axis=1))
    return float(np.mean(nn)), float(np.std(nn))


def orientation_entropy(thetas: np.ndarray, bins: int = 36) -> float:
    """
    Histogram entropy on [0, 2pi).
    Returns entropy in nats (ln base e).
    """
    if thetas.size == 0:
        return 0.0
    twopi = 2.0 * math.pi
    th = np.mod(thetas, twopi)
    hist, _ = np.histogram(th, bins=bins, range=(0.0, twopi), density=False)
    p = hist.astype(float)
    p_sum = p.sum()
    if p_sum <= 0:
        return 0.0
    p /= p_sum
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def boundary_fraction(xy: np.ndarray, L: float, br: float, k: float = 2.0) -> float:
    """
    Fraction of centers within distance k*br of any square boundary.
    Square is [-L/2, L/2]^2.
    """
    if xy.size == 0:
        return 0.0
    half = 0.5 * L
    d = k * br
    x = xy[:, 0]
    y = xy[:, 1]
    near = (np.abs(x) >= (half - d)) | (np.abs(y) >= (half - d))
    return float(np.mean(near))


def safe_float(x: Optional[str]) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def safe_int(x: Optional[str]) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except Exception:
        return None


def compute_record(path: str, bins: int, boundary_k: float) -> RunRecord:
    df, meta = load_positions_csv(path)

    # Determine N, L from metadata; fall back to dataframe length if needed
    N_meta = safe_int(meta.get("N"))
    L = safe_float(meta.get("L"))
    best_feas = safe_float(meta.get("best_feas"))
    prefix = meta.get("prefix") or os.path.basename(path).split("_best_polys_")[0]
    run_id = safe_int(meta.get("run_id"))
    seed = safe_int(meta.get("seed"))

    N_df = int(df.shape[0])
    N = N_meta if (N_meta is not None and N_meta > 0) else N_df
    if L is None or L <= 0:
        raise ValueError(f"{path}: could not parse positive L from header")

    xy = df[["cx", "cy"]].to_numpy(dtype=float)
    thetas = df["theta_rad"].to_numpy(dtype=float)

    dens = (N * POLY_AREA) / (L * L)
    gap = (L * L) - (N * POLY_AREA)
    gap_frac = 1.0 - dens

    nn_m, nn_s = nearest_neighbor_stats(xy)
    ent = orientation_entropy(thetas, bins=bins)
    bfrac = boundary_fraction(xy, L, BR, k=boundary_k)

    return RunRecord(
        path=path,
        prefix=str(prefix),
        run_id=run_id,
        seed=seed,
        N=N,
        L=float(L),
        best_feas=best_feas,
        density=float(dens),
        area_gap=float(gap),
        area_gap_frac=float(gap_frac),
        boundary_fraction=float(bfrac),
        nn_mean=float(nn_m),
        nn_std=float(nn_s),
        orient_entropy=float(ent),
    )


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def save_plot(x, y, xlabel, ylabel, title, outpath):
    plt.figure()
    plt.plot(x, y, marker="o", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_glob", default="csv/*_best_polys_N*.csv",
                    help="Glob for input CSVs (best or checkpoint).")
    ap.add_argument("--outdir", default="analysis", help="Output directory.")
    ap.add_argument("--bins", type=int, default=36, help="Orientation histogram bins.")
    ap.add_argument("--boundary_k", type=float, default=2.0,
                    help="Boundary thickness in units of bounding radius (k*br).")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    plots_dir = os.path.join(args.outdir, "plots")
    ensure_dir(plots_dir)

    paths = sorted(glob.glob(args.csv_glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.csv_glob}")

    records: List[RunRecord] = []
    for p in paths:
        try:
            rec = compute_record(p, bins=args.bins, boundary_k=args.boundary_k)
            records.append(rec)
        except Exception as e:
            print(f"[skip] {p}: {e}")

    if not records:
        raise SystemExit("No valid records parsed.")

    df = pd.DataFrame([r.__dict__ for r in records])

    # Sort by N then by best_feas if available
    df.sort_values(by=["N", "best_feas"], ascending=[True, True], inplace=True)

    # If multiple runs exist per N, keep the best (lowest L, then lowest feas)
    # Usually you will have exactly one per N per job, but this makes it robust.
    df_best = (
        df.sort_values(by=["N", "L", "best_feas"], ascending=[True, True, True])
          .groupby("N", as_index=False)
          .first()
          .copy()
    )

    summary_path = os.path.join(args.outdir, "summary.csv")
    df_best.to_csv(summary_path, index=False)

    print("=== Parsed sweep summary ===")
    print(f"Polygon area = {POLY_AREA:.12g}")
    print(f"Bounding radius = {BR:.12g}")
    print(f"Wrote: {summary_path}")
    print(f"Ns: {df_best['N'].min()}..{df_best['N'].max()}  (count={len(df_best)})")

    # Plots
    N = df_best["N"].to_numpy()
    inv_sqrtN = 1.0 / np.sqrt(N.astype(float))

    save_plot(
        N, df_best["density"].to_numpy(),
        xlabel="N", ylabel="Density ρ = N·area(P)/L²",
        title="Packing density vs N",
        outpath=os.path.join(plots_dir, "density_vs_N.png"),
    )

    save_plot(
        inv_sqrtN, df_best["density"].to_numpy(),
        xlabel="1/√N", ylabel="Density ρ",
        title="Density vs 1/√N (boundary scaling diagnostic)",
        outpath=os.path.join(plots_dir, "density_vs_inv_sqrtN.png"),
    )

    save_plot(
        N, df_best["area_gap"].to_numpy(),
        xlabel="N", ylabel="Area gap ΔA = L² - N·area(P)",
        title="Absolute empty area vs N",
        outpath=os.path.join(plots_dir, "area_gap_vs_N.png"),
    )

    save_plot(
        N, df_best["boundary_fraction"].to_numpy(),
        xlabel="N", ylabel=f"Boundary fraction (within {args.boundary_k}·br)",
        title="Fraction of centers near boundary vs N",
        outpath=os.path.join(plots_dir, "boundary_fraction_vs_N.png"),
    )

    save_plot(
        N, df_best["nn_mean"].to_numpy(),
        xlabel="N", ylabel="Mean nearest-neighbor distance",
        title="Mean nearest-neighbor distance vs N",
        outpath=os.path.join(plots_dir, "nn_mean_vs_N.png"),
    )

    save_plot(
        N, df_best["orient_entropy"].to_numpy(),
        xlabel="N", ylabel=f"Orientation entropy (bins={args.bins}, nats)",
        title="Orientation entropy vs N",
        outpath=os.path.join(plots_dir, "orientation_entropy_vs_N.png"),
    )

    print(f"Wrote plots to: {plots_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
