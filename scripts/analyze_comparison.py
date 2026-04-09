#!/usr/bin/env python3
"""
analyze_comparison.py

Parse sweep bisection CSVs and produce comparison plots:
  - Best L vs N (per method)
  - Wall-clock runtime vs N (per method)
  - Packing density η vs N (per method)
  - Bracket width vs N
  - Probes completed vs N

Usage (from run/HPC_DEMO):
  python3 scripts/analyze_comparison.py
  python3 scripts/analyze_comparison.py --glob "out/sweep_*_bisection.csv"
  python3 scripts/analyze_comparison.py --outdir analysis/comparison
"""

import argparse
import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Polygon area (must match BASE_V in HPC_parallel.c / base_geometry.c)
BASE_V = np.array([
    [  0.0,     0.8  ],
    [  0.125,   0.5  ],
    [  0.0625,  0.5  ],
    [  0.2,     0.25 ],
    [  0.1,     0.25 ],
    [  0.35,    0.0  ],
    [  0.075,   0.0  ],
    [  0.075,  -0.2  ],
    [ -0.075,  -0.2  ],
    [ -0.075,   0.0  ],
    [ -0.35,    0.0  ],
    [ -0.1,     0.25 ],
    [ -0.2,     0.25 ],
    [ -0.0625,  0.5  ],
    [ -0.125,   0.5  ],
], dtype=float)

def polygon_area(verts):
    x, y = verts[:, 0], verts[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

POLY_AREA = polygon_area(BASE_V)

# CSV filename pattern: out/sweep_{method}_N{n}_s{seed}_{method}_N{NNN}_s{seed}_bisection.csv
BISECTION_RE = re.compile(
    r"sweep_(?P<method>\w+)_N(?P<n>\d+)_s(?P<seed>\d+)_"
    r"\w+_N\d+_s\d+_bisection\.csv$"
)


def parse_bisection(path):
    """Parse a bisection CSV and return summary dict."""
    fname = os.path.basename(path)
    m = BISECTION_RE.search(fname)
    if not m:
        # Try more relaxed parsing from CSV content
        pass

    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return None

    method = rows[0].get("method", "")
    N = int(rows[0].get("N", 0))
    R = int(rows[0].get("R", 1))
    seed = int(rows[0].get("seed", 0))

    # Total wall time = last probe's wall_sec_end
    wall_total = max(float(r["wall_sec_end"]) for r in rows)

    # Count probes and feasible probes
    n_probes = len(rows)
    n_feasible = sum(1 for r in rows if int(r.get("feasible", 0)) == 1)

    # Best L
    L_bests = []
    for r in rows:
        lb = r.get("L_best", "").strip()
        if lb:
            try:
                L_bests.append(float(lb))
            except ValueError:
                pass
    L_best = min(L_bests) if L_bests else None

    # Final bracket
    last = rows[-1]
    bracket_width = float(last.get("bracket_width", 0))
    L_lo = float(last.get("L_lo", 0))
    L_hi = float(last.get("L_hi", 0))

    # Min energy seen
    min_energy = min(float(r.get("min_energy", 1e30)) for r in rows)

    # Density
    density = (N * POLY_AREA) / (L_best ** 2) if L_best and L_best > 0 else None

    return {
        "path": path,
        "method": method,
        "N": N,
        "R": R,
        "seed": seed,
        "wall_sec": wall_total,
        "n_probes": n_probes,
        "n_feasible": n_feasible,
        "L_best": L_best,
        "L_lo": L_lo,
        "L_hi": L_hi,
        "bracket_width": bracket_width,
        "min_energy": min_energy,
        "density": density,
    }


def main():
    ap = argparse.ArgumentParser(description="Compare sweep results across N and methods")
    ap.add_argument("--glob", default="out/sweep_*_bisection.csv",
                    help="Glob pattern for bisection CSVs")
    ap.add_argument("--outdir", default="analysis/comparison",
                    help="Output directory for plots and summary")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.glob))
    if not paths:
        print(f"No files found matching: {args.glob}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(paths)} bisection CSVs")

    records = []
    for p in paths:
        try:
            rec = parse_bisection(p)
            if rec and rec["N"] > 0:
                records.append(rec)
                print(f"  {rec['method']:5s} N={rec['N']:3d} seed={rec['seed']} "
                      f"L={rec['L_best'] or 'N/A':>10s} "
                      f"wall={rec['wall_sec']:.1f}s "
                      f"probes={rec['n_probes']} "
                      f"feas={rec['n_feasible']}"
                      if isinstance(rec['L_best'], str) else
                      f"  {rec['method']:5s} N={rec['N']:3d} seed={rec['seed']} "
                      f"L={rec['L_best']:10.4f} "
                      f"wall={rec['wall_sec']:.1f}s "
                      f"probes={rec['n_probes']} "
                      f"feas={rec['n_feasible']}")
        except Exception as e:
            print(f"  SKIP {p}: {e}", file=sys.stderr)

    if not records:
        print("No valid records parsed.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    # Write summary CSV
    summary_path = os.path.join(args.outdir, "sweep_summary.csv")
    fields = ["method", "N", "seed", "R", "L_best", "density", "wall_sec",
              "n_probes", "n_feasible", "bracket_width", "min_energy", "path"]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in sorted(records, key=lambda r: (r["N"], r["method"], r["seed"])):
            writer.writerow(rec)
    print(f"\nWrote: {summary_path}")

    # Group by method — for each (method, N), pick best seed (lowest L)
    best = defaultdict(dict)  # best[method][N] = record
    for rec in records:
        key = (rec["method"], rec["N"])
        if rec["L_best"] is None:
            continue
        prev = best[rec["method"]].get(rec["N"])
        if prev is None or rec["L_best"] < prev["L_best"]:
            best[rec["method"]][rec["N"]] = rec

    # Also compute mean across seeds
    by_method_n = defaultdict(list)
    for rec in records:
        by_method_n[(rec["method"], rec["N"])].append(rec)

    methods = sorted(best.keys())
    colors = {"ms": "#1f77b4", "erms": "#ff7f0e", "pt": "#2ca02c"}
    markers = {"ms": "o", "erms": "s", "pt": "^"}

    def plot_metric(metric, ylabel, title, filename, log_y=False, use_mean=False):
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in methods:
            if use_mean:
                ns, vals = [], []
                for (m, n), recs in sorted(by_method_n.items()):
                    if m != method:
                        continue
                    v = [r[metric] for r in recs if r[metric] is not None]
                    if v:
                        ns.append(n)
                        vals.append(np.mean(v))
            else:
                data = best[method]
                ns = sorted(data.keys())
                vals = [data[n][metric] for n in ns if data[n][metric] is not None]
                ns = [n for n in ns if data[n][metric] is not None]

            ax.plot(ns, vals,
                    marker=markers.get(method, "o"),
                    color=colors.get(method, None),
                    label=method.upper(),
                    linewidth=1.5, markersize=6)

        ax.set_xlabel("N (number of polygons)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        if log_y:
            ax.set_yscale("log")
        fig.tight_layout()
        outpath = os.path.join(args.outdir, filename)
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        print(f"  Plot: {outpath}")

    print("\nGenerating plots...")

    plot_metric("L_best", "Best L (side length)", "Best Packing Size vs N",
                "L_best_vs_N.png")

    plot_metric("wall_sec", "Wall time (seconds)", "Runtime vs N",
                "runtime_vs_N.png")

    plot_metric("density", "Packing density η = N·A_poly / L²",
                "Packing Density vs N", "density_vs_N.png")

    plot_metric("n_probes", "Probes completed", "Bisection Probes vs N",
                "probes_vs_N.png")

    plot_metric("bracket_width", "Final bracket width (L_hi - L_lo)",
                "Bracket Width vs N", "bracket_vs_N.png", log_y=True)

    # Mean runtime comparison (averaging over seeds)
    plot_metric("wall_sec", "Mean wall time (seconds)",
                "Mean Runtime vs N (averaged over seeds)",
                "mean_runtime_vs_N.png", use_mean=True)

    plot_metric("L_best", "Mean best L",
                "Mean Best L vs N (averaged over seeds)",
                "mean_L_vs_N.png", use_mean=True)

    # Summary table to stdout
    print("\n" + "=" * 70)
    print(f"{'Method':>6s} {'N':>4s} {'L_best':>10s} {'η':>8s} {'Time(s)':>8s} {'Probes':>6s}")
    print("-" * 70)
    for method in methods:
        for n in sorted(best[method].keys()):
            rec = best[method][n]
            eta = f"{rec['density']:.4f}" if rec["density"] else "N/A"
            lb = f"{rec['L_best']:.4f}" if rec["L_best"] else "N/A"
            print(f"{method:>6s} {n:4d} {lb:>10s} {eta:>8s} {rec['wall_sec']:8.1f} {rec['n_probes']:6d}")
    print("=" * 70)

    print(f"\nPolygon area = {POLY_AREA:.6f}")
    print("Done.")


if __name__ == "__main__":
    main()
