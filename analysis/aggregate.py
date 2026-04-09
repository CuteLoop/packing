#!/usr/bin/env python3
"""
Phase 6 — analysis/aggregate.py
Reads a bisection CSV, plots L_best vs wall time, and computes packing fraction η.

Usage:
    python3 analysis/aggregate.py <bisection.csv> [--A_poly <float>]

If --A_poly is not given, it defaults to computing η assuming unit circles (π/4).
For our irregular polygon, pass the actual area from the C code.
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Aggregate bisection results")
    parser.add_argument("csv", help="Path to *_bisection.csv file")
    parser.add_argument("--A_poly", type=float, default=None,
                        help="Area of one polygon (default: π/4 for unit circle)")
    parser.add_argument("--outdir", default=None,
                        help="Output directory for plots (default: same as CSV)")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"Error: file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    # ---- Parse CSV ----
    import csv
    import math

    rows = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("Error: CSV has no data rows.", file=sys.stderr)
        sys.exit(1)

    # Validate required columns
    required = {"probe_idx", "wall_sec_end", "L_mid", "feasible", "min_energy",
                "min_feas", "L_best", "bracket_width", "N", "R", "method"}
    headers = set(rows[0].keys())
    missing = required - headers
    if missing:
        print(f"Error: missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(rows)} probes from {args.csv}")

    # Extract data
    N = int(rows[0]["N"])
    R = int(rows[0]["R"])
    method = rows[0]["method"]
    print(f"  N={N}  R={R}  method={method}")

    wall_times = []
    min_energies = []
    L_bests = []
    feasible_count = 0

    for row in rows:
        wall_times.append(float(row["wall_sec_end"]))
        min_energies.append(float(row["min_energy"]))
        lb = row["L_best"].strip()
        if lb:
            L_bests.append(float(lb))
        feasible_count += int(row["feasible"])

    print(f"  probes={len(rows)}  feasible_probes={feasible_count}")

    # ---- Compute η ----
    if args.A_poly is not None:
        A_poly = args.A_poly
    else:
        # Default: π/4 for unit circle (r=0.5), but print warning
        A_poly = math.pi * 0.25
        print(f"  WARNING: using default A_poly={A_poly:.6f} (unit circle r=0.5)")
        print(f"           Pass --A_poly with actual polygon area for correct η")

    if L_bests:
        L_final = L_bests[-1]
        eta = (N * A_poly) / (L_final * L_final)
        print(f"  L_best={L_final:.6f}")
        print(f"  η = N×A_poly / L² = {N}×{A_poly:.6f} / {L_final:.6f}² = {eta:.4f}")
        if eta > 0.65:
            print(f"  Rating: HERO-TIER (η > 0.65)")
        elif eta > 0.55:
            print(f"  Rating: SOLID (η > 0.55)")
        else:
            print(f"  Rating: needs improvement (η < 0.55)")
    else:
        L_final = None
        eta = None
        print(f"  No feasible L_best found; cannot compute η.")

    # ---- Plot ----
    outdir = args.outdir or os.path.dirname(args.csv) or "."
    os.makedirs(outdir, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: min energy vs wall time
        ax = axes[0]
        ax.plot(wall_times, min_energies, "b.-", markersize=4)
        ax.set_xlabel("Wall time (s)")
        ax.set_ylabel("Min energy")
        ax.set_title(f"Energy trace — {method} N={N} R={R}")
        ax.grid(True, alpha=0.3)

        # Plot 2: L_best vs probe index (only feasible probes)
        ax = axes[1]
        if L_bests:
            probe_indices = []
            lb_vals = []
            for i, row in enumerate(rows):
                lb = row["L_best"].strip()
                if lb:
                    probe_indices.append(i)
                    lb_vals.append(float(lb))
            ax.plot(probe_indices, lb_vals, "g.-", markersize=4)
            ax.set_xlabel("Probe index")
            ax.set_ylabel("L_best")
            ax.set_title(f"Bracket shrinkage — L_best(probe)")
            if eta is not None:
                ax.text(0.02, 0.02, f"η = {eta:.4f}",
                        transform=ax.transAxes, fontsize=12,
                        verticalalignment="bottom",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        else:
            ax.text(0.5, 0.5, "No feasible solutions",
                    transform=ax.transAxes, ha="center", va="center", fontsize=14)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        basename = os.path.splitext(os.path.basename(args.csv))[0]
        out_path = os.path.join(outdir, f"{basename}_trace.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Plot saved: {out_path}")

    except ImportError:
        print("  WARNING: matplotlib not available; skipping plot generation.")
        print("  Install with: pip install matplotlib")

    print("Done.")

if __name__ == "__main__":
    main()
