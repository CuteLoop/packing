#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt


def _downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Uniformly downsample rows to at most max_points (keeps endpoints)."""
    n = len(df)
    if max_points <= 0 or n <= max_points:
        return df
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    idx = np.unique(idx)
    return df.iloc[idx].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Pretty plots for SA trace CSV.")
    ap.add_argument("trace_csv", help="Path to *_trace.csv")
    ap.add_argument("--out-prefix", default=None,
                    help="Output prefix for png files. Default: derive from trace path.")
    ap.add_argument("--window", type=int, default=2000,
                    help="Rolling window for acceptance rate (default: 2000)")
    ap.add_argument("--max-points", type=int, default=25000,
                    help="Max points to plot per time series (default: 25000)")
    ap.add_argument("--bins", type=int, default=160,
                    help="Histogram bins for ΔE (default: 160)")
    ap.add_argument("--clip-quantile", type=float, default=0.995,
                    help="Clip |ΔE| to this quantile for histogram stability (default: 0.995). Set 1.0 to disable.")
    args = ap.parse_args()

    df = pd.read_csv(args.trace_csv)

    required = {"step", "E", "T", "accepted", "moved"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in CSV: {sorted(missing)}")

    # Derived series (computed on full df)
    df["E_best"] = df["E"].cummin()
    df["acc_roll"] = df["accepted"].rolling(
        args.window, min_periods=max(1, args.window // 10)
    ).mean()
    df["dE"] = df["E"].diff().fillna(0.0)

    # Output prefix
    if args.out_prefix is None:
        outdir = os.path.dirname(os.path.abspath(args.trace_csv)) or "."
        base = os.path.basename(args.trace_csv).replace("_trace.csv", "")
        out_prefix = os.path.join(outdir, base)
    else:
        out_prefix = args.out_prefix

    out_plots = f"{out_prefix}_plots.png"
    out_dE = f"{out_prefix}_dE.png"
    out_moved = f"{out_prefix}_moved.png"

    # Downsample for the main timeseries figure (keeps histograms robust)
    dfp = _downsample(df, args.max_points)

    # -------------------- Style (minimal, readable) --------------------
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.7,
    })

    # -------------------- Main 2×2 panel figure --------------------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)

    # Panel A: Energy (total + best)
    ax = axes[0, 0]
    ax.plot(dfp["step"], dfp["E"], label="E")
    ax.plot(dfp["step"], dfp["E_best"], label="best")
    ax.set_title("Energy")
    ax.set_xlabel("step")
    ax.set_ylabel("E")
    ax.legend(loc="best", frameon=True)

    # Panel B: Temperature (log)
    ax = axes[0, 1]
    ax.plot(dfp["step"], dfp["T"])
    ax.set_yscale("log")
    ax.set_title("Temperature")
    ax.set_xlabel("step")
    ax.set_ylabel("T")

    # Panel C: Rolling acceptance
    ax = axes[1, 0]
    ax.plot(dfp["step"], dfp["acc_roll"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Acceptance (roll {args.window})")
    ax.set_xlabel("step")
    ax.set_ylabel("rate")

    # Panel D: Penalty decomposition (pair/wall)
    ax = axes[1, 1]
    if "E_pair" in dfp.columns and "E_wall" in dfp.columns:
        ax.plot(dfp["step"], dfp["E_pair"], label="pair")
        ax.plot(dfp["step"], dfp["E_wall"], label="wall")
        ax.set_title("Penalties")
        ax.set_xlabel("step")
        ax.set_ylabel("E")
        ax.legend(loc="best", frameon=True)
    else:
        ax.text(0.5, 0.5, "E_pair/E_wall not logged", ha="center", va="center")
        ax.set_title("Penalties")
        ax.set_xticks([])
        ax.set_yticks([])

    # Clean title (short)
    fig.suptitle(os.path.basename(args.trace_csv), y=1.02, fontsize=12)

    fig.savefig(out_plots, bbox_inches="tight")
    plt.close(fig)

    # -------------------- ΔE histogram (separate) --------------------
    dE = df["dE"].to_numpy()
    if args.clip_quantile < 1.0:
        q = np.quantile(np.abs(dE), args.clip_quantile)
        dE = dE[np.abs(dE) <= q]

    fig2 = plt.figure(figsize=(7.0, 5.0))
    plt.hist(dE, bins=args.bins)
    plt.title(r"$\Delta E = E_t - E_{t-1}$")
    plt.xlabel(r"$\Delta E$")
    plt.ylabel("count")
    plt.grid(True, alpha=0.22)
    plt.tight_layout()
    fig2.savefig(out_dE, bbox_inches="tight")
    plt.close(fig2)

    # -------------------- moved index histogram (separate) --------------------
    moved = df["moved"].to_numpy()
    mmin = int(np.min(moved))
    mmax = int(np.max(moved))
    bins = (mmax - mmin + 1) if mmax >= mmin else 10

    fig3 = plt.figure(figsize=(7.0, 5.0))
    plt.hist(moved, bins=bins)
    plt.title("Moved index")
    plt.xlabel("index")
    plt.ylabel("count")
    plt.grid(True, alpha=0.22)
    plt.tight_layout()
    fig3.savefig(out_moved, bbox_inches="tight")
    plt.close(fig3)

    print("Saved:")
    print(" ", out_plots)
    print(" ", out_dE)
    print(" ", out_moved)


if __name__ == "__main__":
    main()
