#!/usr/bin/env python3
"""
analyze.py — Post-process bench_results.csv.

Produces:
  1. Throughput plot   : Mpairs/s vs N, one line per method, faceted by regime
  2. Speedup plot      : C/B speedup ratio vs N, faceted by regime
  3. Rejection cascade : stacked bar for Method C (per regime, N=100)
  4. LaTeX tables      : primary stats + Method C diagnostic (printed to stdout)

Usage:
  python3 analyze.py [bench_results.csv]
"""

import sys
import csv
import math
from collections import defaultdict

# ── optional matplotlib ───────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[analyze] matplotlib not found — skipping plots, printing tables only")

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "bench_results.csv"

# ── load data ────────────────────────────────────────────────
rows = []
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append({
            "N":             int(row["N"]),
            "regime":        row["regime"],
            "method":        row["method"],
            "pairs_total":   int(row["pairs_total"]),
            "bp_pass":       int(row["broadphase_pass"]),
            "narrow":        int(row["narrow_calls"]),
            "collisions":    int(row["collisions"]),
            "time_s":        float(row["wall_time_s"]),
            "mpairs_s":      float(row["mpairs_per_s"]),
            "part_pairs":    int(row["part_pairs"]),
            "part_aabb_pass":int(row["part_aabb_pass"]),
            "hint_tests":    int(row["hint_tests"]),
            "hint_rejects":  int(row["hint_rejects"]),
            "sat_full":      int(row["sat_full_calls"]),
            "sat_hits":      int(row["sat_hits"]),
            "sat_axes":      int(row["sat_axes_tested"]),
            "avg_axes":      float(row["avg_axes_per_sat"]),
        })

N_values  = sorted(set(r["N"]      for r in rows))
regimes   = sorted(set(r["regime"] for r in rows))
methods   = ["A", "B", "C"]
METHOD_LABELS = {"A": "Triangulation", "B": "Baseline SAT", "C": "Cached SAT"}
COLORS = {"A": "#e74c3c", "B": "#3498db", "C": "#2ecc71"}

def get(method, regime, N, key):
    for r in rows:
        if r["method"]==method and r["regime"]==regime and r["N"]==N:
            return r[key]
    return None

# ── 1. Throughput plot ────────────────────────────────────────
if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, regime in zip(axes, regimes):
        for m in methods:
            ys = [get(m, regime, N, "mpairs_s") for N in N_values]
            ys = [y if y is not None else 0 for y in ys]
            ax.plot(N_values, ys, marker="o", label=METHOD_LABELS[m],
                    color=COLORS[m], linewidth=2)
        ax.set_title(regime.capitalize(), fontsize=12)
        ax.set_xlabel("N (polygons)")
        ax.set_ylabel("Mpairs / s")
        ax.set_xticks(N_values)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("Collision Throughput vs N", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("throughput.pdf", bbox_inches="tight")
    plt.savefig("throughput.png", dpi=150, bbox_inches="tight")
    print("[analyze] saved throughput.pdf / .png")
    plt.close()

# ── 2. Speedup plot (C vs B) ──────────────────────────────────
if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, regime in zip(axes, regimes):
        speedups = []
        for N in N_values:
            b = get("B", regime, N, "mpairs_s")
            c = get("C", regime, N, "mpairs_s")
            speedups.append(c/b if (b and b > 0) else 1.0)
        ax.bar(range(len(N_values)), speedups,
               color=COLORS["C"], alpha=0.8, edgecolor="black")
        ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
        ax.set_xticks(range(len(N_values)))
        ax.set_xticklabels([str(n) for n in N_values])
        ax.set_xlabel("N (polygons)")
        ax.set_title(regime.capitalize())
    axes[0].set_ylabel("Speedup (C / B)")
    fig.suptitle("Cached SAT Speedup Over Baseline SAT", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("speedup.pdf", bbox_inches="tight")
    plt.savefig("speedup.png", dpi=150, bbox_inches="tight")
    print("[analyze] saved speedup.pdf / .png")
    plt.close()

# ── 3. Rejection cascade bar (Method C, N=100) ───────────────
if HAS_MPL:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, regime in zip(axes, regimes):
        r = next((x for x in rows
                  if x["method"]=="C" and x["regime"]==regime
                  and x["N"]==100), None)
        if r is None:
            ax.set_visible(False)
            continue
        # Cascade fractions (normalised to part_pairs)
        pp   = max(r["part_pairs"], 1)
        cats  = ["Part AABB\npass", "Hint-axis\npass", "Full SAT\ncalls", "SAT\nhits"]
        vals  = [
            100.0 * r["part_aabb_pass"] / pp,
            100.0 * (r["part_aabb_pass"] - r["hint_rejects"]) / pp,
            100.0 * r["sat_full"]       / pp,
            100.0 * r["sat_hits"]       / pp,
        ]
        bars = ax.bar(cats, vals,
                      color=["#3498db","#f39c12","#e74c3c","#9b59b6"],
                      alpha=0.85, edgecolor="black")
        ax.set_ylabel("% of part pairs")
        ax.set_ylim(0, 110)
        ax.set_title(f"Cascade — {regime} (N=100)")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Method C Filter Cascade (N=100)", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig("cascade.pdf", bbox_inches="tight")
    plt.savefig("cascade.png", dpi=150, bbox_inches="tight")
    print("[analyze] saved cascade.pdf / .png")
    plt.close()

# ── 4. LaTeX table: primary metrics ──────────────────────────
def pct(num, den):
    return f"{100.0*num/den:.1f}\\%" if den > 0 else "---"

print()
print("% ─── PRIMARY METRICS TABLE ──────────────────────────────────────")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Collision benchmark: per-method summary across regimes.}")
print("\\label{tab:bench-primary}")
print("\\begin{tabular}{llrrrrrr}")
print("\\toprule")
print("N & Method & Pairs & BP pass & Narrow & Collisions & Time (s) & Mpairs/s \\\\")
print("\\midrule")
for N in N_values:
    first_N = True
    for regime in regimes:
        first_r = True
        for method in methods:
            r = next((x for x in rows
                      if x["N"]==N and x["regime"]==regime
                      and x["method"]==method), None)
            if r is None:
                continue
            N_str  = str(N)      if (first_N and first_r) else ""
            re_str = regime      if first_r               else ""
            first_N = False
            first_r = False
            print(f"  {N_str} & {re_str} & {method} & "
                  f"{r['pairs_total']:,} & "
                  f"{pct(r['bp_pass'], r['pairs_total'])} & "
                  f"{r['narrow']:,} & "
                  f"{r['collisions']:,} & "
                  f"{r['time_s']:.3f} & "
                  f"{r['mpairs_s']:.2f} \\\\")
    print("\\midrule")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# ── 5. LaTeX table: Method C cascade ─────────────────────────
print()
print("% ─── METHOD C DIAGNOSTIC TABLE ──────────────────────────────────")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Method C filter cascade diagnostics (dense regime).}")
print("\\label{tab:bench-cascade}")
print("\\begin{tabular}{rrrrrrrr}")
print("\\toprule")
print("N & Part pairs & Part AABB\\% & Hint tests & Hint rej\\% "
      "& SAT calls & SAT hits & Avg axes \\\\")
print("\\midrule")
for N in N_values:
    r = next((x for x in rows
              if x["N"]==N and x["regime"]=="dense"
              and x["method"]=="C"), None)
    if r is None:
        continue
    print(f"  {N} & {r['part_pairs']:,} & "
          f"{pct(r['part_aabb_pass'], r['part_pairs'])} & "
          f"{r['hint_tests']:,} & "
          f"{pct(r['hint_rejects'], r['hint_tests'])} & "
          f"{r['sat_full']:,} & "
          f"{r['sat_hits']:,} & "
          f"{r['avg_axes']:.1f} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# ── 6. Speedup summary to stdout ──────────────────────────────
print()
print("=== Speedup C vs B (Mpairs/s ratio, dense regime) ===")
print(f"{'N':>6}  {'B':>8}  {'C':>8}  {'speedup':>8}")
for N in N_values:
    b = get("B", "dense", N, "mpairs_s")
    c = get("C", "dense", N, "mpairs_s")
    if b and c:
        print(f"{N:>6}  {b:>8.2f}  {c:>8.2f}  {c/b:>8.2f}×")
