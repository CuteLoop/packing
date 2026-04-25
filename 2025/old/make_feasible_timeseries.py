#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
from collections import defaultdict
from typing import Dict, List


def parse_prefix(prefix: str) -> Dict[str, str]:
    # Supports both old and run-tagged prefixes, e.g.:
    # N10_job5489985_node_00_i12n0_w001
    # N10_20260424_193109_job5489989_node_00_i12n0_w001
    out = {
        "N": "",
        "job_id": "",
        "node_tag": "",
        "host": "",
        "worker": "",
        "run_tag": "",
    }

    m = re.match(
        r"^N(?P<N>\d+)_"
        r"(?:(?P<run_tag>.+?)_)?"
        r"job(?P<job>\d+)_"
        r"(?P<node>node_\d+)_"
        r"(?P<host>[^_]+)_"
        r"w(?P<worker>\d+)$",
        prefix,
    )
    if not m:
        return out

    out["N"] = m.group("N") or ""
    out["job_id"] = m.group("job") or ""
    out["node_tag"] = m.group("node") or ""
    out["host"] = m.group("host") or ""
    out["worker"] = m.group("worker") or ""
    out["run_tag"] = m.group("run_tag") or ""
    return out


def parse_feasible_rows(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    prefix = os.path.basename(path).replace("_history_log.csv", "")
    meta = parse_prefix(prefix)

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("feasible", "")).strip() != "1":
                continue
            try:
                elapsed = float(row["elapsed_sec"])
                L = float(row["L"])
                event_idx = int(row["event_idx"])
            except (ValueError, KeyError):
                continue

            rows.append(
                {
                    "N": int(meta["N"]) if meta["N"] else "",
                    "run_tag": meta["run_tag"],
                    "job_id": meta["job_id"],
                    "node_tag": meta["node_tag"],
                    "host": meta["host"],
                    "worker": meta["worker"],
                    "prefix": prefix,
                    "event_idx": event_idx,
                    "stage": row.get("stage", ""),
                    "elapsed_sec": elapsed,
                    "L": L,
                    "source_history_log": path,
                }
            )

    rows.sort(key=lambda r: r["elapsed_sec"])

    best = float("inf")
    for r in rows:
        best = min(best, float(r["L"]))
        r["best_so_far_run"] = best

    return rows


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_run_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_run: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_run[str(r["prefix"])].append(r)

    out: List[Dict[str, object]] = []
    for prefix, rr in by_run.items():
        rr.sort(key=lambda x: float(x["elapsed_sec"]))
        out.append(
            {
                "N": rr[0]["N"],
                "run_tag": rr[0]["run_tag"],
                "job_id": rr[0]["job_id"],
                "node_tag": rr[0]["node_tag"],
                "host": rr[0]["host"],
                "worker": rr[0]["worker"],
                "prefix": prefix,
                "feasible_points": len(rr),
                "first_feasible_t": rr[0]["elapsed_sec"],
                "last_feasible_t": rr[-1]["elapsed_sec"],
                "best_feasible_L": min(float(x["L"]) for x in rr),
                "final_best_so_far_run": rr[-1]["best_so_far_run"],
            }
        )

    out.sort(key=lambda x: (int(x["N"]) if x["N"] != "" else 10**9, str(x["prefix"])))
    return out


def maybe_plot(rows: List[Dict[str, object]], path: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot")
        return

    by_n: Dict[int, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["N"] == "":
            continue
        by_n[int(r["N"])][str(r["prefix"])].append(r)

    n_values = sorted(by_n.keys())
    if not n_values:
        print("No plottable rows; skipping plot")
        return

    fig_h = max(4, 2 + 2.2 * len(n_values))
    fig, axes = plt.subplots(len(n_values), 1, figsize=(10, fig_h), squeeze=False)

    for i, n in enumerate(n_values):
        ax = axes[i][0]
        runs = by_n[n]
        for prefix, rr in sorted(runs.items()):
            rr.sort(key=lambda x: float(x["elapsed_sec"]))
            t = [float(x["elapsed_sec"]) for x in rr]
            l = [float(x["L"]) for x in rr]
            b = [float(x["best_so_far_run"]) for x in rr]
            ax.scatter(t, l, s=10, alpha=0.2)
            ax.step(t, b, where="post", linewidth=1.6, alpha=0.85)

        ax.set_title(f"Feasible L vs Time (N={n})")
        ax.set_xlabel("elapsed_sec")
        ax.set_ylabel("L")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build feasible-only time series (L vs elapsed time) from *_history_log.csv files"
    )
    parser.add_argument("--logs-dir", default="logs", help="Directory containing *_history_log.csv")
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default="*_history_log.csv",
        help="Glob pattern inside logs dir (default: *_history_log.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis",
        help="Directory for generated CSV/PNG outputs (default: analysis)",
    )
    parser.add_argument("--n", type=int, default=None, help="Optional N filter")
    parser.add_argument("--plot", default="feasible_L_timeseries.png", help="Plot filename")
    args = parser.parse_args()

    pattern = os.path.join(args.logs_dir, args.glob_pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No history logs found for pattern: {pattern}")

    all_rows: List[Dict[str, object]] = []
    for path in files:
        all_rows.extend(parse_feasible_rows(path))

    if args.n is not None:
        all_rows = [r for r in all_rows if r["N"] == args.n]

    if not all_rows:
        raise SystemExit("No feasible rows found in selected history logs")

    all_rows.sort(
        key=lambda r: (
            int(r["N"]) if r["N"] != "" else 10**9,
            str(r["prefix"]),
            float(r["elapsed_sec"]),
        )
    )

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    ts_csv = os.path.join(out_dir, "feasible_L_timeseries.csv")
    summary_csv = os.path.join(out_dir, "feasible_L_run_summary.csv")
    plot_path = os.path.join(out_dir, args.plot)

    ts_fields = [
        "N",
        "run_tag",
        "job_id",
        "node_tag",
        "host",
        "worker",
        "prefix",
        "event_idx",
        "stage",
        "elapsed_sec",
        "L",
        "best_so_far_run",
        "source_history_log",
    ]
    write_csv(ts_csv, all_rows, ts_fields)

    summary_rows = build_run_summary(all_rows)
    summary_fields = [
        "N",
        "run_tag",
        "job_id",
        "node_tag",
        "host",
        "worker",
        "prefix",
        "feasible_points",
        "first_feasible_t",
        "last_feasible_t",
        "best_feasible_L",
        "final_best_so_far_run",
    ]
    write_csv(summary_csv, summary_rows, summary_fields)

    maybe_plot(all_rows, plot_path)

    print(f"Wrote time series: {ts_csv}")
    print(f"Wrote run summary: {summary_csv}")
    print(f"Wrote plot (if matplotlib available): {plot_path}")


if __name__ == "__main__":
    main()
