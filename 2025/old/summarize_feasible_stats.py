#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Dict, Iterable, List

from feasible_analysis_common import (
    POLY_AREA,
    best_so_far_feasible_rows,
    density_from_L,
    parse_n_from_name,
    parse_n_values,
    read_history_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize feasible-run statistics from history logs")
    parser.add_argument("--logs-dir", default="logs", help="Directory containing *_history_log.csv files")
    parser.add_argument("--n-values", default="5,10,25,50,100,200", help="Comma-separated N values to include")
    parser.add_argument("--output-dir", default=None, help="Directory for generated CSV outputs")
    return parser.parse_args()


def default_output_dir(logs_dir: Path) -> Path:
    return logs_dir.parent / "analysis"


def median_or_blank(values: Iterable[float]) -> str:
    data = list(values)
    return "" if not data else f"{statistics.median(data):.9f}"


def build_run_summary(n: int, path: Path) -> Dict[str, object] | None:
    rows = read_history_rows(path)
    feasible = [row for row in rows if int(row["feasible"]) == 1]
    if not feasible:
        return None

    best_updates = best_so_far_feasible_rows(rows)
    first_feasible = feasible[0]
    best_row = min(feasible, key=lambda row: float(row["L"]))
    best_L = float(best_row["L"])
    first_L = float(first_feasible["L"])
    best_update_times = [float(row["elapsed_sec"]) for row in best_updates]
    plateau_gaps = [b - a for a, b in zip(best_update_times, best_update_times[1:])]
    run_last_time = float(rows[-1]["elapsed_sec"]) if rows else float(best_row["elapsed_sec"])
    if best_update_times:
        plateau_gaps.append(run_last_time - best_update_times[-1])

    return {
        "N": n,
        "prefix": path.stem,
        "history_log": str(path),
        "total_events": len(rows),
        "feasible_points": len(feasible),
        "first_event_t": f"{float(rows[0]['elapsed_sec']):.6f}",
        "last_event_t": f"{run_last_time:.6f}",
        "first_feasible_t": f"{float(first_feasible['elapsed_sec']):.6f}",
        "first_feasible_L": f"{first_L:.9f}",
        "first_feasible_density": f"{density_from_L(n, first_L):.9f}",
        "best_feasible_t": f"{float(best_row['elapsed_sec']):.6f}",
        "best_feasible_L": f"{best_L:.9f}",
        "best_feasible_density": f"{density_from_L(n, best_L):.9f}",
        "best_stage": best_row["stage"],
        "time_to_best_from_first_feasible": f"{float(best_row['elapsed_sec']) - float(first_feasible['elapsed_sec']):.6f}",
        "abs_improvement_after_first_feasible": f"{first_L - best_L:.9f}",
        "relative_polish_gain": f"{(first_L - best_L) / first_L:.9f}",
        "best_updates": len(best_updates),
        "longest_best_plateau_sec": f"{max(plateau_gaps) if plateau_gaps else 0.0:.6f}",
    }


def build_n_summary(run_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    by_n: Dict[int, List[Dict[str, object]]] = {}
    for row in run_rows:
        by_n.setdefault(int(row["N"]), []).append(row)

    summary: List[Dict[str, object]] = []
    for n in sorted(by_n):
        rows = by_n[n]
        best_L_values = [float(row["best_feasible_L"]) for row in rows]
        best_density_values = [float(row["best_feasible_density"]) for row in rows]
        t_first_values = [float(row["first_feasible_t"]) for row in rows]
        t_best_values = [float(row["best_feasible_t"]) for row in rows]
        polish_values = [float(row["relative_polish_gain"]) for row in rows]
        update_values = [float(row["best_updates"]) for row in rows]
        plateau_values = [float(row["longest_best_plateau_sec"]) for row in rows]
        summary.append(
            {
                "N": n,
                "polygon_area": f"{POLY_AREA:.9f}",
                "runs_with_feasible_points": len(rows),
                "median_first_feasible_t": median_or_blank(t_first_values),
                "median_time_to_best": median_or_blank(t_best_values),
                "best_of_best_L": f"{min(best_L_values):.9f}",
                "median_best_L": median_or_blank(best_L_values),
                "best_of_best_density": f"{max(best_density_values):.9f}",
                "median_best_density": median_or_blank(best_density_values),
                "median_relative_polish_gain": median_or_blank(polish_values),
                "median_best_updates": median_or_blank(update_values),
                "median_longest_plateau_sec": median_or_blank(plateau_values),
            }
        )
    return summary


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    n_values = parse_n_values(args.n_values)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(logs_dir)

    run_rows: List[Dict[str, object]] = []
    for path in sorted(logs_dir.glob("*_history_log.csv")):
        n = parse_n_from_name(path)
        if n not in n_values:
            continue
        row = build_run_summary(n, path)
        if row is not None:
            run_rows.append(row)

    if not run_rows:
        raise SystemExit("No feasible runs found for selected N values")

    run_rows.sort(key=lambda row: (int(row["N"]), str(row["prefix"])))
    n_rows = build_n_summary(run_rows)

    run_path = output_dir / "feasible_run_statistics.csv"
    n_path = output_dir / "feasible_n_statistics.csv"
    write_csv(run_path, run_rows)
    write_csv(n_path, n_rows)

    print(f"Wrote run statistics: {run_path}")
    print(f"Wrote N statistics: {n_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
