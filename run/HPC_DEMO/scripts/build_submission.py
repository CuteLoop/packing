#!/usr/bin/env python3
"""
build_submission.py

Build a Kaggle-style submission.csv from *_best_polys_N*.csv files.

Default usage (from run/HPC_DEMO):
  python3 scripts/build_submission.py --input-dir csv --output submission.csv
"""

import argparse
import csv
import glob
import math
import os
import re
from typing import Dict, List, Optional, Tuple

HEADER_RE = re.compile(
    r"#\s*prefix=(?P<prefix>\S+)\s+run_id=(?P<run_id>\d+)\s+seed=(?P<seed>\d+)\s+"
    r"L=(?P<L>[-+eE0-9\.]+)\s+best_feas=(?P<best_feas>[-+eE0-9\.]+)\s+N=(?P<N>\d+)"
)
N_IN_NAME_RE = re.compile(r"_N(?P<N>\d+)")


def parse_header(lines: List[str]) -> Dict[str, Optional[str]]:
    for line in lines:
        if not line.startswith("#"):
            break
        m = HEADER_RE.search(line.strip())
        if m:
            return m.groupdict()
    return {}


def parse_n_from_filename(path: str) -> Optional[int]:
    m = N_IN_NAME_RE.search(os.path.basename(path))
    if not m:
        return None
    try:
        return int(m.group("N"))
    except ValueError:
        return None


def read_positions(path: str) -> Tuple[List[Dict[str, str]], Dict[str, Optional[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    meta = parse_header(lines)

    data_lines = [ln for ln in lines if not ln.startswith("#") and ln.strip()]
    reader = csv.DictReader(data_lines)
    required = {"i", "cx", "cy", "theta_rad"}
    if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
        missing = sorted(required - set(reader.fieldnames or []))
        raise ValueError(f"{path}: missing columns {missing}")

    rows = [row for row in reader]
    return rows, meta


def choose_best_files(paths: List[str]) -> Dict[int, str]:
    """Choose one file per N, preferring lowest best_feas then lowest L."""
    best: Dict[int, Tuple[float, float, str]] = {}

    for path in paths:
        n = parse_n_from_filename(path)
        if n is None:
            continue

        with open(path, "r", encoding="utf-8") as f:
            header_lines = []
            for _ in range(3):
                line = f.readline()
                if not line:
                    break
                header_lines.append(line)
                if not line.startswith("#"):
                    break
        meta = parse_header(header_lines)

        best_feas = float(meta.get("best_feas")) if meta.get("best_feas") else float("inf")
        L = float(meta.get("L")) if meta.get("L") else float("inf")

        score = (best_feas, L)
        if n not in best or score < (best[n][0], best[n][1]):
            best[n] = (score[0], score[1], path)

    return {n: info[2] for n, info in best.items()}


def collect_all_files(input_dir: str) -> List[str]:
    best = glob.glob(os.path.join(input_dir, "*_best_polys_N*.csv"))
    checkpoints = glob.glob(os.path.join(input_dir, "*_checkpoint_N*.csv"))
    return sorted(set(best + checkpoints))


def format_val(val: float) -> str:
    return f"s{val:.10g}"


def build_submission(input_dir: str, output_path: str) -> None:
    paths = collect_all_files(input_dir)
    if not paths:
        raise SystemExit(f"No best_polys or checkpoint files found in: {input_dir}")

    best_files = choose_best_files(paths)

    def materialize_rows(n_target: int, path: str, trim_to: Optional[int]) -> List[Dict[str, str]]:
        positions, _meta = read_positions(path)
        if trim_to is not None:
            positions = positions[:trim_to]
        out = []
        for row in positions:
            idx = int(row["i"])
            cx = float(row["cx"])
            cy = float(row["cy"])
            theta = float(row["theta_rad"])
            deg = math.degrees(theta)
            tree_id = f"{str(n_target).zfill(3)}_{idx}"
            out.append({
                "id": tree_id,
                "x": format_val(cx),
                "y": format_val(cy),
                "deg": format_val(deg),
            })
        return out

    rows_out = []
    for n in range(1, 201):
        if n in best_files:
            rows_out.extend(materialize_rows(n, best_files[n], trim_to=None))
            continue

        # Fallback: use next available N and drop one polygon to match target
        higher = [k for k in sorted(best_files.keys()) if k > n]
        if not higher:
            raise SystemExit(f"No fallback available for missing N={n}")
        src_n = higher[0]
        rows_out.extend(materialize_rows(n, best_files[src_n], trim_to=n))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "x", "y", "deg"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"Wrote {output_path} with {len(rows_out)} rows from {len(best_files)} N values.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="csv", help="Directory with *_best_polys_N*.csv")
    ap.add_argument("--output", default="submission.csv", help="Output submission CSV path")
    args = ap.parse_args()
    build_submission(args.input_dir, args.output)


if __name__ == "__main__":
    main()
