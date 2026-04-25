#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Optional


def parse_best_header(path: str) -> Dict[str, Optional[float]]:
    out = {"L": None, "best_feas": None, "N": None}
    try:
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
        # Example:
        # # prefix=... run_id=... seed=... L=1.234 best_feas=0 N=20
        mL = re.search(r"\bL=([0-9eE+\-.]+)", line)
        mF = re.search(r"\bbest_feas=([0-9eE+\-.]+)", line)
        mN = re.search(r"\bN=([0-9]+)", line)
        if mL:
            out["L"] = float(mL.group(1))
        if mF:
            out["best_feas"] = float(mF.group(1))
        if mN:
            out["N"] = int(mN.group(1))
    except Exception:
        pass
    return out


def parse_prefix(prefix: str) -> Dict[str, Optional[str]]:
    # Expected pattern from launcher:
    # N{N}_job{jobid}_node_{xx}_{host}_w{worker}
    out: Dict[str, Optional[str]] = {
        "N": None,
        "job_id": None,
        "node_tag": None,
        "host": None,
        "worker": None,
    }
    m = re.search(r"^N(\d+)_job(\d+)_(node_\d+)_([^_]+)_w(\d+)$", prefix)
    if m:
        out["N"] = m.group(1)
        out["job_id"] = m.group(2)
        out["node_tag"] = m.group(3)
        out["host"] = m.group(4)
        out["worker"] = m.group(5)
    return out


def summarize_history(path: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "good_hits": 0,
        "first_good_t": None,
        "last_good_t": None,
        "best_feasible_L": None,
    }
    try:
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                feasible = row.get("feasible", "")
                if str(feasible) == "1":
                    t = float(row["elapsed_sec"]) if row.get("elapsed_sec") else None
                    L = float(row["L"]) if row.get("L") else None
                    out["good_hits"] = int(out["good_hits"] or 0) + 1
                    if t is not None:
                        if out["first_good_t"] is None or t < out["first_good_t"]:
                            out["first_good_t"] = t
                        if out["last_good_t"] is None or t > out["last_good_t"]:
                            out["last_good_t"] = t
                    if L is not None:
                        if out["best_feasible_L"] is None or L < out["best_feasible_L"]:
                            out["best_feasible_L"] = L
    except Exception:
        pass
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize old-run outputs under out/node_*/")
    parser.add_argument("--root", default="out", help="Root output directory (default: out)")
    parser.add_argument("--output", default="summary_old_runs.csv", help="Output CSV path")
    args = parser.parse_args()

    hist_paths = glob.glob(os.path.join(args.root, "node_*", "logs", "*_history_log.csv"))
    rows: List[Dict[str, object]] = []

    for hist in sorted(hist_paths):
        base = os.path.basename(hist)
        prefix = base.replace("_history_log.csv", "")
        pref = parse_prefix(prefix)
        node_dir = os.path.basename(os.path.dirname(os.path.dirname(hist)))

        hist_sum = summarize_history(hist)

        N_guess = pref.get("N")
        best_path = None
        best_hdr = {"L": None, "best_feas": None, "N": None}
        if N_guess is not None:
            best_glob = os.path.join(
                os.path.dirname(os.path.dirname(hist)),
                "csv",
                f"{prefix}_best_polys_N{int(N_guess):03d}.csv",
            )
            matches = glob.glob(best_glob)
            if matches:
                best_path = matches[0]
                best_hdr = parse_best_header(best_path)

        rows.append(
            {
                "N": pref.get("N") or best_hdr.get("N"),
                "job_id": pref.get("job_id"),
                "node_tag": pref.get("node_tag") or node_dir,
                "host": pref.get("host"),
                "worker": pref.get("worker"),
                "prefix": prefix,
                "history_log": hist,
                "best_csv": best_path,
                "canonical_best_L": best_hdr.get("L"),
                "canonical_best_feas": best_hdr.get("best_feas"),
                "best_feasible_L_from_history": hist_sum.get("best_feasible_L"),
                "good_hits": hist_sum.get("good_hits"),
                "first_good_t": hist_sum.get("first_good_t"),
                "last_good_t": hist_sum.get("last_good_t"),
            }
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fields = [
        "N",
        "job_id",
        "node_tag",
        "host",
        "worker",
        "prefix",
        "history_log",
        "best_csv",
        "canonical_best_L",
        "canonical_best_feas",
        "best_feasible_L_from_history",
        "good_hits",
        "first_good_t",
        "last_good_t",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
