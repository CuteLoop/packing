#!/usr/bin/env python3
"""Validate study CSV files against the locked schema."""
import sys
import csv
import os

BISECTION_COLS = [
    'run_id','seed','method','N','R','probe_idx',
    'wall_sec_start','wall_sec_end','L_lo','L_hi','L_mid',
    'slice_budget_sec','slice_used_sec','feasible',
    'min_energy','min_feas','resample_events','L_best','bracket_width'
]

LOG_COLS = [
    'run_id','seed','method','N','R','wall_sec','probe_idx',
    'L_current','best_energy','best_feas','feasible_ever','L_best','event'
]


def check(path, required_cols, label):
    errors = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return [f"{label}: empty file or no header"]
        for col in required_cols:
            if col not in reader.fieldnames:
                errors.append(f"{label}: missing column '{col}'")
        rows = list(reader)
        if len(rows) == 0:
            errors.append(f"{label}: no data rows")
        if 'probe_idx' in required_cols and label == 'bisection':
            if len(rows) < 3:
                errors.append(f"{label}: fewer than 3 probes ({len(rows)} rows)")
            for i, row in enumerate(rows):
                try:
                    idx = int(row.get('probe_idx', -1))
                    if idx != i:
                        errors.append(f"{label} row {i}: probe_idx={idx} expected {i}")
                except ValueError:
                    errors.append(f"{label} row {i}: non-integer probe_idx")
                if row.get('feasible', '') not in ('0', '1'):
                    errors.append(f"{label} row {i}: feasible not 0/1")
        if label == 'log':
            prev_t = -1.0
            for i, row in enumerate(rows):
                try:
                    t = float(row.get('wall_sec', -1))
                    if t < prev_t - 0.01:
                        errors.append(f"{label} row {i}: wall_sec not monotonic")
                    prev_t = t
                except ValueError:
                    errors.append(f"{label} row {i}: non-numeric wall_sec")
    return errors


def main():
    if len(sys.argv) < 2:
        print("Usage: validate_schema.py <file_or_dir> [..]")
        sys.exit(1)
    all_errors = {}
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            files = [os.path.join(arg, f) for f in os.listdir(arg) if f.endswith('.csv')]
        else:
            files = [arg]
        for path in files:
            name = os.path.basename(path)
            if '_bisection' in name:
                errs = check(path, BISECTION_COLS, 'bisection')
            elif '_log' in name:
                errs = check(path, LOG_COLS, 'log')
            else:
                continue
            if errs:
                all_errors[path] = errs
    if all_errors:
        for p, errs in all_errors.items():
            print(f"\n=== ERRORS: {p} ===")
            for e in errs:
                print(f"  {e}")
        sys.exit(1)
    else:
        print("All CSV files valid.")


if __name__ == '__main__':
    main()
