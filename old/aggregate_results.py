#!/usr/bin/env python3
import glob
import os
import sys

def extract_L(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                if line.startswith("#") and " L=" in line:
                    try:
                        return float(line.split(" L=")[1].split()[0])
                    except Exception:
                        return None
    except Exception:
        return None
    return None

def find_best_packing(results_dir):
    pattern = os.path.join(results_dir, "*.csv")
    files = glob.glob(pattern)
    if not files:
        print(f"No CSV files found in {results_dir}")
        return 1

    best_L = float("inf")
    best_file = None

    for path in files:
        L = extract_L(path)
        if L is None:
            continue
        if L < best_L:
            best_L = L
            best_file = path

    if best_file is None:
        print("No valid L values found in headers.")
        return 1

    print("--- Global Best Found ---")
    print(f"Minimum L: {best_L}")
    print(f"Source: {best_file}")
    return 0

if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    sys.exit(find_best_packing(results_dir))
