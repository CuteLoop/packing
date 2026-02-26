#!/usr/bin/env python3
import argparse
import csv
import math
import os
import sys
from typing import List, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
except Exception as exc:
    print("matplotlib is required: pip install matplotlib", file=sys.stderr)
    raise

BASE_V = [
    ( 0.0,     0.8  ),
    ( 0.125,   0.5  ),
    ( 0.0625,  0.5  ),
    ( 0.2,     0.25 ),
    ( 0.1,     0.25 ),
    ( 0.35,    0.0  ),
    ( 0.075,   0.0  ),
    ( 0.075,  -0.2  ),
    (-0.075,  -0.2  ),
    (-0.075,   0.0  ),
    (-0.35,    0.0  ),
    (-0.1,     0.25 ),
    (-0.2,     0.25 ),
    (-0.0625,  0.5  ),
    (-0.125,   0.5  ),
]


def parse_header_L(path: str) -> float:
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
                        return 0.0
    except Exception:
        return 0.0
    return 0.0


def read_csv(path: str) -> List[Tuple[int, float, float, float]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            if line.strip().lower().startswith("i,cx,cy,theta_rad"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                i = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                th = float(parts[3])
                rows.append((i, cx, cy, th))
            except Exception:
                continue
    return rows


def transform_poly(cx: float, cy: float, th: float) -> List[Tuple[float, float]]:
    c = math.cos(th)
    s = math.sin(th)
    out = []
    for x, y in BASE_V:
        out.append((c * x - s * y + cx, s * x + c * y + cy))
    return out


def plot_packing(csv_path: str, out_path: str, size: int, margin: float, show: bool) -> None:
    L = parse_header_L(csv_path)
    rows = read_csv(csv_path)
    if not rows:
        print("No polygon rows found in CSV.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(size / 100.0, size / 100.0), dpi=100)

    # Draw square boundary if L found
    if L > 0.0:
        half = 0.5 * L
        ax.add_patch(Rectangle((-half, -half), L, L, fill=False, linewidth=2))
    else:
        half = None

    for _, cx, cy, th in rows:
        poly = transform_poly(cx, cy, th)
        ax.add_patch(Polygon(poly, closed=True, facecolor="#888888", edgecolor="#000000", linewidth=0.5, alpha=0.25))

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    if half is not None:
        ax.set_xlim(-half - margin, half + margin)
        ax.set_ylim(-half - margin, half + margin)
    else:
        xs = [p[0] for _, cx, cy, th in rows for p in transform_poly(cx, cy, th)]
        ys = [p[1] for _, cx, cy, th in rows for p in transform_poly(cx, cy, th)]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        ax.set_xlim(xmin - margin, xmax + margin)
        ax.set_ylim(ymin - margin, ymax + margin)

    fig.tight_layout(pad=0)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)

    if show:
        plt.show()
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot packing CSV to an image.")
    parser.add_argument("csv", help="Path to csv/<prefix>_best_polys_N###.csv")
    parser.add_argument("out", help="Output image path (.png or .svg)")
    parser.add_argument("--size", type=int, default=1100, help="Image size in pixels (default: 1100)")
    parser.add_argument("--margin", type=float, default=0.02, help="Margin around square (in world units)")
    parser.add_argument("--show", action="store_true", help="Show interactive window")
    args = parser.parse_args()

    plot_packing(args.csv, args.out, args.size, args.margin, args.show)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
