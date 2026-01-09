#!/usr/bin/env python3
"""
triangulate_tree.py

- Builds the fixed 15-vertex Christmas tree polygon (scaled Decimals).
- Triangulates the simple polygon using ear clipping (triangles use only original vertices).
- Plots polygon + triangulation, with vertex indices.
- Prints C-ready triangle index triples.

Dependencies:
  pip install matplotlib
  (optional) pip install shapely

Run:
  python triangulate_tree.py

Outputs:
  - triangulation.png
  - prints TRIS indices to stdout
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import List, Tuple
import math

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Precision + scale (matches your contest style)
# ------------------------------------------------------------
getcontext().prec = 25
scale_factor = Decimal("1e18")


# ------------------------------------------------------------
# Geometry helpers (work in float on "unscaled" coordinates)
# We triangulate in float (safe here: 15 vertices; exact-ish).
# ------------------------------------------------------------
@dataclass(frozen=True)
class Pt:
    x: float
    y: float

def cross(a: Pt, b: Pt, c: Pt) -> float:
    """2D cross product (b-a) x (c-a). Positive => left turn."""
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

def polygon_area(poly: List[Pt]) -> float:
    """Signed area. >0 => CCW."""
    s = 0.0
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        s += poly[i].x * poly[j].y - poly[j].x * poly[i].y
    return 0.5 * s

def is_ccw(poly: List[Pt]) -> bool:
    return polygon_area(poly) > 0.0

def point_in_triangle(p: Pt, a: Pt, b: Pt, c: Pt, eps: float = 1e-15) -> bool:
    """
    Barycentric sign test; treats boundary as inside.
    Works for CCW or CW as long as consistent.
    """
    c1 = cross(a, b, p)
    c2 = cross(b, c, p)
    c3 = cross(c, a, p)
    has_neg = (c1 < -eps) or (c2 < -eps) or (c3 < -eps)
    has_pos = (c1 > eps) or (c2 > eps) or (c3 > eps)
    return not (has_neg and has_pos)

def is_convex(prev: Pt, cur: Pt, nxt: Pt, ccw: bool, eps: float = 1e-15) -> bool:
    z = cross(prev, cur, nxt)
    return z > eps if ccw else z < -eps

def earclip_triangulate(vertices: List[Pt]) -> List[Tuple[int, int, int]]:
    """
    Ear clipping triangulation for a simple polygon.
    Returns list of triangles as index triples into original vertex list.
    Triangles cover the polygon (non-overlapping).
    """
    n = len(vertices)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    ccw = is_ccw(vertices)

    # Work on a list of active indices
    idx = list(range(n))
    tris: List[Tuple[int, int, int]] = []

    # Safety to prevent infinite loops if polygon is not simple
    max_iters = 10_000
    iters = 0

    while len(idx) > 3:
        iters += 1
        if iters > max_iters:
            raise RuntimeError("Ear clipping failed (polygon may be non-simple or nearly degenerate).")

        ear_found = False
        m = len(idx)

        for k in range(m):
            i_prev = idx[(k - 1) % m]
            i_cur  = idx[k]
            i_next = idx[(k + 1) % m]

            a, b, c = vertices[i_prev], vertices[i_cur], vertices[i_next]

            # Must be a convex corner (an ear candidate)
            if not is_convex(a, b, c, ccw):
                continue

            # No other active vertex may lie inside the ear triangle
            ok = True
            for j in idx:
                if j in (i_prev, i_cur, i_next):
                    continue
                if point_in_triangle(vertices[j], a, b, c):
                    ok = False
                    break
            if not ok:
                continue

            # Found an ear
            tris.append((i_prev, i_cur, i_next))
            del idx[k]
            ear_found = True
            break

        if not ear_found:
            # This can happen if polygon is not strictly simple or has precision issues
            raise RuntimeError("No ear found. Polygon may be non-simple or numerically degenerate.")

    # Remaining triangle
    tris.append((idx[0], idx[1], idx[2]))
    return tris


# ------------------------------------------------------------
# Build the tree vertices (scaled Decimals) then convert to float
# ------------------------------------------------------------
def build_tree_vertices_scaled() -> List[Tuple[Decimal, Decimal]]:
    trunk_w = Decimal("0.15")
    trunk_h = Decimal("0.2")
    base_w  = Decimal("0.7")
    mid_w   = Decimal("0.4")
    top_w   = Decimal("0.25")
    tip_y   = Decimal("0.8")
    tier_1_y= Decimal("0.5")
    tier_2_y= Decimal("0.25")
    base_y  = Decimal("0.0")
    trunk_bottom_y = -trunk_h

    sf = scale_factor
    verts = [
        (Decimal("0.0") * sf, tip_y * sf),
        (top_w / Decimal("2") * sf, tier_1_y * sf),
        (top_w / Decimal("4") * sf, tier_1_y * sf),
        (mid_w / Decimal("2") * sf, tier_2_y * sf),
        (mid_w / Decimal("4") * sf, tier_2_y * sf),
        (base_w / Decimal("2") * sf, base_y * sf),
        (trunk_w / Decimal("2") * sf, base_y * sf),
        (trunk_w / Decimal("2") * sf, trunk_bottom_y * sf),
        (-(trunk_w / Decimal("2")) * sf, trunk_bottom_y * sf),
        (-(trunk_w / Decimal("2")) * sf, base_y * sf),
        (-(base_w / Decimal("2")) * sf, base_y * sf),
        (-(mid_w / Decimal("4")) * sf, tier_2_y * sf),
        (-(mid_w / Decimal("2")) * sf, tier_2_y * sf),
        (-(top_w / Decimal("4")) * sf, tier_1_y * sf),
        (-(top_w / Decimal("2")) * sf, tier_1_y * sf),
    ]
    return verts

def to_unscaled_float(verts_scaled: List[Tuple[Decimal, Decimal]]) -> List[Pt]:
    inv = Decimal(1) / scale_factor
    out: List[Pt] = []
    for (xS, yS) in verts_scaled:
        x = float(xS * inv)
        y = float(yS * inv)
        out.append(Pt(x, y))
    return out


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def plot_triangulation(verts: List[Pt], tris: List[Tuple[int,int,int]], out_png: str = "triangulation.png"):
    xs = [p.x for p in verts] + [verts[0].x]
    ys = [p.y for p in verts] + [verts[0].y]

    fig, ax = plt.subplots(figsize=(7, 8))
    ax.plot(xs, ys, linewidth=2)

    # draw vertices + labels
    for i, p in enumerate(verts):
        ax.scatter([p.x], [p.y], s=30)
        ax.text(p.x, p.y, f" {i}", fontsize=10, va="bottom")

    # draw triangle edges
    for t_idx, (a,b,c) in enumerate(tris):
        tri = [verts[a], verts[b], verts[c], verts[a]]
        ax.plot([q.x for q in tri], [q.y for q in tri], linewidth=1)

        # optional: label triangle near centroid
        cx = (verts[a].x + verts[b].x + verts[c].x) / 3.0
        cy = (verts[a].y + verts[b].y + verts[c].y) / 3.0
        ax.text(cx, cy, str(t_idx), fontsize=8, ha="center", va="center", alpha=0.8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"ChristmasTree triangulation: V={len(verts)}  T={len(tris)}")
    ax.grid(True, linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"[ok] wrote {out_png}")


# ------------------------------------------------------------
# Optional sanity check with shapely (area match)
# ------------------------------------------------------------
def shapely_sanity_check(verts: List[Pt], tris: List[Tuple[int,int,int]]):
    try:
        from shapely.geometry import Polygon
    except Exception:
        print("[info] shapely not installed; skipping sanity check.")
        return

    poly = Polygon([(p.x, p.y) for p in verts])
    if not poly.is_valid:
        print("[warn] shapely says polygon is invalid (self-intersection?). Triangulation may be meaningless.")
        return

    tri_area = 0.0
    for (a,b,c) in tris:
        tri = Polygon([(verts[a].x, verts[a].y), (verts[b].x, verts[b].y), (verts[c].x, verts[c].y)])
        tri_area += tri.area

    print(f"[check] polygon area = {poly.area:.18g}")
    print(f"[check] sum triangle areas = {tri_area:.18g}")
    print(f"[check] abs diff = {abs(poly.area - tri_area):.3e}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    verts_scaled = build_tree_vertices_scaled()
    verts = to_unscaled_float(verts_scaled)

    # Ensure ear clipping expects a simple polygon. If winding is CW, it still works (we handle orientation),
    # but you can reverse to make CCW if you want consistent output.
    tris = earclip_triangulate(verts)

    print("\n=== Triangles (vertex indices) ===")
    for t in tris:
        print(t)

    print("\n=== C-ready TRIS[] ===")
    print("typedef struct { int a,b,c; } Tri;")
    print("static const Tri TRIS[] = {")
    for (a,b,c) in tris:
        print(f"    {{{a}, {b}, {c}}},")
    print("};")
    print(f"static const int NTRI = {len(tris)};")

    plot_triangulation(verts, tris, out_png="triangulation.png")
    shapely_sanity_check(verts, tris)


if __name__ == "__main__":
    main()
