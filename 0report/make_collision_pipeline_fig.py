#!/usr/bin/env python3
"""
make_collision_pipeline_fig_improved.py
Four-panel illustration of the collision detection pipeline.
This version implements a professional, academic style with:
  - Tight layout and minimal white space.
  - Consistent color scheme: trees (green), overlaps (red), AABBs (black).
  - No text overlapping objects.
  - Professionally designed triangle decomposition palette.
  - Comprehensive object placement within figure bounds.

Original file description:
(a) Bounding circle  (b) Triangle decomposition (T=13, 3 clusters)
(c) AABB pruning     (d) SAT — penetration depth on min-overlap axis

Outputs:  img/collision_pipeline.pdf   img/collision_pipeline.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Publication style ────────────────────────────────────────────────────────
# Use professional academic standard settings
plt.rcParams.update({
    "font.family":       "Dejavu Serif",
    "mathtext.fontset":  "dejavuserif",
    "font.size":           9.0, # increased slightly for readability
    "axes.titlesize":     10.0,
    "axes.titleweight":   "bold",
    "axes.labelsize":      8.5,
    "xtick.labelsize":     7.5,
    "ytick.labelsize":     7.5,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#444444",
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "lines.linewidth":    1.3,
    "legend.fontsize":     8.0,
    "legend.framealpha":  0.92,
    "legend.edgecolor":  "#cccccc",
})

# ── Specific Color Schema for Professional Style ─────────────────────────────
# Theme-driven colors per user request:
# Green for trees, red for overlaps, black for bounding boxes.

# Tree colors (soft green, distinct edge)
C_TREE_GREEN_FILL = "#d9f0df"
C_TREE_GREEN_EDGE = "#2a7a4a"

# Overlap colors (soft red, bold red for conflict)
C_OVERLAP_RED_FILL = "#f7dcdb"
C_OVERLAP_RED_EDGE = "#c13832"

# Bounding Box color
C_AABB_BLACK = "#333333"

# Common annotation colors
C_GREY_TEXT = "#555555"
C_ANNOTATE_ORANGE = "#d97706" # from original, still professional
C_GREY_PRUNED = "#7a7a7a"

# Cluster palette: three harmonious, distinct, print-safe colors for decomp
# Palette is professional and distinct
CLUSTER_COL = ["#3e9ad1", "#de7e66", "#6bb589"]

# ── Polygon geometry  (verbatim from src/base_geometry.c) ────────────────────
# Defines the specific Kaggle competition non-convex shape
BASE_V = np.array([
    [ 0.0,      0.8  ], [ 0.125,   0.5  ], [ 0.0625,  0.5  ],
    [ 0.2,      0.25 ], [ 0.1,     0.25 ], [ 0.35,    0.0  ],
    [ 0.075,    0.0  ], [ 0.075,  -0.2  ], [-0.075,  -0.2  ],
    [-0.075,    0.0  ], [-0.35,    0.0  ], [-0.1,     0.25 ],
    [-0.2,      0.25 ], [-0.0625,  0.5  ], [-0.125,   0.5  ],
])

# Vertex indices for the offline decomposition into T=13 triangles
TRIS = [
    (0,1,2),  (2,3,4),  (0,2,4),  (4,5,6),  (0,4,6),   # cluster 0 (5)
    (0,6,7),  (0,7,8),  (0,8,9),  (9,10,11),(0,9,11),   # cluster 1 (5)
    (11,12,13),(0,11,13),(0,13,14),                      # cluster 2 (3)
]
CLUSTER_SIZES = [5, 5, 3]

# ── Geometry helpers ─────────────────────────────────────────────────────────
def closed(v):
    """Return a stacked array to draw a closed polyline."""
    return np.vstack([v, v[0]])

def transform(v, cx, cy, theta):
    """Rigid body transform: scale=1, rotation by theta, translation to (cx,cy)."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (v @ R.T) + np.array([cx, cy])

def poly_aabb(v):
    """Min-max bounds for the given vertices."""
    return v[:,0].min(), v[:,1].min(), v[:,0].max(), v[:,1].max()

def aabb_patch(v, **kw):
    """Matplotlib patch for the axis-aligned bounding box."""
    x0, y0, x1, y1 = poly_aabb(v)
    return mpatches.Rectangle((x0, y0), x1-x0, y1-y0, **kw)

def edge_normals(tri):
    """Compute normal vectors for each edge of a triangle."""
    ns = []
    for i in range(3):
        e = tri[(i+1) % 3] - tri[i]
        n = np.array([e[1], -e[0]]) # perpendicular vector
        ns.append(n / np.linalg.norm(n)) # unit normal
    return ns

def project(tri, axis):
    """Project triangle vertices onto a 1-D axis."""
    d = tri @ axis
    return d.min(), d.max()

def sat_depth(tA, tB):
    """
    Separating Axis Theorem core: projects two triangles on 6 edge-normals.
    Returns (depth, min_axis) if overlapping, else (None, axis_with_gap).
    """
    axes = edge_normals(tA) + edge_normals(tB)
    min_d, min_ax = np.inf, None
    for ax in axes:
        lo_A, hi_A = project(tA, ax)
        lo_B, hi_B = project(tB, ax)
        overlap = min(hi_A, hi_B) - max(lo_A, lo_B)
        if overlap < 0:
            return None, ax          # separating axis found
        if overlap < min_d:
            min_d, min_ax = overlap, ax
    return min_d, min_ax

def polygon_centroid(v):
    """Area centroid of a simple polygon with vertices in boundary order."""
    x = v[:, 0]
    y = v[:, 1]
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    A2 = cross.sum()  # 2 * signed area
    if abs(A2) < 1e-15:
        # Fallback to mean for degenerate cases
        return np.array([x.mean(), y.mean()])
    cx = ((x + x1) * cross).sum() / (3.0 * A2)
    cy = ((y + y1) * cross).sum() / (3.0 * A2)
    return np.array([cx, cy])

def hide_axes(ax):
    """Clean layout with minimal axes overhead."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

# ── Figure layout ────────────────────────────────────────────────────────────
# Tighter layout for publication
fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0))
axes = axes.ravel()
fig.subplots_adjust(left=0.03, right=0.99, top=0.95,
                    bottom=0.04, wspace=0.06, hspace=0.12)

# ════════════════════════════════════════════════════════════════════════════
# (a) Bounding Circle
# ════════════════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_title("(1) Bounding radius", pad=5)

pc = closed(BASE_V)
ax.fill(pc[:,0], pc[:,1], color=C_TREE_GREEN_FILL, zorder=2)
ax.plot(pc[:,0], pc[:,1], color=C_TREE_GREEN_EDGE, lw=1.6, zorder=3)

# Original code uses maximum vertex distance to origin for simple rotation-invariant bound
r_br = float(np.sqrt((BASE_V**2).sum(axis=1)).max())   # = 0.8
circ = mpatches.Circle((0, 0), r_br, fill=False, lw=1.6,
                        linestyle="--", edgecolor=C_AABB_BLACK, zorder=4)
ax.add_patch(circ)

# radius arrow + label
ax.annotate("", xy=(0, r_br), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=C_OVERLAP_RED_EDGE, lw=1.6))
ax.text(0.03, r_br * 0.52, r"$r_{\mathrm{br}}$",
    color=C_OVERLAP_RED_EDGE, fontsize=9,
    ha="left", va="center")

# h = 2 r_br annotation (spatial hash cell size)
h = 2 * r_br
ax.annotate("", xy=(h/2, -0.85), xytext=(-h/2, -0.85),
            arrowprops=dict(arrowstyle="<->", color=C_ANNOTATE_ORANGE, lw=1.2))
ax.text(0.0, -0.90, r"$h = 2r_{\mathrm{br}}$",
    color=C_ANNOTATE_ORANGE, fontsize=9,
    ha="center", va="top")
ax.plot([-h/2, h/2], [-0.84, -0.84], lw=0, zorder=0)   # reserve space

ax.plot(0, 0, "o", color=C_OVERLAP_RED_EDGE, ms=4.5, zorder=5)
ax.text(0.06, -0.05, r"center $c_i$", color=C_GREY_TEXT,
    fontsize=8.5, ha="left", va="top")
ax.set_xlim(-1.10, 1.10)
ax.set_ylim(-1.05, 1.10)
ax.set_aspect("equal")
hide_axes(ax)

# ════════════════════════════════════════════════════════════════════════════
# (b) Triangle Decomposition
# ════════════════════════════════════════════════════════════════════════════
ax = axes[1]
ax.set_title("(2) Triangle groups", pad=5)

idx = 0
for ci, (n, col) in enumerate(zip(CLUSTER_SIZES, CLUSTER_COL)):
    for t in range(n):
        a, b, c = TRIS[idx + t]
        tri = BASE_V[[a, b, c]]
        patch = mpatches.Polygon(tri, closed=True,
                                  facecolor=col, alpha=0.6,
                                  edgecolor="white", lw=0.75,
                                  zorder=2)
        ax.add_patch(patch)
    idx += n

# Polygon outline on top
pc = closed(BASE_V)
ax.plot(pc[:,0], pc[:,1], color=C_TREE_GREEN_EDGE, lw=1.6, zorder=4)
ax.plot(BASE_V[:,0], BASE_V[:,1], "o", color="#333333", ms=2.5, zorder=5)
ax.text(0.03, 0.96, r"$T=13$ triangles, 3 clusters",
    transform=ax.transAxes, color=C_GREY_TEXT, fontsize=8.5,
    ha="left", va="top",
    bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.92))

ax.set_xlim(-0.62, 0.62)
ax.set_ylim(-0.45, 1.02)
ax.set_aspect("equal")
hide_axes(ax)

# ════════════════════════════════════════════════════════════════════════════
# (c) AABB Pruning
# ════════════════════════════════════════════════════════════════════════════
ax = axes[2]
ax.set_title("(3) AABB pruning", pad=5)

# Three instances of the tree shape to demonstrate pruning cascade.
# All trees are green. Query (standard green), Candidate (lighter green fill), Pruned (faint).
pA_V = BASE_V
pB_V = transform(BASE_V,  0.50,  0.10,  0.45)   # close: AABBs overlap
pC_V = transform(BASE_V, -0.98, -0.05, -0.3 )   # far: AABB gap -> pruned

# Recenter by the global center of mass of all three instances,
# then nudge slightly right so composition is visually balanced.
all_pts_c = np.vstack([pA_V, pB_V, pC_V])
cm_c = all_pts_c.mean(axis=0)
target_cm_c = np.array([0.12, 0.0])
delta_c = target_cm_c - cm_c
pA_V = pA_V + delta_c
pB_V = pB_V + delta_c
pC_V = pC_V + delta_c

# Instances list to process iteratively
polys  = [pA_V,    pB_V,    pC_V    ]
tree_fills = [C_TREE_GREEN_FILL, "#eefaf1", "white"]
tree_edges = [C_TREE_GREEN_EDGE, C_TREE_GREEN_EDGE, C_GREY_PRUNED]
edge_styles = ["-", "-", "--"]
aabb_styles = ["--", "--", ":"]

for poly, fill, edge, es, aa_es in zip(polys, tree_fills, tree_edges, edge_styles, aabb_styles):
    pc2 = closed(poly)
    ax.fill(pc2[:,0], pc2[:,1], color=fill, alpha=1.0 if edge != C_GREY_PRUNED else 0.3, zorder=2)
    ax.plot(pc2[:,0], pc2[:,1], color=edge, lw=1.3, linestyle=es, zorder=3)

    # Black dashed AABB
    rect = aabb_patch(poly, fill=False, edgecolor=C_AABB_BLACK, lw=1.1,
                      linestyle=aa_es, zorder=3)
    ax.add_patch(rect)

# highlight the A–B AABB intersection
xA = poly_aabb(pA_V);  xB = poly_aabb(pB_V)
ov_x0 = max(xA[0], xB[0]);  ov_x1 = min(xA[2], xB[2])
ov_y0 = max(xA[1], xB[1]);  ov_y1 = min(xA[3], xB[3])
if ov_x0 < ov_x1 and ov_y0 < ov_y1:
    # Red for overlaps
    rect_overlap = mpatches.FancyBboxPatch(
        (ov_x0, ov_y0), ov_x1-ov_x0, ov_y1-ov_y0,
        boxstyle="round,pad=0.01",
    facecolor="none", edgecolor=C_OVERLAP_RED_EDGE, lw=1.4, zorder=4)
    ax.add_patch(rect_overlap)

    ax.text(0.03, 0.96, "query + candidates",
        transform=ax.transAxes, color=C_GREY_TEXT, fontsize=8.5,
        ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.92))
    ax.text(0.97, 0.09, "red box = AABB overlap",
        transform=ax.transAxes, color=C_OVERLAP_RED_EDGE, fontsize=8.4,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.92))

all_pts_c = np.vstack([pA_V, pB_V, pC_V])
xmin_c, ymin_c = all_pts_c.min(axis=0)
xmax_c, ymax_c = all_pts_c.max(axis=0)
mx_c, my_c = 0.14, 0.14
ax.set_xlim(xmin_c - mx_c, xmax_c + mx_c)
ax.set_ylim(ymin_c - my_c, ymax_c + my_c)
ax.set_aspect("equal")
hide_axes(ax)

# ════════════════════════════════════════════════════════════════════════════
# (d) SAT — Penetration Depth
# ════════════════════════════════════════════════════════════════════════════
ax = axes[3]
ax.set_title("(4) SAT narrow phase", pad=5)

# Use same transformed Kaggle trees from (c) to isolate relevant narrow phase
# Full trees faint green for context
# Colliding triangles within tree a bit more bold
pA_sat_V = BASE_V
pB_sat_V = transform(BASE_V,  0.50,  0.10,  0.45)

# Recenter by the combined center of mass of both SAT polygons,
# then nudge right so the left polygon fits cleanly in frame.
all_pts_sat = np.vstack([pA_sat_V, pB_sat_V])
cm_sat = all_pts_sat.mean(axis=0)
target_cm_sat = np.array([0.16, 0.04])
delta_sat = target_cm_sat - cm_sat
pA_sat_V = pA_sat_V + delta_sat
pB_sat_V = pB_sat_V + delta_sat

# Faint full context polygons
for poly_V, fill, edge in [(pA_sat_V, C_TREE_GREEN_FILL, C_TREE_GREEN_EDGE),
                           (pB_sat_V, "#eefaf1", C_TREE_GREEN_EDGE)]:
    pc_full = closed(poly_V)
    # ax.fill(pc_full[:,0], pc_full[:,1], color=fill, alpha=0.10, zorder=1)
    ax.plot(pc_full[:,0], pc_full[:,1], color=edge, lw=0.9, alpha=0.35, zorder=1)

# Isolate a specific overlapping triangle pair for narrow phase demo.
best_depth = -1.0
best_pair = None
for ta, tri_idx_a in enumerate(TRIS):
    triA = pA_sat_V[np.array(tri_idx_a)]
    ax0, ay0, ax1, ay1 = poly_aabb(triA)
    for tb, tri_idx_b in enumerate(TRIS):
        triB = pB_sat_V[np.array(tri_idx_b)]
        bx0, by0, bx1, by1 = poly_aabb(triB)
        # Mirror the triangle-level AABB gate used before SAT.
        if (ax1 < bx0) or (bx1 < ax0) or (ay1 < by0) or (by1 < ay0):
            continue
        d_try, ax_try = sat_depth(triA, triB)
        if d_try is not None and d_try > best_depth:
            best_depth = d_try
            best_pair = (ta, tb, triA, triB, ax_try)

if best_pair is None:
    raise RuntimeError("No overlapping triangle pair found for SAT panel.")

_ta_id, _tb_id, tA, tB, min_ax = best_pair

# Render the compared triangles with bolder green outlines. No fill to simplify.
ax.plot(np.append(tA[:,0], tA[0,0]), np.append(tA[:,1], tA[0,1]),
    color=C_TREE_GREEN_EDGE, lw=1.8, zorder=3)

ax.plot(np.append(tB[:,0], tB[0,0]), np.append(tB[:,1], tB[0,1]),
    color=C_TREE_GREEN_EDGE, lw=1.8, linestyle="--", zorder=3)

# Faint projection arrows in grey
all_axes  = edge_normals(tA) + edge_normals(tB)
centroid  = np.vstack([tA, tB]).mean(axis=0)
L_arr = 0.38
for i, n in enumerate(all_axes):
    perp   = np.array([-n[1], n[0]])
    offset = perp * (i - 2.5) * 0.045
    s = centroid - n * L_arr + offset
    e = centroid + n * L_arr + offset
    ax.annotate("", xy=e, xytext=s,
                arrowprops=dict(arrowstyle="->", color=C_GREY_TEXT,
                                lw=0.7, alpha=0.35))

# ── 1-D projection bar chart for the minimum-overlap axis at bottom ──────────
BAR_Y   = -0.42    # vertical center of the "number line"
BAR_H   =  0.038   # half-height of each bar

# Mapping function from world-space dot products to 1-D axis position [0, 1]
# Based on overall projections to keep bars centered
all_pts = np.vstack([tA, tB])
pmin_global = (all_pts @ min_ax).min() - 0.05
pmax_global = (all_pts @ min_ax).max() + 0.05
def pmap(v, lo=pmin_global, hi=pmax_global, x0=0.04, x1=0.92):
    return x0 + (v - lo) / (hi - lo) * (x1 - x0)

lo_A, hi_A = project(tA, min_ax)
lo_B, hi_B = project(tB, min_ax)

# number line arrow
ax.annotate("", xy=(0.95, BAR_Y), xytext=(0.01, BAR_Y),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9))

# bar A (solid tree-green fill)
ax.fill_betweenx([BAR_Y+BAR_H*0.4, BAR_Y+BAR_H*1.6],
                 pmap(lo_A), pmap(hi_A),
                 color=C_TREE_GREEN_FILL, edgecolor=C_TREE_GREEN_EDGE, lw=1.1, zorder=4)

# bar B (dashed outline tree-green)
ax.fill_betweenx([BAR_Y-BAR_H*1.6, BAR_Y-BAR_H*0.4],
                 pmap(lo_B), pmap(hi_B),
                 color="#f4fbf6", edgecolor=C_TREE_GREEN_EDGE, lw=1.1, linestyle="--", zorder=4)

# overlap bracket in Red
ov_lo = pmap(max(lo_A, lo_B))
ov_hi = pmap(min(hi_A, hi_B))
ax.annotate("", xy=(ov_hi, BAR_Y), xytext=(ov_lo, BAR_Y),
            arrowprops=dict(arrowstyle="<->", color=C_OVERLAP_RED_EDGE, lw=1.8),
            zorder=5)
ax.text((ov_lo + ov_hi) / 2.0, BAR_Y + 0.05, r"depth $d$",
    color=C_OVERLAP_RED_EDGE, fontsize=8.8,
    ha="center", va="bottom")

ax.text(0.03, 0.96, "6 edge-normal axes tested",
    transform=ax.transAxes, color=C_GREY_TEXT, fontsize=8.5,
    ha="left", va="top",
    bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.92))
ax.text(0.97, 0.07, "minimum-overlap axis",
    transform=ax.transAxes, color=C_GREY_TEXT, fontsize=8.3,
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.92))

# Reserve bottom annotation space
ax.plot([0.02, 0.98], [BAR_Y - 0.16, BAR_Y - 0.16], lw=0, zorder=0)

all_drawn_sat = np.vstack([pA_sat_V, pB_sat_V, tA, tB])
xmin_sat, ymin_sat = all_drawn_sat.min(axis=0)
xmax_sat, ymax_sat = all_drawn_sat.max(axis=0)
ax.set_xlim(xmin_sat - 0.22, xmax_sat + 0.14)
ax.set_ylim(min(ymin_sat - 0.28, -0.68), ymax_sat + 0.16)
ax.set_aspect("equal")
hide_axes(ax)

# ── Final shared cosmetics and Save ─────────────────────────────────────────
# fig.tight_layout()
os.makedirs("img", exist_ok=True)

# Ensure renderer exists for tight per-axis exports.
fig.canvas.draw()

# Save each quadrant/panel as an individual asset for LaTeX-side grid assembly.
for i, ax in enumerate(axes, start=1):
    extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(f"img/collision_panel_{i}.pdf", bbox_inches=extent, dpi=200, transparent=True)
    fig.savefig(f"img/collision_panel_{i}.png", bbox_inches=extent, dpi=240, transparent=True)

# Save transparent + white background variants
# Using transparent outputs helps with compositing,
# white background is best for direct inclusion.

# Transparent variants for design compositing
fig.savefig("img/collision_pipeline_transparent.pdf", bbox_inches="tight", dpi=200, transparent=True)
fig.savefig("img/collision_pipeline_transparent.png", bbox_inches="tight", dpi=240, transparent=True)

# White-background variants for print / direct LaTeX includegraphics
fig.savefig("img/collision_pipeline.pdf", bbox_inches="tight", dpi=200,
            facecolor="white", edgecolor="white", transparent=False)
fig.savefig("img/collision_pipeline.png", bbox_inches="tight", dpi=240,
            facecolor="white", edgecolor="white", transparent=False)

print("Saved transparent + white variants of professional figure in img/")
plt.show()