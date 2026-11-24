"""Render a top view of the 2-mode MUX geometry from dump_geometry.py's JSON.

usage:  python draw_layout.py <geo.json> <out.png>
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

geo = json.load(open(sys.argv[1]))
out_png = os.path.abspath(sys.argv[2])
objs = geo["objects"]
UM = 1e6

DARK = "#3d3d3d"
BOX = "#dcdcdc"


def box(name):
    e = objs[name]
    x, y = e["x"] * UM, e["y"] * UM
    w, h = e["x span"] * UM, e["y span"] * UM
    return x - w / 2, y - h / 2, w, h


fig, ax = plt.subplots(figsize=(8.4, 6.0))

# --- FDTD region -------------------------------------------------------
x, y, w, h = box("FDTD")
ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#9a9a9a", ls=(0, (6, 4)), lw=1.0, zorder=1))
ax.text(x + 0.08, y + h - 0.30, "FDTD  {:.1f} x {:.1f} um".format(w, h),
        color="#8a8a8a", fontsize=8, zorder=6)

# --- design region -----------------------------------------------------
x, y, w, h = box("opt_fields")
ax.add_patch(Rectangle((x, y), w, h, fc=BOX, ec="#222222", lw=1.5, zorder=2))
ax.text(0, 0.10, "Two-mode MUX", ha="center", va="center", fontsize=13, zorder=6)
ax.text(0, -0.30, "{:.1f} x {:.1f} um".format(w, h), ha="center", va="center",
        fontsize=9, color="#444444", zorder=6)
ax.text(0, -0.62, "30 x 30 px @ 120 nm", ha="center", va="center",
        fontsize=9, color="#444444", zorder=6)

# --- waveguides --------------------------------------------------------
for i in (1, 2):
    x, y, w, h = box("input wg {}".format(i))
    ax.add_patch(Rectangle((x, y), w, h, fc=DARK, ec="none", zorder=4))
xo, yo, wo, ho = box("output wg1")
ax.add_patch(Rectangle((xo, yo), wo, ho, fc=DARK, ec="none", zorder=4))

# --- monitors ----------------------------------------------------------
x, y, w, h = box("fom1")
ax.plot([x, x], [y, y + h], color="#2471a3", lw=2.4, zorder=5)

# --- in / out labels ---------------------------------------------------
ax.text(-4.62, 0.0, "in", ha="right", va="center", fontsize=9, color="#666666")
ax.text(4.62, yo, "out", ha="left", va="center", fontsize=9, color="#666666")

# --- dimension: input pitch (inside the design region, clear of guides) --
y1 = objs["input wg 1"]["y"] * UM
y2 = objs["input wg 2"]["y"] * UM
ax.annotate("", xy=(-1.58, y1), xytext=(-1.58, y2),
            arrowprops=dict(arrowstyle="<|-|>", color="#111111", lw=1.2,
                            shrinkA=0, shrinkB=0), zorder=6)
ax.text(-1.50, (y1 + y2) / 2, "1.3 um", ha="left", va="center", fontsize=10, zorder=6)

# --- dimension: input width (on the upper guide) ------------------------
xw = -3.30
ax.annotate("", xy=(xw, y1 + 0.25), xytext=(xw, y1 - 0.25),
            arrowprops=dict(arrowstyle="<|-|>", color="#111111", lw=1.2,
                            shrinkA=0, shrinkB=0), zorder=6)
ax.plot([xw, xw], [y1 + 0.25, y1 + 0.44], color="#111111", lw=0.9, zorder=6)
ax.text(xw, y1 + 0.50, "0.5 um", ha="center", va="bottom", fontsize=10, zorder=6)

# --- dimension: output width -------------------------------------------
ax.annotate("", xy=(3.05, yo + ho / 2), xytext=(3.05, yo - ho / 2),
            arrowprops=dict(arrowstyle="<|-|>", color="#111111", lw=1.2,
                            shrinkA=0, shrinkB=0), zorder=6)
ax.text(3.18, yo, "0.9 um", ha="left", va="center", fontsize=10, zorder=6)

# --- sources (drawn above the guides so nothing overlaps) ---------------
for i in (1, 2):
    ys = objs["Source_TE_mode{}_1550".format(i)]["y"] * UM
    ax.annotate("", xy=(-1.86, ys), xytext=(-2.72, ys),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.6), zorder=7)
ax.text(-2.80, 0.0, "TE$_0$ x2", ha="right", va="center", color="#c0392b", fontsize=10, zorder=7)

ax.set_xlim(-5.05, 5.05)
ax.set_ylim(-2.45, 2.45)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("EGADO 2-mode MUX   (paper: Nat. Commun. 16, 2372, Fig. S8)\n"
             "2 single-mode inputs, 500 nm wide, 1.3 um pitch   ->   multimode output, 900 nm",
             fontsize=10.5)
plt.tight_layout()
plt.savefig(out_png, dpi=170)
print("wrote", out_png)
