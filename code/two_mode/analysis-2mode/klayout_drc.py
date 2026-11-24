"""Validate the cheap Python DRC metrics against KLayout, the real DRC engine.

Builds a GDS from the binary pattern (exact 20 nm geometry, row-merged
rectangles), runs KLayout's width_check / space_check at 120 nm, and compares
the violation footprint with the three hand-rolled Python metrics.
"""
import os
import sys

import numpy as np
import klayout.db as db

MESH_NM = 20.0
MFS_NM = 120.0
DBU = 0.001                      # 1 nm database unit; coordinates in um

z = np.load(r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_final.npz")
an = z["eps"].T[:-1, :-1]
an = -1 * np.flip(an, axis=0)
an = (an - an.min()) / (an.max() - an.min())
B = an < 0.5                     # True = silicon
ny, nx = B.shape
print("pattern %s   silicon %.1f%%" % (B.shape, 100 * B.mean()))

# ---------------------------------------------------------------- build GDS
layout = db.Layout()
layout.dbu = DBU
top = layout.create_cell("TOP")
layer = layout.layer(1, 0)

n_rect = 0
for a in range(ny):                      # a is the row index
    b = 0
    while b < nx:
        if B[a, b]:
            b0 = b
            while b < nx and B[a, b]:
                b += 1
            # rectangle for run [b0, b) in x, row a in y
            x1 = b0 * MESH_NM * 1e-3
            x2 = b * MESH_NM * 1e-3
            y1 = a * MESH_NM * 1e-3
            y2 = (a + 1) * MESH_NM * 1e-3
            top.shapes(layer).insert(db.Box(
                int(round(x1 / DBU)), int(round(y1 / DBU)),
                int(round(x2 / DBU)), int(round(y2 / DBU))))
            n_rect += 1
        else:
            b += 1
print("emitted %d rectangles" % n_rect)

region = db.Region(top.begin_shapes_rec(layer))
region.merge()
print("merged region area : %.4f um^2   (= %d mesh cells)"
      % (region.area() * DBU * DBU, round(region.area() * DBU * DBU / (0.02 ** 2))))

# ---------------------------------------------------------------- KLayout DRC
print()
print("=== KLayout DRC, minimum width / space = %.0f nm ===" % MFS_NM)
# NOTE: Region methods take the threshold in DATABASE UNITS, not um.
# With dbu = 0.001 um, a 120 nm rule is 120 dbu.  Passing 0.120 silently
# disables the check (0.12 dbu = 0.12 nm) and every structure "passes".
thr = MFS_NM * 1e-9 / DBU          # um -> dbu

width_viol = region.width_check(thr)          # EdgePairs
space_viol = region.space_check(thr)          # EdgePairs


def ep_area(ep):
    """Footprint of an EdgePairs collection, in um^2 (via its bounding boxes)."""
    try:
        return ep.extents().area() * DBU * DBU
    except Exception:
        return ep.polygons().area() * DBU * DBU


wa = ep_area(width_viol)
sa = ep_area(space_viol)
print("  width violations : %d edge pairs, footprint %.4f um^2" % (width_viol.count(), wa))
print("  space violations : %d edge pairs, footprint %.4f um^2" % (space_viol.count(), sa))
n_viol_cells = round((wa + sa) / (0.02 ** 2))
print("  total violation footprint ~ %d mesh cells" % n_viol_cells)

# the oxide is the complement: check it too by building the inverse region
inv = db.Region(db.Box(0, 0, int(round(nx * MESH_NM * 1e-3 / DBU)),
                       int(round(ny * MESH_NM * 1e-3 / DBU)))) - region
inv.merge()
wid_o = inv.width_check(thr)
spa_o = inv.space_check(thr)
print("  (oxide) width violations : %d edge pairs, footprint %.4f um^2"
      % (wid_o.count(), ep_area(wid_o)))
print("  (oxide) space violations : %d edge pairs, footprint %.4f um^2"
      % (spa_o.count(), ep_area(spa_o)))
n_viol_cells += round((ep_area(wid_o) + ep_area(spa_o)) / (0.02 ** 2))
print("  TOTAL (silicon + oxide)  ~ %d mesh cells" % n_viol_cells)

# ---------------------------------------------------------------- python v1/v3
print()
print("=== hand-rolled metrics on the same pattern ===")
from skimage.morphology import binary_opening, disk, square


def v1(B, r):
    return int((B & ~binary_opening(B, disk(r))).sum()
               + (~B & ~binary_opening(~B, disk(r))).sum())


def v3(B, k):
    return int((B & ~binary_opening(B, square(k))).sum()
               + (~B & ~binary_opening(~B, square(k))).sum())


print("  v1 disk(3)  = %6d mesh cells" % v1(B, 3))
print("  v3 square(6)= %6d mesh cells" % v3(B, 6))
print("  KLayout     = %6d mesh cells  <- ground truth" % n_viol_cells)
print()
print("  ratio v1/KLayout = %.2f" % (v1(B, 3) / max(1, n_viol_cells)))
print("  ratio v3/KLayout = %.2f" % (v3(B, 6) / max(1, n_viol_cells)))

# ------------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))


def r2a(reg, shape):
    m = np.zeros(shape, dtype=bool)
    for poly in reg.each():
        bb = poly.bbox()
        x1 = int(round(bb.left * DBU / (MESH_NM * 1e-3)))
        x2 = int(round(bb.right * DBU / (MESH_NM * 1e-3)))
        y1 = int(round(bb.bottom * DBU / (MESH_NM * 1e-3)))
        y2 = int(round(bb.top * DBU / (MESH_NM * 1e-3)))
        m[y1:y2, x1:x2] = True
    return m


ax = axes[0]
ax.imshow(B, origin="lower", cmap="bone_r", interpolation="nearest")
ax.set_title("binary pattern (silicon black)\nKLayout: %d width + %d space violations"
             % (width_viol.count(), space_viol.count()), fontsize=10)

ax = axes[1]
m = r2a(width_viol, B.shape) | r2a(space_viol, B.shape)
ax.imshow(B, origin="lower", cmap="bone_r", interpolation="nearest", alpha=0.35)
ax.imshow(np.ma.masked_where(~m, m), origin="lower", cmap="autumn",
          interpolation="nearest")
ax.set_title("KLayout violation footprint\n%d mesh cells (%.2f%% of design)"
             % (n_viol_cells, 100.0 * n_viol_cells / B.size), fontsize=10)

ax = axes[2]
pv1 = B & ~binary_opening(B, disk(3))
pv3 = B & ~binary_opening(B, square(6))
ax.imshow(B, origin="lower", cmap="bone_r", interpolation="nearest", alpha=0.35)
ax.imshow(np.ma.masked_where(~pv1, pv1), origin="lower", cmap="autumn", interpolation="nearest")
ax.imshow(np.ma.masked_where(~pv3, pv3), origin="lower", cmap="winter",
          interpolation="nearest", alpha=0.6)
ax.set_title("hand-rolled: v1 disk(3) red, v3 square(6) blue\n"
             "v1 %d cells, v3 %d cells" % (v1(B, 3), v3(B, 6)), fontsize=10)

for ax in axes:
    ax.axis("off")
fig.suptitle("KLayout DRC vs hand-rolled metrics", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.93])
out = r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_klayout_drc.png"
plt.savefig(out, dpi=140)
print("\nwrote", out)
layout.write(r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_pattern.gds")
print("wrote 2mode_pattern.gds")
