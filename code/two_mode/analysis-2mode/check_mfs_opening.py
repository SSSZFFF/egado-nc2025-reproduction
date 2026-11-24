"""Redo the MFS check with morphological opening (the correct tool).

The medial-axis width is small at EVERY convex corner, so it over-reports thin
features.  Opening with a disk of radius MFS/2 removes exactly those parts of
the pattern that are thinner than the MFS -- that is the standard test.
"""
import os
import sys

import numpy as np
from scipy import ndimage
from skimage.morphology import binary_opening, disk

MESH_NM = 20.0
SCALE = 6
MFS_NM = 120.0
R = int(round(MFS_NM / MESH_NM / 2))          # disk radius in mesh pixels = 3

z = np.load(r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_final.npz")
eps = z["eps"]
an = eps.T[:-1, :-1]
an = -1 * np.flip(an, axis=0)
an = (an - an.min()) / (an.max() - an.min())
si = an < 0.5
ny, nx = si.shape
frac = si.reshape(ny // SCALE, SCALE, nx // SCALE, SCALE).mean(axis=(1, 3))
mixed = (frac > 0.0001) & (frac < 0.9999)

print("opening with disk radius %d mesh px (= %.0f nm), MFS = %.0f nm" % (R, 2 * R * MESH_NM, MFS_NM))
print()

# --- silicon thinner than the MFS -----------------------------------------
op_si = binary_opening(si, disk(R))
thin_si = si & ~op_si
lbl, n = ndimage.label(thin_si)
areas = ndimage.sum(thin_si, lbl, range(1, n + 1)) * MESH_NM ** 2
print("silicon thinner than MFS: %d patches, %.2f%% of the silicon area"
      % (n, 100.0 * thin_si.sum() / si.sum()))
if n:
    print("   patch areas (nm^2): min %.0f  median %.0f  max %.0f"
          % (areas.min(), np.median(areas), areas.max()))

# --- holes thinner than the MFS -------------------------------------------
op_h = binary_opening(~si, disk(R))
thin_h = ~si & ~op_h
lbl2, n2 = ndimage.label(thin_h)
areas2 = ndimage.sum(thin_h, lbl2, range(1, n2 + 1)) * MESH_NM ** 2
print("holes thinner than MFS  : %d patches, %.2f%% of the hole area"
      % (n2, 100.0 * thin_h.sum() / (~si).sum()))
if n2:
    print("   patch areas (nm^2): min %.0f  median %.0f  max %.0f"
          % (areas2.min(), np.median(areas2), areas2.max()))

# --- does the composition criterion flag every offending cell? ------------
viol = thin_si | thin_h
cell_viol = viol.reshape(ny // SCALE, SCALE, nx // SCALE, SCALE).any(axis=(1, 3))
print()
print("=== coverage by the composition criterion ===")
print("  cells containing sub-MFS material : %d" % cell_viol.sum())
print("  of those flagged MIXED            : %d" % int((cell_viol & mixed).sum()))
print("  of those NOT flagged (leak)       : %d" % int((cell_viol & ~mixed).sum()))
leak = cell_viol & ~mixed
if leak.sum():
    ys, xs = np.where(leak)
    print("  leak locations (um):")
    for y_, x_ in list(zip(ys, xs))[:15]:
        # undo the flip/transpose used above: index a = y_rev, b = x
        yy = (ny - 1 - (y_ * SCALE)) / (ny - 1) * 3.6 - 1.8
        xx = (x_ * SCALE) / (nx - 1) * 3.6 - 1.8
        print("     cell(%2d,%2d)  (x~%+.2f, y~%+.2f) um   Si fraction %.3f"
              % (y_, x_, xx, yy, frac[y_, x_]))

print()
print("=== verdict ===")
if leak.sum() == 0:
    print("  composition criterion covers EVERY sub-MFS cell -> no extra MFS check needed")
else:
    print("  %d cell(s) contain sub-MFS material yet look 'uniform' ->" % leak.sum())
    print("  an explicit MFS pass would still be required on top of the composition test")
