"""Do the sub-MFS (unmanufacturable) features fall inside MIXED cells?

If yes, the composition criterion automatically flags every manufacturability
violation for DBS.  If some fall inside 'uniform' cells, they would be mapped
directly and silently kept -- that would be the failure mode to worry about.
"""
import os
import sys

import numpy as np
from scipy import ndimage

MESH_NM = 20.0
SCALE = 6
MFS_NM = 120.0

z = np.load(r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_final.npz")
eps = z["eps"]
an = eps.T[:-1, :-1]
an = -1 * np.flip(an, axis=0)
an = (an - an.min()) / (an.max() - an.min())
si = an < 0.5
ny, nx = si.shape

frac = si.reshape(ny // SCALE, SCALE, nx // SCALE, SCALE).mean(axis=(1, 3))
mixed = (frac > 0.0001) & (frac < 0.9999)


def cell_of(iy, ix):
    return iy // SCALE, ix // SCALE


print("=== 1) sub-MFS holes ===")
lbl, n = ndimage.label(~si)
objs = ndimage.find_objects(lbl)
bad = 0
tot = 0
for i, sl in enumerate(objs, start=1):
    area = int((lbl[sl] == i).sum()) * MESH_NM ** 2
    if area >= MFS_NM ** 2:
        continue
    tot += 1
    cells = set()
    ys, xs = np.where(lbl == i)
    for y_, x_ in zip(ys, xs):
        cells.add(cell_of(y_, x_))
    flag = all(mixed[c] for c in cells)
    bad += 1 if flag else 0
    if tot <= 12:
        print("  hole @(%6.2f,%6.2f) um  area %6.0f nm^2  touches %d cell(s)  all MIXED: %s"
              % (z["y"][min(xs)] * 1e6 if False else 0, 0, area, len(cells), flag))
print("  total sub-MFS holes: %d ; fully inside MIXED cells: %d" % (tot, bad))

print()
print("=== 2) narrow silicon necks (medial width < MFS) ===")
dt = ndimage.distance_transform_edt(si)
inside = si & (dt > 0)
axis = np.zeros_like(si, dtype=bool)
for ax in (0, 1):
    axis |= (dt == ndimage.maximum_filter1d(dt, 3, axis=ax))
axis &= inside
narrow = axis & ((2 * dt * MESH_NM) < MFS_NM)
ys, xs = np.where(narrow)
print("  narrow medial points: %d" % len(ys))
in_mixed = 0
for y_, x_ in zip(ys, xs):
    if mixed[cell_of(y_, x_)]:
        in_mixed += 1
print("  of those, sitting in a MIXED cell : %d (%.1f%%)" % (in_mixed, 100.0 * in_mixed / max(1, len(ys))))
print("  NOT in a mixed cell               : %d" % (len(ys) - in_mixed))

print()
print("=== 3) summary ===")
print("  mixed cells                : %d of 900" % mixed.sum())
print("  uniform-Si cells           : %d" % int((frac > 0.9999).sum()))
print("  uniform-hole cells         : %d" % int((frac < 0.0001).sum()))
viol = np.zeros_like(si)
viol |= narrow
for i, sl in enumerate(ndimage.find_objects(ndimage.label(~si)[0]), start=1):
    arr = ndimage.label(~si)[0]
    if int((arr[sl] == i).sum()) * MESH_NM ** 2 < MFS_NM ** 2:
        viol |= (arr == i)
cell_has_viol = viol.reshape(ny // SCALE, SCALE, nx // SCALE, SCALE).any(axis=(1, 3))
print("  cells containing ANY sub-MFS feature : %d" % cell_has_viol.sum())
print("  of those, flagged as MIXED           : %d" % int((cell_has_viol & mixed).sum()))
print("  of those, NOT flagged (leak!)        : %d" % int((cell_has_viol & ~mixed).sum()))
