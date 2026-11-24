"""Lexicographic DRC rule: how many candidate pixels still need FDTD?

For each digital pixel i we have D_i^0 (violation if forced to oxide) and
D_i^1 (if forced to silicon), both measured on the whole pattern.

    D_i^0 != D_i^1  ->  the manufacturing rule alone picks the winner, no FDTD
    D_i^0 == D_i^1 < D(B) -> both choices repair equally well, only the optical
                             FoM can decide -> 2 FDTD solves
"""
import os
import sys

import numpy as np
from skimage.morphology import binary_opening, disk

MESH_NM = 20.0
SCALE = 6
MFS_NM = 120.0
R = 3

z = np.load(r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_final.npz")
an = z["eps"].T[:-1, :-1]
an = -1 * np.flip(an, axis=0)
an = (an - an.min()) / (an.max() - an.min())
B = an < 0.5
ny, nx = B.shape
NC = nx // SCALE
SE = disk(R)


def violation(mask):
    return int((mask & ~binary_opening(mask, SE)).sum()
               + (~mask & ~binary_opening(~mask, SE)).sum())


D0 = violation(B)
print("D(B) = %d mesh cells" % D0)
print("sweeping ...")

Dox = np.zeros((NC, NC), dtype=int)
Dsi = np.zeros((NC, NC), dtype=int)
for i in range(NC):
    for j in range(NC):
        sl = (slice(i * SCALE, (i + 1) * SCALE), slice(j * SCALE, (j + 1) * SCALE))
        b1 = B.copy(); b1[sl] = True
        b0 = B.copy(); b0[sl] = False
        Dsi[i, j] = violation(b1)
        Dox[i, j] = violation(b0)

better = np.minimum(Dox, Dsi)
cand = better < D0
determined = cand & (Dox != Dsi)          # DRC alone decides
ambiguous = cand & (Dox == Dsi)           # need optical FoM

print()
print("=== candidate breakdown (D(B) = %d) ===" % D0)
print("  flips that reduce the violation      : %d" % cand.sum())
print("    of which DRC alone decides the state (0 FDTD) : %d" % determined.sum())
print("    of which both states repair equally (2 FDTD)  : %d" % ambiguous.sum())
print()
print("=== FDTD solves for the digital stage ===")
rows = [("EGADO D=0", 238), ("EGADO D=2", 308), ("EGADO D=4", 365)]
ours = 2 * int(ambiguous.sum())
for name, n in rows:
    print("  %-12s %3d TBD  ->  %3d FDTD solves" % (name, n, 2 * n))
print("  %-12s %3d TBD  ->  %3d FDTD solves   (%.1f%% fewer than D=2)"
      % ("DRC-guided", int(cand.sum()), ours, 100.0 * (1 - ours / 616.0)))
print()
if determined.sum():
    ys, xs = np.where(determined)
    base = D0
    print("  examples of DRC-decided pixels (first 8):")
    for k in range(min(8, len(ys))):
        i, j = ys[k], xs[k]
        pick = "Si" if Dsi[i, j] < Dox[i, j] else "oxide"
        print("     cell(%2d,%2d)  D_oxide=%3d  D_silicon=%3d  -> choose %-5s (no FDTD)"
              % (i, j, Dox[i, j], Dsi[i, j], pick))
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_drc", "Dox.npy"), Dox)
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_drc", "Dsi.npy"), Dsi)
