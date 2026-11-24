"""D(B) v2: local-width criterion instead of disk-fitting.

Why v1 was wrong
----------------
v1 used  thin = B & ~opening(B, disk(r)).
 * A discrete disk of radius r only lets features of width >= 2r+1 *pixels*
   survive, so r=3 tested 140 nm, not the intended 120 nm.
 * Opening with a disk ROUNDS OFF CONVEX CORNERS; the removed corners are then
   counted as violations.  A 180 nm square island reported 20 violation cells.

v2
--
Measure the LOCAL WIDTH of each region on its own ridge (medial axis):
    dt    = distance transform (distance to the nearest other-material pixel)
    ridge = dt is a local maximum along BOTH axes simultaneously
            (this is what excludes convex-corner branches, where dt rises
             monotonically towards the interior and is therefore not a maximum)
    width = 2*dt at ridge points
A ridge point violates the MFS if width < MFS.

Calibration on straight lines of known width tells us the exact width->2*dt
mapping at 20 nm sampling.
"""
import numpy as np
from scipy import ndimage

MESH_NM = 20.0
N = 120


def make(kind, w):
    B = np.zeros((N, N), dtype=bool)
    c = N // 2
    if kind == "silicon line":
        B[c - w // 2: c - w // 2 + w, :] = True
    elif kind == "oxide gap":
        B[:] = True
        B[c - w // 2: c - w // 2 + w, :] = False
    elif kind == "silicon island":
        B[c - w // 2: c - w // 2 + w, c - w // 2: c - w // 2 + w] = True
    elif kind == "oxide hole":
        B[:] = True
        B[c - w // 2: c - w // 2 + w, c - w // 2: c - w // 2 + w] = False
    elif kind == "silicon L":
        B[c - w: c + w, :] = True          # 2w-wide bar
        B[:, c - w: c + w] = True
        B[c - w: c + w, :] = True
    return B


def ridges(mask):
    """Ridge points of `mask` with their local width in mesh cells."""
    dt = ndimage.distance_transform_edt(mask)
    mx0 = ndimage.maximum_filter1d(dt, 3, axis=0)
    mx1 = ndimage.maximum_filter1d(dt, 3, axis=1)
    ridge = mask & (dt > 0) & (dt >= mx0) & (dt >= mx1)
    return ridge, 2.0 * dt


def violation_v2(B, mfs_cells):
    """Sum of ridge lengths (in mesh cells) that are thinner than the MFS."""
    v = 0
    for m in (B, ~B):
        ridge, width = ridges(m)
        v += int((ridge & (width < mfs_cells)).sum())
    return v


print("=== calibration: straight lines of known width ===")
print("  %-10s %-10s %-12s %s" % ("width_nm", "width_cells", "max 2*dt", "2*dt at ridge"))
for w in (4, 5, 6, 7, 8, 9, 10, 12):
    B = make("silicon line", w)
    ridge, width = ridges(B)
    print("  %-10.0f %-10d %-12.1f %s" % (
        w * MESH_NM, w, width.max(), sorted(set(width[ridge].tolist()))))

print()
print("=== pass/fail of v2 at MFS = 120 nm (mfs_cells = 6) ===")
KINDS = ["silicon line", "oxide gap", "silicon island", "oxide hole", "silicon L"]
WIDTHS_NM = [80, 100, 120, 140, 160, 180, 200]
hdr = "  %-16s" % "structure"
for w in WIDTHS_NM:
    hdr += "%8d" % w
print(hdr + "    (nm -> violation cells)")
for kind in KINDS:
    row = "  %-16s" % kind
    for wnm in WIDTHS_NM:
        d = violation_v2(make(kind, wnm // 20), 6)
        row += "%8s" % ("PASS" if d == 0 else str(d))
    print(row)

print()
print("=== corner behaviour (the v1 failure mode) ===")
for w in (6, 8, 10):
    B = make("silicon island", w)
    ridge, width = ridges(B)
    print("  %3d nm island : v2 violation = %d cells, ridge width range %s"
          % (w * 20, violation_v2(B, 6), sorted(set(width[ridge].tolist()))))
