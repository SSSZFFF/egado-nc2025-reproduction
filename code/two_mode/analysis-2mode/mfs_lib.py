"""Reusable KLayout-backed MFS utilities.

Conventions
-----------
* boolean arrays are on the 20 nm simulation grid, True = SILICON (high eps)
* all thresholds go through `thr_dbu()`; the controls and the repair loop must
  use the same function.  An earlier version verified the control path with a
  literal `120` while the main path had a broken unit conversion, which produced
  a confident but completely wrong "0 violations" result.
"""
import numpy as np
import klayout.db as db

MESH_NM = 20.0
MFS_NM = 120.0
DBU = 0.001                      # um per database unit


def thr_dbu(mfs_nm=MFS_NM, dbu=DBU):
    """1 dbu = dbu*1000 nm, so a 120 nm rule is 120 dbu."""
    return mfs_nm / (dbu * 1000.0)


THR = thr_dbu()


def to_region(B):
    ny, nx = B.shape
    lay = db.Layout()
    lay.dbu = DBU
    cell = lay.create_cell("T")
    layer = lay.layer(1, 0)
    for a in range(ny):
        b = 0
        while b < nx:
            if B[a, b]:
                b0 = b
                while b < nx and B[a, b]:
                    b += 1
                cell.shapes(layer).insert(db.Box(
                    int(round(b0 * MESH_NM * 1e-3 / DBU)), int(round(a * MESH_NM * 1e-3 / DBU)),
                    int(round(b * MESH_NM * 1e-3 / DBU)), int(round((a + 1) * MESH_NM * 1e-3 / DBU))))
            else:
                b += 1
    r = db.Region(cell.begin_shapes_rec(layer))
    r.merge()
    return r


def frame_region(B):
    ny, nx = B.shape
    f = db.Region(db.Box(0, 0, int(round(nx * MESH_NM * 1e-3 / DBU)),
                         int(round(ny * MESH_NM * 1e-3 / DBU))))
    f.merge()
    return f


def counts(B):
    """(Si narrower than MFS, gap narrower than MFS, oxide narrower than MFS)."""
    r = to_region(B)
    inv = frame_region(B) - r
    inv.merge()
    return (r.width_check(THR).count(),
            r.space_check(THR).count(),
            inv.width_check(THR).count())


def violation_mask(B, which):
    """Rasterise KLayout's violation bounding boxes onto the 20 nm grid."""
    r = to_region(B)
    if which == "oxide":
        inv = frame_region(B) - r
        inv.merge()
        ep = inv.width_check(THR)
    elif which == "si_width":
        ep = r.width_check(THR)
    else:
        ep = r.space_check(THR)
    m = np.zeros(B.shape, dtype=bool)
    for e in ep.each():
        bb = e.bbox()
        x1 = int(np.floor(bb.left * DBU / (MESH_NM * 1e-3)))
        x2 = int(np.ceil(bb.right * DBU / (MESH_NM * 1e-3)))
        y1 = int(np.floor(bb.bottom * DBU / (MESH_NM * 1e-3)))
        y2 = int(np.ceil(bb.top * DBU / (MESH_NM * 1e-3)))
        m[max(0, y1):min(B.shape[0], y2), max(0, x1):min(B.shape[1], x2)] = True
    return m


def project(B, max_iter=25, verbose=False):
    """Greedy, KLayout-driven local repair.  Never makes a global change."""
    B = B.copy()
    for it in range(max_iter):
        c = counts(B)
        if sum(c) == 0:
            if verbose:
                print("      clean after %d rounds" % it)
            return B, it, True
        changed = False
        m = violation_mask(B, "si_width")
        if m.any():
            B[m] = False
            changed = True
        m = violation_mask(B, "space")
        if m.any():
            B[m] = True
            changed = True
        m = violation_mask(B, "oxide")
        if m.any():
            B[m] = True
            changed = True
        if verbose and it % 5 == 0:
            print("      [%2d] %s" % (it, counts(B)))
        if not changed:
            break
    return B, max_iter, sum(counts(B)) == 0
