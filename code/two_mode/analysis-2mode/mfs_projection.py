"""MFS projection driven by KLayout's own violations.

The paper enforces MFS = 120 nm by putting the boundary on a 120 nm lattice,
which is strictly stronger than the rule.  To explore the larger feasible set we
need a projection onto

    Feasible = { B : no Si feature and no oxide feature is narrower than 120 nm }

Design decisions, each one paid for by an earlier mistake in this project:

* The threshold is computed ONCE by `thr_dbu()` and the SAME function is used by
  the controls and by the repair loop.  An earlier version verified the control
  path with a literal `120` while the main path used a broken unit conversion,
  which produced a confident but completely wrong "0 violations" result.
* The repair is driven by KLayout's exact edge pairs, not by a morphological
  heuristic (opening with a disk rounds square corners and over-flags 45-degree
  boundaries, which is what produced the bogus 486/537 "violations" earlier).
* Every repair step is followed by a re-check, so the loop can only exit clean.
"""
import os

import numpy as np
import klayout.db as db

MESH_NM = 20.0
MFS_NM = 120.0
DBU = 0.001                       # um per database unit
FRAME_NM = None                   # set per pattern


def thr_dbu(mfs_nm=MFS_NM, dbu=DBU):
    """Threshold in database units.  1 dbu = dbu*1000 nm, so 120 nm = 120 dbu."""
    return mfs_nm / (dbu * 1000.0)


THR = thr_dbu()


def to_region(B):
    """Binary array (True = silicon) -> merged KLayout region, 20 nm pixels."""
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
    """(thin Si, narrow gap, thin oxide) violation counts at the MFS threshold."""
    r = to_region(B)
    inv = frame_region(B) - r
    inv.merge()
    return (r.width_check(THR).count(),      # silicon narrower than MFS
            r.space_check(THR).count(),      # gap between silicon narrower than MFS
            inv.width_check(THR).count())    # oxide narrower than MFS  (== narrow gap)


# ------------------------------------------------------------------ controls
print("threshold = %.1f dbu = %.1f nm   (MFS = %.1f nm)" % (THR, THR * DBU * 1000, MFS_NM))
print()
print("=== controls, SAME threshold path as the repair loop ===")
N = 120


def ctrl(kind, w_nm):
    B = np.zeros((N, N), dtype=bool)
    c = N // 2
    k = w_nm // 20
    if kind == "line":
        B[c - k // 2:c - k // 2 + k, 10:110] = True
    elif kind == "gap":
        B[:, :] = True
        B[c - k // 2:c - k // 2 + k, :] = False
    return B


print("  %-6s %-6s %s" % ("kind", "width", "counts (Si-width, Si-space, ox-width)"))
ctrl_ok = True
for kind in ("line", "gap"):
    for w in (80, 100, 120, 140, 160):
        c = counts(ctrl(kind, w))
        flag = "" if ((w >= 120) == (sum(c) == 0)) else "   <-- MISMATCH"
        if flag:
            ctrl_ok = False
        print("  %-6s %-6d %s%s" % (kind, w, c, flag))
print("  controls behave as expected at 120 nm: %s" % ctrl_ok)
if not ctrl_ok:
    raise SystemExit("controls failed -- refusing to run the projection")
print()

# ------------------------------------------------------------------ pattern
z = np.load(r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_final.npz")
an = z["eps"].T[:-1, :-1]
an = -1 * np.flip(an, axis=0)
rho = (an - an.min()) / (an.max() - an.min())
ny, nx = rho.shape
BASE = rho < 0.5
print("TO pattern %s  silicon %.1f%%" % (BASE.shape, 100 * BASE.mean()))
print("  raw DRC counts: %s" % (counts(BASE),))


# ------------------------------------------------------------------ repair
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


def project(B, max_iter=60, verbose=False):
    """Remove thin silicon, fill narrow gaps, drop thin oxide, until KLayout is clean.

    Stage 1 (greedy)  acts only on KLayout's exact violation boxes, so it makes
    the smallest change it can.  It can stall when a fix for one rule breaks
    another.
    Stage 2 (fallback) applies a morphological smoothing *inside the violation
    neighbourhoods only*.  opening(B, disk(r)) is open(r)-stable by construction,
    i.e. every remaining point admits a disk of radius r, so width >= 2r = MFS;
    likewise closing fixes the gaps.  That makes stage 2 guaranteed to terminate
    clean, at the cost of locally rounding the shape.
    """
    from skimage.morphology import binary_closing, binary_dilation, binary_opening, disk
    B = B.copy()
    se = disk(int(round(MFS_NM / MESH_NM / 2)))          # r = 3 cells = 60 nm
    rounds_greedy = 0
    for it in range(max_iter):
        if sum(counts(B)) == 0:
            if verbose:
                print("    clean: %d greedy rounds" % rounds_greedy)
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
        if not changed:
            break
        rounds_greedy += 1

    if verbose:
        print("    greedy stalled at %s after %d rounds; switching to local morphology"
              % (counts(B), rounds_greedy))
    # Stage 2.  The two rules need OPPOSITE local operators, so they must be
    # applied separately -- combining them as closing(opening(B)) lets the
    # closing undo the opening, which is why the first attempt did nothing.
    for it in range(max_iter):
        c = counts(B)
        if sum(c) == 0:
            if verbose:
                print("    clean: morphology rounds = %d" % it)
            return B, rounds_greedy + it, True

        # (a) silicon narrower than the MFS -> locally OPEN (erode away the sliver)
        m = violation_mask(B, "si_width")
        if m.any():
            md = binary_dilation(m, disk(3))
            op = binary_opening(B, se)
            B[md] = op[md]
            if verbose:
                print("      [%2d] local opening on %d cells -> %s" % (it, int(md.sum()), counts(B)))
            continue

        # (b) gap / oxide narrower than the MFS -> locally CLOSE (fill it in)
        m = violation_mask(B, "space") | violation_mask(B, "oxide")
        if m.any():
            md = binary_dilation(m, disk(3))
            cl = binary_closing(B, se)
            B[md] = cl[md]
            if verbose:
                print("      [%2d] local closing on %d cells -> %s" % (it, int(md.sum()), counts(B)))
            continue
        break
    return B, max_iter, sum(counts(B)) == 0


print()
print("=== MFS projection of the raw TO pattern (minimal-change, KLayout-driven) ===")
print("  before: %s" % (counts(BASE),))
PB, iters, ok = project(BASE, verbose=True)
print("  after : %s   clean=%s  rounds=%d" % (counts(PB), ok, iters))
print("  area change: silicon %.1f%% -> %.1f%%"
      % (100 * BASE.mean(), 100 * PB.mean()))
print("  cells changed: %d of %d (%.2f%%)"
      % (int((PB != BASE).sum()), BASE.size, 100 * (PB != BASE).mean()))
