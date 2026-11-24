"""Independent verification of the DBS handover package.

Checks, all with our OWN code paths rather than the package's tools:

  1. A's 238 candidates against EGADO's D=0 Canny edge set, recomputed from
     inputs/common/2mode_final.npz with get_conversion_matrix(buff=0)
  2. A n B intersection (package claims 106)
  3. candidate_mask vs candidates.csv vs order_rc consistency
  4. an independent KLayout width/space run on both initial.gds files, to see
     whether the archived numbers (A = 22 official-rule markers, B = 0) are
     at least consistent with what the geometry itself shows
"""
import csv
import os
import sys

import numpy as np
import klayout.db as db

PKG = r"G:\codex_project\inverse_design_sun\DBS_交接包_2026-10-11"
REPO = r"G:\codex_project\inverse_design_sun\EGADO-2mode"
sys.path.insert(0, REPO)
os.chdir(REPO)
from func_get_conversion_matrix import get_conversion_matrix  # noqa: E402

WORK = os.path.join(PKG, "_verify_tmp")
os.makedirs(WORK, exist_ok=True)

# ---------------------------------------------------------------- load inputs
A = np.load(os.path.join(PKG, r"inputs\A_D0\initial_state.npz"))
B = np.load(os.path.join(PKG, r"inputs\B_curve_DRC120\initial_state.npz"))
maskA, silA, rcA = A["candidate_mask"], A["silicon_state"], A["order_rc"]
maskB, silB, rcB = B["candidate_mask"], B["silicon_state"], B["order_rc"]
print("A: mask %d cells, order_rc %s" % (maskA.sum(), rcA.shape))
print("B: mask %d cells, order_rc %s" % (maskB.sum(), rcB.shape))


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


csvA = read_csv(os.path.join(PKG, r"inputs\A_D0\candidates.csv"))
csvB = read_csv(os.path.join(PKG, r"inputs\B_curve_DRC120\candidates.csv"))
print("A csv rows %d, B csv rows %d" % (len(csvA), len(csvB)))

# ------------------------------------------------- 3. internal consistency
print()
print("=== 3. internal consistency (mask vs csv vs order_rc) ===")
for tag, mask, rc, rows, sil in (("A", maskA, rcA, csvA, silA), ("B", maskB, rcB, csvB, silB)):
    from_csv = np.zeros((30, 30), dtype=bool)
    for r in rows:
        from_csv[int(r["row_y_up"]), int(r["column_x_right"])] = True
    from_rc = np.zeros((30, 30), dtype=bool)
    for r, c in rc:
        from_rc[r, c] = True
    print("  %s: mask==csv %s   mask==order_rc %s   csv==order_rc %s"
          % (tag, np.array_equal(mask, from_csv), np.array_equal(mask, from_rc),
             np.array_equal(from_csv, from_rc)))
    # geometry: does x_centre / y_centre follow from row/col?
    bad = 0
    for r in rows:
        c, rr = int(r["column_x_right"]), int(r["row_y_up"])
        if abs(float(r["x_centre_um"]) - (-1.8 + (c + 0.5) * 0.12)) > 1e-9:
            bad += 1
        if abs(float(r["y_centre_um"]) - (-1.8 + (rr + 0.5) * 0.12)) > 1e-9:
            bad += 1
    print("      coordinate mismatches: %d" % bad)
    # material vs silicon_state
    mm = 0
    for r in rows:
        si = (r["initial_material"] == "silicon")
        st = (r["legacy_state_0_Si_1_oxide"] == "0")
        if si != st:
            mm += 1
        if bool(sil[int(r["row_y_up"]), int(r["column_x_right"])]) != si:
            mm += 1
    print("      material/state mismatches: %d" % mm)

# ------------------------------------------------- 2. intersection
inter = maskA & maskB
print()
print("=== 2. intersection ===")
print("  |A| = %d   |B| = %d   |A n B| = %d   (package claims 106)"
      % (maskA.sum(), maskB.sum(), inter.sum()))

# ------------------------------------------------- 1. A vs the EGADO D=0 set
print()
print("=== 1. is A really EGADO's D=0 Canny edge set? ===")
cm = get_conversion_matrix(analog_file=os.path.join(PKG, r"inputs\common\2mode_final.npz"),
                           target_folder=WORK, pixel_size=120e-9, mesh_size=20, buff=0)
E0 = (cm >= 0.49) & (cm <= 0.51)
print("  EGADO D=0 edge mask: %d cells (expected 238)" % E0.sum())


def symmetries(m):
    out = {}
    for k in range(4):
        r = np.rot90(m, k)
        out["rot%d" % (90 * k)] = r
        out["rot%d+flip" % (90 * k)] = np.fliplr(r)
    return out


best = None
for name, cand in symmetries(E0).items():
    if cand.shape != maskA.shape:
        continue
    d = int((cand != maskA).sum())
    if best is None or d < best[1]:
        best = (name, d)
    if d == 0:
        print("  EXACT match under transform: %s" % name)
print("  closest symmetry: %s, differing cells = %d" % best)

# ------------------------------------------------- 4. independent DRC
print()
print("=== 4. independent KLayout width/space on the initial GDS files ===")
MESH_NM = 120.0        # these GDS files are on the 120 nm digital grid
DBU = 0.0001           # um per dbu, as declared in the package


def gds_to_binary(path, nx=30, ny=30):
    """Read ONLY the device layer (13/1); 13/2 and 251/16 are background/frame."""
    lay = db.Layout()
    lay.read(path)
    top = lay.top_cell()
    dev_idx = None
    for li in range(lay.layers()):
        info = lay.get_info(li)
        if (info.layer, info.datatype) == (13, 1):
            dev_idx = li
            break
    if dev_idx is None:
        raise RuntimeError("device layer 13/1 not found in %s" % path)
    r = db.Region(top.begin_shapes_rec(dev_idx))
    r.merge()
    B = np.zeros((ny, nx), dtype=bool)
    for poly in r.each():
        bb = poly.bbox()
        c1 = int(round(bb.left * DBU / (MESH_NM * 1e-3)))
        c2 = int(round(bb.right * DBU / (MESH_NM * 1e-3)))
        r1 = int(round(bb.bottom * DBU / (MESH_NM * 1e-3)))
        r2 = int(round(bb.top * DBU / (MESH_NM * 1e-3)))
        B[r1:r2, c1:c2] = True
    return r, B


def drc_counts(region, extent_um, thr_nm=120):
    thr = thr_nm / (DBU * 1000.0)
    wall = db.Region(db.Box(0, 0, int(round(extent_um / DBU)), int(round(extent_um / DBU))))
    inv = wall - region
    inv.merge()
    return (region.width_check(thr).count(), region.space_check(thr).count(),
            inv.width_check(thr).count(), int(thr))


for tag, path in (("A", os.path.join(PKG, r"inputs\A_D0\initial.gds")),
                  ("B", os.path.join(PKG, r"inputs\B_curve_DRC120\initial.gds"))):
    reg, Bg = gds_to_binary(path)
    w, s, o, thr = drc_counts(reg, 3.6)
    print("  %s: cells=%d  Si=%.1f%%   120 nm check -> Si-width %d, Si-space %d, oxide-width %d"
          % (tag, Bg.sum(), 100 * Bg.mean(), w, s, o))
    print("      (threshold passed to KLayout = %d dbu = %.0f nm)" % (thr, thr * DBU * 1000))
    # compare with the npz state
    match = np.array_equal(Bg, silA if tag == "A" else silB)
    print("      matches initial_state silicon_state: %s" % match)
    if not match:
        for name, cand in symmetries(silA if tag == "A" else silB).items():
            if cand.shape == Bg.shape and np.array_equal(cand, Bg):
                print("      ...exact under transform %s" % name)
                break
