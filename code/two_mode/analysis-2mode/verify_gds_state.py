"""Cross-check every initial-state artefact of the handover package against its GDS.

For each scheme:
  * rasterise the device layer (13/1) inside the 3.6 x 3.6 um design region
    back onto the 30 x 30 digital grid
  * compare with initial_state.npz::silicon_state
  * compare with candidates.csv / final_cells_original.csv material columns
  * try all 8 symmetries so a pure orientation difference is not mistaken for a
    content difference
"""
import csv
import os

import numpy as np
import klayout.db as db

PKG = r"G:\codex_project\inverse_design_sun\DBS_交接包_2026-10-11"
DBU = 0.0001
CELL = 0.120          # um
N = 30


def raster(path):
    lay = db.Layout()
    lay.read(path)
    top = lay.top_cell()
    idx = [li for li in range(lay.layers())
           if (lay.get_info(li).layer, lay.get_info(li).datatype) == (13, 1)][0]
    r = db.Region(top.begin_shapes_rec(idx))
    r.merge()
    # sample the CENTRE of each 120 nm cell; cell (row r, col c) centre is
    # x = -1.8 + (c+0.5)*0.12, y = -1.8 + (r+0.5)*0.12
    M = np.zeros((N, N), dtype=bool)
    for rr in range(N):
        for cc in range(N):
            x = -1.8 + (cc + 0.5) * CELL
            y = -1.8 + (rr + 0.5) * CELL
            box = db.Box(int(round((x - 0.001) / DBU)), int(round((y - 0.001) / DBU)),
                         int(round((x + 0.001) / DBU)), int(round((y + 0.001) / DBU)))
            M[rr, cc] = not (r & db.Region(box)).is_empty()
    return M


def syms(m):
    out = {}
    for k in range(4):
        rot = np.rot90(m, k)
        out["rot%d" % (90 * k)] = rot
        out["rot%d+fliplr" % (90 * k)] = np.fliplr(rot)
    return out


def best_match(state, ref):
    b = None
    for name, cand in syms(state).items():
        d = int((cand != ref).sum())
        if b is None or d < b[1]:
            b = (name, d)
    return b


for tag, sub, csvname in (("A", r"inputs\A_D0", "candidates.csv"),
                          ("B", r"inputs\B_curve_DRC120", "candidates.csv")):
    gds = os.path.join(PKG, sub, "initial.gds")
    st = np.load(os.path.join(PKG, sub, "initial_state.npz"))
    sil = st["silicon_state"]
    mask = st["candidate_mask"]
    G = raster(gds)
    print("=== scheme %s ===" % tag)
    print("  GDS raster          : %d silicon of %d  (%.1f%%)" % (G.sum(), G.size, 100 * G.mean()))
    print("  silicon_state       : %d silicon of %d  (%.1f%%)" % (sil.sum(), sil.size, 100 * sil.mean()))
    print("  candidate_mask      : %d cells" % mask.sum())
    print("  GDS vs silicon_state: identical=%s   closest symmetry=%s (diff %d cells)"
          % (np.array_equal(G, sil), *best_match(sil, G)))
    if not np.array_equal(G, sil):
        same = int((G == sil).sum())
        print("      agreement = %d/%d = %.1f%%" % (same, G.size, 100 * same / G.size))
    # candidates.csv material column vs the GDS raster
    rows = list(csv.DictReader(open(os.path.join(PKG, sub, csvname), newline="", encoding="utf-8")))
    bad = 0
    for row in rows:
        rr, cc = int(row["row_y_up"]), int(row["column_x_right"])
        if bool(G[rr, cc]) != (row["initial_material"] == "silicon"):
            bad += 1
    print("  candidates.csv material vs GDS: %d mismatches of %d" % (bad, len(rows)))
    print()
