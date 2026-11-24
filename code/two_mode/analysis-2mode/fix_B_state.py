"""Produce a CORRECTED initial_state for scheme B, taken from its own GDS.

Finding: inputs/B_curve_DRC120/initial_state.npz::silicon_state has only 93 True
cells.  Its 66 candidate cells agree exactly with candidates.csv, but the 792
non-candidate cells are almost all False, whereas B's own initial.gds has 707
silicon cells inside the design region.  The array is missing the fixed curve
background.  (optimized_cells_original.npz carries the same array.)

The authoritative geometry -- initial.gds and candidates.csv -- is internally
consistent (0 mismatches), so the corrected state is simply the GDS rasterised
onto the 30 x 30 grid.  A's artefacts are already consistent and are left alone.

Writes:  inputs/B_curve_DRC120/initial_state_CORRECTED.npz
         inputs/B_curve_DRC120/initial_state_CORRECTED.csv
"""
import csv
import os

import numpy as np
import klayout.db as db

PKG = r"G:\codex_project\inverse_design_sun\DBS_交接包_2026-10-11"
DBU = 0.0001
CELL = 0.120
N = 30
SUB = os.path.join(PKG, r"inputs\B_curve_DRC120")


def raster(path):
    lay = db.Layout()
    lay.read(path)
    top = lay.top_cell()
    idx = [li for li in range(lay.layers())
           if (lay.get_info(li).layer, lay.get_info(li).datatype) == (13, 1)][0]
    r = db.Region(top.begin_shapes_rec(idx))
    r.merge()
    M = np.zeros((N, N), dtype=bool)
    for rr in range(N):
        for cc in range(N):
            x = -1.8 + (cc + 0.5) * CELL
            y = -1.8 + (rr + 0.5) * CELL
            box = db.Box(int(round((x - 0.001) / DBU)), int(round((y - 0.001) / DBU)),
                         int(round((x + 0.001) / DBU)), int(round((y + 0.001) / DBU)))
            M[rr, cc] = not (r & db.Region(box)).is_empty()
    return M


st = np.load(os.path.join(SUB, "initial_state.npz"))
old_sil = st["silicon_state"]
mask = st["candidate_mask"]
rc = st["order_rc"]

G = raster(os.path.join(SUB, "initial.gds"))

# sanity: the GDS must agree with the candidate list on every candidate cell
rows = list(csv.DictReader(open(os.path.join(SUB, "candidates.csv"), newline="", encoding="utf-8")))
bad = sum(1 for r in rows
          if bool(G[int(r["row_y_up"]), int(r["column_x_right"])])
          != (r["initial_material"] == "silicon"))
print("GDS vs candidates.csv on the 108 candidates: %d mismatches" % bad)
assert bad == 0, "GDS and candidate list disagree -- do not trust the correction"

# and it must reproduce the old array on the candidate cells
diff_in = int((G[mask] != old_sil[mask]).sum())
print("corrected vs old, inside the candidate mask : %d differing cells" % diff_in)
print("corrected vs old, outside                    : %d differing cells"
      % int((G[~mask] != old_sil[~mask]).sum()))

np.savez(os.path.join(SUB, "initial_state_CORRECTED.npz"),
         candidate_mask=mask, silicon_state=G, order_rc=rc)

with open(os.path.join(SUB, "initial_state_CORRECTED.csv"), "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["row_y_up", "column_x_right", "x_centre_um", "y_centre_um",
                "in_candidate_mask", "material", "legacy_state_0_Si_1_oxide"])
    for rr in range(N):
        for cc in range(N):
            w.writerow([rr, cc, "%.2f" % (-1.8 + (cc + 0.5) * CELL),
                        "%.2f" % (-1.8 + (rr + 0.5) * CELL),
                        int(mask[rr, cc]), "silicon" if G[rr, cc] else "oxide",
                        0 if G[rr, cc] else 1])

print()
print("wrote initial_state_CORRECTED.npz  (silicon cells: %d -> %d)" % (old_sil.sum(), G.sum()))
print("wrote initial_state_CORRECTED.csv")
