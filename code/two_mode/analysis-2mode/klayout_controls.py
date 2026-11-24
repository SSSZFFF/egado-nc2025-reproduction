"""Positive/negative controls for the KLayout DRC setup.

Before trusting "0 violations" on the real pattern, prove that the same setup
DOES flag a genuine sub-MFS feature.  Builds known structures and checks that
width_check / space_check switch at exactly 120 nm.
"""
import klayout.db as db
import numpy as np

DBU = 0.001        # 1 nm
THR = 0.120        # 120 nm in um


def region_from_runs(rows, nx):
    """rows: list of (y_index, x_start, x_end) in mesh cells of 20 nm."""
    lay = db.Layout()
    lay.dbu = DBU
    cell = lay.create_cell("T")
    layer = lay.layer(1, 0)
    for (y, x0, x1) in rows:
        cell.shapes(layer).insert(db.Box(
            int(round(x0 * 0.02 / DBU)), int(round(y * 0.02 / DBU)),
            int(round(x1 * 0.02 / DBU)), int(round((y + 1) * 0.02 / DBU))))
    r = db.Region(cell.begin_shapes_rec(layer))
    r.merge()
    return r


def check(label, r):
    w = r.width_check(THR)
    s = r.space_check(THR)
    print("  %-34s width viol = %2d   space viol = %2d   %s"
          % (label, w.count(), s.count(),
             "FLAGGED" if (w.count() or s.count()) else "pass"))
    return w.count(), s.count()


print("=== control 1: silicon line of width w in vacuum ===")
for w_nm in (80, 100, 120, 140):
    k = w_nm // 20
    rows = [(y, 10, 10 + 60) for y in range(40, 40 + k)]
    check("Si line %3d nm (bar %d rows)" % (w_nm, k), region_from_runs(rows, 80))

print()
print("=== control 2: gap of width w between two silicon blocks ===")
for w_nm in (80, 100, 120, 140):
    k = w_nm // 20
    rows = [(y, 10, 40) for y in range(20, 60)]
    rows += [(y, 40 + k, 80) for y in range(20, 60)]
    check("gap %3d nm" % w_nm, region_from_runs(rows, 80))

print()
print("=== control 3: square island of side w ===")
for w_nm in (80, 100, 120, 140):
    k = w_nm // 20
    rows = [(y, 30, 30 + k) for y in range(30, 30 + k)]
    check("island %3d nm" % w_nm, region_from_runs(rows, 100))

print()
print("=== control 4: the real TO pattern, for reference ===")
z = np.load(r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_final.npz")
an = z["eps"].T[:-1, :-1]
an = -1 * np.flip(an, axis=0)
an = (an - an.min()) / (an.max() - an.min())
B = an < 0.5
ny, nx = B.shape
rows = []
for a in range(ny):
    b = 0
    while b < nx:
        if B[a, b]:
            b0 = b
            while b < nx and B[a, b]:
                b += 1
            rows.append((a, b0, b))
        else:
            b += 1
r = region_from_runs(rows, nx)
check("TO pattern (%d rects)" % len(rows), r)

print()
print("=== what the smallest axis-aligned silicon run is in that pattern ===")
runs = []
for a in range(ny):
    b = 0
    while b < nx:
        if B[a, b]:
            b0 = b
            while b < nx and B[a, b]:
                b += 1
            runs.append(b - b0)
        else:
            b += 1
runs = np.array(runs)
print("  horizontal Si runs: %d,  min %d cells (%.0f nm),  1st pct %d cells"
      % (len(runs), runs.min(), runs.min() * 20, int(np.percentile(runs, 1))))
print("  runs shorter than 6 cells (120 nm): %d (%.1f%%)"
      % (int((runs < 6).sum()), 100.0 * (runs < 6).mean()))
print("  -> a short run on ONE row is not a thin feature if the rows above/below")
print("     are solid; only KLayout's edge-to-edge test decides that correctly.")
