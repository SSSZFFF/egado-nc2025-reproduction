"""Curve extraction -> DRC -> local 120 nm homogeneous cells.

Start with the original-TO curve, not the previously pixel-repaired input.
Outside the selected cells the vector curve is retained. Material assignment
uses silicon area majority only; no optical optimisation is claimed.
"""

from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import re
import sys

sys.dont_write_bytecode = True
import numpy as np
import klayout.db as db

HERE = Path(__file__).resolve().parent
CURVES = HERE / "continuous_curve_trial"
OUT = HERE / "curve_then_grid_trial"
OUT.mkdir(exist_ok=True)
SOURCE_GDS = CURVES / "original_curve_sigma20.gds"
SOURCE_REPORT = CURVES / "check_original_curve_sigma20" / "result.json"
report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
assert report["gds_sha256"] == hashlib.sha256(SOURCE_GDS.read_bytes()).hexdigest().upper()
assert report["rule_categories"] == 248
layout = db.Layout()
layout.read(str(SOURCE_GDS))
DBU = layout.dbu
UNITS_PER_NM = 1 / (DBU * 1000)
top = layout.top_cell()
original = db.Region(top.begin_shapes_rec(layout.layer(13, 1)))
original.merge()
OTHER = {(layer, datatype): db.Region(top.begin_shapes_rec(layout.layer(layer, datatype)))
         for layer, datatype in ((13, 2), (251, 16))}
CELL_NM = 120
ORIGIN_NM = -1800
ROWS = COLS = 30
NUM = r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def scalar_units(nm):
    return int(round(nm * UNITS_PER_NM))


def source_items():
    for item in report["items"]:
        for value in item["values"]:
            points = np.array([(float(x) * 1000, float(y) * 1000)
                               for x, y in re.findall("(" + NUM + r")\s*,\s*(" + NUM + ")", value)])
            if len(points):
                minimum, maximum = points.min(axis=0), points.max(axis=0)
                yield item["category"], tuple(np.r_[minimum, maximum])


def extra_checks(region, target_nm):
    t = scalar_units(target_nm)
    width = region.width_check(t, ignore_angle=85)
    space = region.space_check(t, ignore_angle=85)
    return width, space


def supplemental_items(target_nm):
    for rule, edges in zip((f"MFS{target_nm}_WIDTH", f"MFS{target_nm}_SPACE"), extra_checks(original, target_nm)):
        for edge_pair in edges.each():
            bb = edge_pair.bbox()
            yield rule, tuple(np.array([bb.left, bb.bottom, bb.right, bb.top]) / UNITS_PER_NM)


def candidate_cells(items):
    mask = np.zeros((ROWS, COLS), dtype=bool)
    rule_masks = defaultdict(lambda: np.zeros_like(mask))
    for rule, (left, bottom, right, upper) in items:
        j0 = max(0, int(np.floor((left - ORIGIN_NM - 0.5) / CELL_NM)))
        j1 = min(COLS, int(np.ceil((right - ORIGIN_NM + 0.5) / CELL_NM)))
        i0 = max(0, int(np.floor((bottom - ORIGIN_NM - 0.5) / CELL_NM)))
        i1 = min(ROWS, int(np.ceil((upper - ORIGIN_NM + 0.5) / CELL_NM)))
        if i1 > i0 and j1 > j0:
            mask[i0:i1, j0:j1] = True
            rule_masks[rule][i0:i1, j0:j1] = True
    return mask, dict(rule_masks)


def box_for(row, col):
    x0, y0 = ORIGIN_NM + col * CELL_NM, ORIGIN_NM + row * CELL_NM
    return db.Box(scalar_units(x0), scalar_units(y0),
                  scalar_units(x0 + CELL_NM), scalar_units(y0 + CELL_NM))


manifest = {
    "input": "Continuous contour extracted from the original TO; no earlier 120 nm repair",
    "source_gds": str(SOURCE_GDS), "source_gds_sha256": sha(SOURCE_GDS),
    "source_actual_sp90_report": str(SOURCE_REPORT),
    "source_sp90_rule_categories": 248,
    "source_sp90_violations": report["violation_count"],
    "source_sp90_violation_counts": report["violation_counts"],
    "cell_nm": CELL_NM, "candidate_buffer_nm": 0,
    "boundary_tolerance_nm": 0.5,
    "material_assignment": "silicon area >= 50% -> full silicon; otherwise full oxide",
    "optical_simulations": 0,
    "cases": {},
}
for name, items in (("sp90_90nm", list(source_items())),
                    ("target_120nm", list(supplemental_items(120)))):
    mask, rule_masks = candidate_cells(items)
    rectangles = db.Region()
    silicon_cells = db.Region()
    numbering = np.zeros(mask.shape, dtype=int)
    silicon_state = np.zeros(mask.shape, dtype=bool)
    records = []
    indices = [(row, col) for row in range(ROWS - 1, -1, -1)
               for col in range(COLS) if mask[row, col]]
    for number, (row, col) in enumerate(indices, 1):
        box = box_for(row, col)
        cell = db.Region(box)
        fraction = (original & cell).area() / box.area()
        si = fraction >= 0.5
        rectangles.insert(box)
        if si:
            silicon_cells.insert(box)
        numbering[row, col] = number
        silicon_state[row, col] = si
        records.append({
            "id": number, "row_y_up": row, "column_x_right": col,
            "x_centre_um": (ORIGIN_NM + (col + 0.5) * CELL_NM) / 1000,
            "y_centre_um": (ORIGIN_NM + (row + 0.5) * CELL_NM) / 1000,
            "original_silicon_area_fraction": fraction,
            "chosen_material": "silicon" if si else "oxide",
            "selection_rules": ";".join(sorted(rule for rule, rm in rule_masks.items() if rm[row, col])),
        })
    rectangles.merge()
    silicon_cells.merge()
    filled = (original - rectangles) | silicon_cells
    filled.merge()
    difference = filled ^ original
    # Boolean intersections may move by one 0.1 nm DBU on vector boundaries.
    outside_change = (difference - rectangles).area() * DBU ** 2
    assert outside_change < 1e-5
    for row, col in indices:
        cell = db.Region(box_for(row, col))
        if silicon_state[row, col]:
            assert (cell - filled).is_empty()
        else:
            assert (cell & filled).is_empty()

    gds = OUT / f"{name}_filled.gds"
    if not gds.exists():
        output_layout = db.Layout()
        output_layout.dbu = DBU
        output_top = output_layout.create_cell(name.upper() + "_FILLED")
        output_top.shapes(output_layout.layer(13, 1)).insert(filled)
        for layer, r in OTHER.items():
            output_top.shapes(output_layout.layer(*layer)).insert(r)
        output_layout.write(str(gds))
    readback = db.Layout()
    readback.read(str(gds))
    actual = db.Region(readback.top_cell().begin_shapes_rec(readback.layer(13, 1)))
    actual.merge()
    serialization_difference = (actual ^ filled).area() * DBU ** 2
    # GDS hole fracture can round a curve/cut intersection by one DBU.
    assert serialization_difference < 1e-6
    for layer, r in OTHER.items():
        back = db.Region(readback.top_cell().begin_shapes_rec(readback.layer(*layer)))
        assert (back ^ r).is_empty()
    filled = actual
    difference = filled ^ original
    outside_change = (difference - rectangles).area() * DBU ** 2
    assert outside_change < 1e-5
    for row, col in indices:
        cell = db.Region(box_for(row, col))
        assert (cell - filled).is_empty() if silicon_state[row, col] else (cell & filled).is_empty()

    counts_before = {}
    counts_after = {}
    for target in (90, 120):
        before = extra_checks(original, target)
        after = extra_checks(filled, target)
        counts_before[str(target)] = {"width": before[0].count(), "space": before[1].count()}
        counts_after[str(target)] = {"width": after[0].count(), "space": after[1].count()}
    case = {
        "gds": str(gds), "gds_sha256": sha(gds), "candidate_cells": len(records),
        "selection_basis": "actual 248-category SP90 report" if name == "sp90_90nm" else "supplemental 120 nm width/space, 85 degree angle exclusion",
        "rule_specific_cell_counts": {rule: int(m.sum()) for rule, m in rule_masks.items()},
        "filled_silicon_cells": int(silicon_state[mask].sum()),
        "filled_oxide_cells": int((~silicon_state[mask]).sum()),
        "selected_area_percent": 100 * len(records) / 900,
        "changed_area_um2_from_original_curve": difference.area() * DBU ** 2,
        "changed_area_percent_of_design": difference.area() * DBU ** 2 / 12.96 * 100,
        "outside_candidate_area_difference_um2": outside_change,
        "selected_cells_are_homogeneous": True,
        "gds_readback_difference_area_um2": serialization_difference,
        "gds_readback_within_0p1nm_boolean_quantization_tolerance": True,
        "supplemental_before": counts_before, "supplemental_after": counts_after,
    }
    manifest["cases"][name] = case
    np.savez_compressed(OUT / f"{name}_cells.npz", candidate_mask=mask,
                        numbering=numbering, silicon_state=silicon_state,
                        **{f"mask_{rule}": m for rule, m in rule_masks.items()})
    with (OUT / f"{name}_cells.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(name, case, flush=True)

(OUT / "grid_parameters.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved trial GDS, masks and cell records")
