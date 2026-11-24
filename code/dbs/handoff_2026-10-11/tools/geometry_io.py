"""Portable geometry preparation and checks; this module never runs FDTD.

All states use rows increasing in +y, columns increasing in +x, True = Si.
Only cells in candidate_mask may change. Preserve vector geometry elsewhere.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import klayout.db as db
import numpy as np

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DBU_UM = 0.0001
PITCH_UM = 0.120
ORIGIN_UM = -1.8
GRID_SHAPE = (30, 30)
CORE = (13, 1)
ENVELOPES = ((13, 2), (251, 16))
AREA_TOL_UM2 = 1e-5  # 0.1 nm boolean / GDS serialization tolerance
SCHEMES = {
    "A": "inputs/A_D0",
    "B": "inputs/B_curve_DRC120",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_gds(path: Path):
    layout = db.Layout()
    layout.read(str(path))
    if abs(layout.dbu - DBU_UM) > 1e-12:
        raise ValueError(f"Expected dbu {DBU_UM} um, got {layout.dbu}: {path}")
    tops = list(layout.top_cells())
    if len(tops) != 1:
        raise ValueError(f"Expected one top cell, got {len(tops)}: {path}")
    regions = {}
    for key in (CORE,) + ENVELOPES:
        idx = layout.find_layer(*key)
        source = db.Region() if idx is None else db.Region(tops[0].begin_shapes_rec(idx))
        # Detach polygons explicitly. Single-shape Regions can retain a view of
        # layout-owned data; the caller may release the Layout before export.
        region = db.Region()
        for polygon in source.each():
            region.insert(polygon.dup())
        region.merge()
        regions[key] = region
    return layout, tops[0], regions


def cell_box(row: int, col: int):
    x0 = ORIGIN_UM + col * PITCH_UM
    y0 = ORIGIN_UM + row * PITCH_UM
    coords = [x0, y0, x0 + PITCH_UM, y0 + PITCH_UM]
    return db.Box(*[int(round(v / DBU_UM)) for v in coords])


def selected_region(mask):
    result = db.Region()
    for row, col in np.argwhere(mask):
        result.insert(cell_box(int(row), int(col)))
    result.merge()
    return result


def binary_array(value, label: str):
    array = np.asarray(value)
    if array.shape != GRID_SHAPE or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{label} must be a binary 30 x 30 array; got {array.shape}, {array.dtype}")
    return array.astype(bool)


def inputs(root: Path, scheme: str):
    folder = root / SCHEMES[scheme]
    with np.load(folder / "initial_state.npz", allow_pickle=False) as archive:
        mask = binary_array(archive["candidate_mask"], "candidate_mask")
        initial = binary_array(archive["silicon_state"], "silicon_state")
        order_rc = np.array(archive["order_rc"], dtype=int)
    count = 238 if scheme == "A" else 108
    if int(mask.sum()) != count:
        raise ValueError(f"Scheme {scheme}: expected {count} candidates")
    expected_order = np.array(sorted(map(tuple, np.argwhere(mask)), key=lambda rc: (rc[1], -rc[0])))
    if not np.array_equal(order_rc, expected_order):
        raise ValueError("order_rc differs from column-by-column, top-to-bottom scan")
    return folder, mask, initial, order_rc


def read_states(root: Path, scheme: str, path: Path | None):
    folder, mask, initial, order = inputs(root, scheme)
    states = initial.copy()
    if path:
        with np.load(path, allow_pickle=False) as data:
            states = binary_array(data["silicon_state"], "silicon_state")
            if "candidate_mask" in data and not np.array_equal(binary_array(data["candidate_mask"], "candidate_mask"), mask):
                raise ValueError("The run changed the frozen candidate_mask")
        if not np.array_equal(states[~mask], initial[~mask]):
            raise ValueError("A non-candidate state changed; this is not the agreed masked DBS")
    return folder, mask, initial, states, order


def write_gds(path: Path, name: str, core, windows):
    path.parent.mkdir(parents=True, exist_ok=True)
    layout = db.Layout()
    layout.dbu = DBU_UM
    top = layout.create_cell(name)
    # GDS has no native holes: KLayout writes a hole-aware polygon decomposition.
    # Read back every file and check its actual boolean region below.
    for key, region in [(CORE, core)] + [(key, windows[key]) for key in ENVELOPES]:
        top.shapes(layout.layer(*key)).insert(region)
    layout.write(str(path))
    _, _, loaded = read_gds(path)
    if (loaded[CORE] ^ core).area() * DBU_UM**2 > AREA_TOL_UM2:
        raise ValueError(f"GDS silicon readback mismatch: {path}")
    for key in ENVELOPES:
        if not (loaded[key] ^ windows[key]).is_empty():
            raise ValueError(f"GDS envelope readback mismatch: {key}")


def export_geometry(root: Path, scheme: str, states_path: Path | None, output: Path):
    folder, mask, _, states, _ = read_states(root, scheme, states_path)
    _, _, original = read_gds(folder / "initial.gds")
    _, _, fixed = read_gds(folder / "fixed_background.gds")
    active = selected_region(mask)
    if not (fixed[CORE] & active).is_empty():
        raise ValueError("Fixed background must exclude every candidate box")
    core = fixed[CORE].dup()
    for row, col in np.argwhere(mask & states):
        core.insert(cell_box(int(row), int(col)))
    core.merge()
    write_gds(output, f"{scheme}_FULL_DEVICE", core, original)
    return check_geometry(root, scheme, states_path, output)


def check_geometry(root: Path, scheme: str, states_path: Path | None, gds_path: Path):
    folder, mask, _, states, _ = read_states(root, scheme, states_path)
    _, _, original = read_gds(folder / "initial.gds")
    _, top, actual = read_gds(gds_path)
    for key in (CORE,) + ENVELOPES:
        if original[key].is_empty() or actual[key].is_empty():
            raise ValueError(f"Missing / empty required drawn layer: {key}")
    active = selected_region(mask)
    outside_difference = ((actual[CORE] ^ original[CORE]) - active).area() * DBU_UM**2
    if outside_difference > AREA_TOL_UM2:
        raise ValueError(f"Fixed geometry changed outside candidates: {outside_difference} um2")
    bad_cells = []
    for row, col in np.argwhere(mask):
        box = db.Region(cell_box(int(row), int(col)))
        desired = box if states[row, col] else db.Region()
        error_area = ((actual[CORE] & box) ^ desired).area() * DBU_UM**2
        # A full cell area is 0.0144 um2; a partial cell must fail.
        if error_area > AREA_TOL_UM2:
            bad_cells.append([int(row), int(col), error_area])
    if bad_cells:
        raise ValueError(f"Selected cells not homogeneous / state mismatch: {bad_cells[:8]}")
    for key in ENVELOPES:
        if not (actual[key] ^ original[key]).is_empty():
            raise ValueError(f"Processing layer {key} changed")
    return {
        "scheme": scheme, "candidate_cells": int(mask.sum()),
        "candidate_silicon_cells": int(states[mask].sum()),
        "candidate_oxide_cells": int((~states[mask]).sum()),
        "outside_candidate_difference_um2": outside_difference,
        "selected_cells_homogeneous": True, "processing_layers_unchanged": True,
        "gds": str(gds_path.resolve()), "gds_sha256": digest(gds_path),
        "top_cell": top.name, "dbu_um": DBU_UM,
        "optical_simulations": 0,
    }


def supplemental(gds_path: Path, threshold_nm: float):
    if not np.isfinite(threshold_nm) or threshold_nm <= 0:
        raise ValueError("threshold-nm must be positive and finite")
    _, _, regions = read_gds(gds_path)
    threshold_dbu = int(round((threshold_nm / 1000.0) / DBU_UM))
    core = regions[CORE]
    return {
        "gds": str(gds_path.resolve()), "gds_sha256": digest(gds_path),
        "dbu_um": DBU_UM, "threshold_nm": threshold_nm,
        "threshold_dbu": threshold_dbu, "ignore_angle_deg": 85,
        "width": core.width_check(threshold_dbu, ignore_angle=85).count(),
        "space": core.space_check(threshold_dbu, ignore_angle=85).count(),
        "scope": "Supplemental Si width and space only; not a full foundry rule deck",
    }


def verify(root: Path):
    checksum_path = root / "checksums.sha256"
    file_count = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or digest(path) != expected.upper():
            raise ValueError(f"Missing or changed handoff file: {relative}")
        file_count += 1
    checkpoint = root / "inputs/common/2mode_final.npz"
    if digest(checkpoint) != "8591BCF9612DAE081D454161C435289E02D7DEED6FAFF793B8509FC600DFF36D":
        raise ValueError("Not the agreed TO checkpoint")
    with np.load(checkpoint, allow_pickle=False) as data:
        if data["eps"].shape != (181, 181):
            raise ValueError("Wrong TO shape")
    result = {"verified_files": file_count, "optical_simulations": 0, "schemes": {}}
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    masks = {}
    for scheme in SCHEMES:
        folder, masks[scheme], _, _ = inputs(root, scheme)
        result["schemes"][scheme] = check_geometry(root, scheme, None, folder / "initial.gds")
        result["schemes"][scheme]["supplemental_120nm"] = supplemental(folder / "initial.gds", 120)
        report_folder = "A_initial_official_DRC" if scheme == "A" else "B_portable_initial_official_DRC"
        report = json.loads((root / "evidence" / report_folder / "result.json").read_text(encoding="utf-8"))
        expected_report = manifest[f"{scheme}_initial_official_DRC"]
        if report["gds_sha256"] != digest(folder / "initial.gds"):
            raise ValueError(f"{scheme}: archived DRC report belongs to a different GDS")
        for key in ("rule_categories", "violation_count", "violation_counts", "source_runset_sha256"):
            if report[key] != expected_report[key]:
                raise ValueError(f"{scheme}: archived DRC summary differs from manifest: {key}")
        if report["rule_categories"] != 248:
            raise ValueError("Archived report is not the agreed full common deck")
        result["schemes"][scheme]["archived_official_DRC"] = expected_report
        _, _, initial_regions = read_gds(folder / "initial.gds")
        _, _, fixed_regions = read_gds(folder / "fixed_background.gds")
        expected_fixed = initial_regions[CORE] - selected_region(masks[scheme])
        if (expected_fixed ^ fixed_regions[CORE]).area() * DBU_UM**2 > AREA_TOL_UM2:
            raise ValueError(f"{scheme}: fixed_background does not match initial.gds")
        with (folder / "candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        listed = {(int(row["row_y_up"]), int(row["column_x_right"])) for row in rows}
        if listed != set(map(tuple, np.argwhere(masks[scheme]))) or len(rows) != int(masks[scheme].sum()):
            raise ValueError(f"{scheme}: candidates.csv differs from candidate_mask")
        for row in rows:
            r, c = int(row["row_y_up"]), int(row["column_x_right"])
            if int(row["legacy_n"]) != (29 - r) * 30 + c:
                raise ValueError("Legacy pixel index error")
            if abs(float(row["x_centre_um"]) - (ORIGIN_UM+(c+0.5)*PITCH_UM)) > 1e-10:
                raise ValueError("x coordinate error")
            if abs(float(row["y_centre_um"]) - (ORIGIN_UM+(r+0.5)*PITCH_UM)) > 1e-10:
                raise ValueError("y coordinate error")
    result["mask_overlap"] = {
        "shared": int((masks["A"] & masks["B"]).sum()),
        "A_only": int((masks["A"] & ~masks["B"]).sum()),
        "B_only": int((masks["B"] & ~masks["A"]).sum()),
    }
    return result


def regenerate_d0(root: Path):
    from skimage.feature import canny

    with np.load(root / "inputs/common/2mode_final.npz", allow_pickle=False) as source:
        mapped = -np.flipud(source["eps"].T[:-1, :-1])
    mapped = (mapped - mapped.min()) / (mapped.max() - mapped.min())
    # Reproduce the original Canny defaults and 6x6 ANY pooling, D=0.
    top_down = canny(mapped).reshape(30, 6, 30, 6).any(axis=(1, 3))
    bottom_up = np.flipud(top_down)
    _, mask, _, _ = inputs(root, "A")
    equal = np.array_equal(bottom_up, mask)
    return {"generated_candidates": int(bottom_up.sum()), "equals_saved_D0_mask": equal,
            "note": "Use saved masks for the main run; a different skimage version can change Canny"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    sub.add_parser("regenerate-d0")
    for command in ("export", "check"):
        child = sub.add_parser(command)
        child.add_argument("--scheme", choices=SCHEMES, required=True)
        child.add_argument("--states", type=Path)
        child.add_argument("--gds", type=Path, required=True)
    child = sub.add_parser("supplemental")
    child.add_argument("--gds", type=Path, required=True)
    child.add_argument("--threshold-nm", type=float, default=120)
    child.add_argument("--json", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "verify":
        result = verify(root)
    elif args.command == "regenerate-d0":
        result = regenerate_d0(root)
        if not result["equals_saved_D0_mask"]:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1
    elif args.command == "export":
        result = export_geometry(root, args.scheme, args.states, args.gds)
    elif args.command == "check":
        result = check_geometry(root, args.scheme, args.states, args.gds)
    else:
        result = supplemental(args.gds, args.threshold_nm)
        if args.json:
            write_json(args.json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if args.command == "supplemental" and (result["width"] or result["space"]) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
