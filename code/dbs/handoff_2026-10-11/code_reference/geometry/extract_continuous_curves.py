"""Compare staircase polygons with interpolated and smoothed level-set contours.

All variants preserve the same drawn-layer mapping and CLD/DEV envelopes.
No optical simulation.  Gaussian smoothing changes geometry and is recorded.
"""

from pathlib import Path
import hashlib
import json
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import klayout.db as db
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from skimage.measure import find_contours

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "local_coarsening_trial" / "patterns.npz"
BASE = HERE / "wuyue_sp90_checks"
OUT = HERE / "continuous_curve_trial"
OUT.mkdir(exist_ok=True)
data = np.load(SOURCE, allow_pickle=False)
old_meta = json.loads((BASE / "layout_parameters.json").read_text(encoding="utf-8"))
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest().upper() == old_meta["source_sha256"]

DBU_UM = 0.0001  # 0.1 nm coordinate quantisation for fitted curves.
UNITS_PER_NM = 1.0 / (DBU_UM * 1000)
GRID_NM = 20.0
FINE_GRID_NM = 2.5
CHORD_STEP_NM = 10.0  # Do not rely on sub-6 nm edges escaping WGF03.
XS = np.arange(-2990.0, 3000.0, GRID_NM)
YS = np.arange(-2190.0, 2200.0, GRID_NM)
X_FINE = np.arange(XS[0], XS[-1] + 0.01, FINE_GRID_NM)
Y_FINE = np.arange(YS[0], YS[-1] + 0.01, FINE_GRID_NM)
T0 = time.perf_counter()


def u(nm):
    return int(round(nm * UNITS_PER_NM))


def log(message):
    print(f"[{time.perf_counter() - T0:6.2f}s] {message}", flush=True)


def full_field(array):
    xx, yy = np.meshgrid(XS, YS)
    field = np.zeros(xx.shape, dtype=float)
    for centre in (-650, 650):
        lead = ((xx >= -2800) & (xx < -1800) & (yy >= centre - 250) & (yy < centre + 250))
        field[lead] = 1.0
    lead = (xx >= 1800) & (xx < 2800) & (yy >= 900) & (yy < 1800)
    field[lead] = 1.0
    x0 = int(round((-1790 - XS[0]) / GRID_NM))
    y0 = int(round((-1790 - YS[0]) / GRID_NM))
    field[y0:y0 + 180, x0:x0 + 180] = array
    return field


def resample_ring(xy, step_nm):
    # Equal arclength spacing avoids near-coincident marching-squares points.
    if np.linalg.norm(xy[0] - xy[-1]) > 1e-6:
        raise ValueError("Open contour at field boundary")
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.r_[True, ds > 1e-7]
    xy = xy[keep]
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    number = max(8, int(np.ceil(length[-1] / step_nm)))
    samples = np.linspace(0, length[-1], number, endpoint=False)
    out = np.c_[np.interp(samples, length, xy[:, 0]), np.interp(samples, length, xy[:, 1])]
    return out


def load_reference(name):
    layout = db.Layout()
    layout.read(str(BASE / f"{name}.gds"))
    top = layout.top_cell()
    core = db.Region(top.begin_shapes_rec(layout.layer(13, 1)))
    core.merge()
    scale = layout.dbu / DBU_UM
    core = core.transformed(db.ICplxTrans(scale, 0, False, 0, 0))
    other = {}
    for key in ((13, 2), (251, 16)):
        r = db.Region(top.begin_shapes_rec(layout.layer(*key)))
        other[key] = r.transformed(db.ICplxTrans(scale, 0, False, 0, 0))
    return core, other


def export(case, array, sigma_nm, cubic, reference_case):
    field = full_field(array)
    if sigma_nm:
        field = gaussian_filter(field, sigma_nm / GRID_NM, mode="constant", cval=0.0)
    if cubic:
        fit = RectBivariateSpline(YS, XS, field, kx=3, ky=3, s=0)
        fine = fit(Y_FINE, X_FINE)
    else:
        fit = RectBivariateSpline(YS, XS, field, kx=1, ky=1, s=0)
        fine = fit(Y_FINE, X_FINE)
    contours = find_contours(fine, 0.5, fully_connected="low", positive_orientation="low")
    region = db.Region()
    ring_stats = []
    for contour in contours:
        xy = np.c_[XS[0] + contour[:, 1] * FINE_GRID_NM,
                   YS[0] + contour[:, 0] * FINE_GRID_NM]
        xy = resample_ring(xy, CHORD_STEP_NM)
        points = np.rint(xy * UNITS_PER_NM).astype(np.int64)
        keep = np.r_[True, np.any(np.diff(points, axis=0), axis=1)]
        points = points[keep]
        if len(points) < 3:
            continue
        polygon = db.Polygon([db.Point(int(x), int(y)) for x, y in points])
        # XOR expresses nesting (holes / islands) without guessing orientation.
        region ^= db.Region(polygon)
        ring_stats.append({"vertices": len(points), "area_um2": polygon.area() * DBU_UM ** 2})
    region.merge()
    reference, other = load_reference(reference_case)
    roi = db.Region(db.Box(u(-2400), u(-2100), u(2400), u(2100)))
    # Keep the finite lead ends exactly as before.  The contour meets the cut on
    # straight waveguide segments, 600 nm away from the variable design region.
    region = (region & roi) | (reference - roi)
    region.merge()
    path = OUT / f"{case}.gds"
    if path.exists():
        raise FileExistsError(path)
    layout = db.Layout()
    layout.dbu = DBU_UM
    top = layout.create_cell(case.upper())
    top.shapes(layout.layer(13, 1)).insert(region)
    for (layer, datatype), shape in other.items():
        top.shapes(layout.layer(layer, datatype)).insert(shape)
    layout.write(str(path))
    check = db.Layout()
    check.read(str(path))
    actual = db.Region(check.top_cell().begin_shapes_rec(check.layer(13, 1)))
    actual.merge()
    assert (actual ^ region).is_empty()
    assert ((region ^ reference) - roi).is_empty()
    changed = (region ^ reference).area() * DBU_UM ** 2
    parameters = {
        "gds": str(path), "gds_sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        "reference_case": reference_case,
        "reference_gds_sha256": old_meta["cases"][reference_case]["gds_sha256"],
        "array_source": str(SOURCE), "array_key": "rho" if reference_case == "before_20nm" else "local_coarsened",
        "isovalue": 0.5, "gaussian_sigma_nm": sigma_nm,
        "interpolation": "bicubic" if cubic else "bilinear",
        "extraction_grid_nm": FINE_GRID_NM, "contour_chord_sampling_nm": CHORD_STEP_NM,
        "dbu_um": DBU_UM, "ring_stats": ring_stats,
        "silicon_area_um2": region.area() * DBU_UM ** 2,
        "symmetric_difference_area_um2": changed,
        "changed_area_percent_of_3p6um_design": changed / (3.6 * 3.6) * 100,
        "fixed_lead_ends_unchanged": True,
        "cld_and_dev_same_as_reference": True,
        "gds_readback_xor_empty": True,
        "optical_simulations": 0,
    }
    log(f"{case}: {len(contours)} contours, changed area {changed:.5f} um2 ({parameters['changed_area_percent_of_3p6um_design']:.3f}%)")
    return parameters


manifest = {"source_arrays": str(SOURCE), "source_sha256": old_meta["source_sha256"],
            "method": "0.5 level set of bilinear / Gaussian-filtered bicubic scalar field; equal-arclength vector contour",
            "ports": old_meta["ports"], "cases": {}}
manifest_path = OUT / "curve_parameters.json"
if manifest_path.exists():
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert previous["source_sha256"] == manifest["source_sha256"]
    manifest["cases"].update(previous["cases"])
variants = (
    ("after_linear", data["local_coarsened"].astype(float), 0.0, False, "after_local120"),
    ("after_curve_sigma10", data["local_coarsened"].astype(float), 10.0, True, "after_local120"),
    ("after_curve_sigma20", data["local_coarsened"].astype(float), 20.0, True, "after_local120"),
    ("original_curve_sigma20", data["rho"], 20.0, True, "before_20nm"),
    ("after_curve_sigma15", data["local_coarsened"].astype(float), 15.0, True, "after_local120"),
)
for case, array, sigma, cubic, reference in variants:
    if case in manifest["cases"]:
        existing = manifest["cases"][case]
        assert hashlib.sha256(Path(existing["gds"]).read_bytes()).hexdigest().upper() == existing["gds_sha256"]
        assert existing["gaussian_sigma_nm"] == sigma and existing["reference_case"] == reference
        continue
    manifest["cases"][case] = export(case, array, sigma, cubic, reference)
(OUT / "curve_parameters.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
log("Saved curve inputs and geometry parameters")
