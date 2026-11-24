"""Extract smooth continuous polygon skeleton (Scheme B) for Four-mode MUX from Checkpoint 153 (Iter 266).

Reads: parameters_run3_153.npz (beta=38.34, discreteness=92.7%, binary area=91.8%)
Generates:
1. baseline_4mode_scheme_b_iter266.gds (Layer 13/1 Core, Layer 13/2 Design Region, Layer 251/16 Boundary)
2. DRC compliance report (width & space checks with 120 nm threshold)
3. High-resolution layout visualization images
"""

import os
import shutil
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RectBivariateSpline
from skimage.measure import find_contours
import klayout.db as db
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Rectangle

# Directories
REPO_DIR = Path(r"G:\codex_project\inverse_design_sun\EGADO-4mode")
ARTIFACT_DIR = Path(r"C:\Users\USER\.gemini\antigravity\brain\39ad5e21-1313-4872-992d-8dc8c32f9b61")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

NPZ_PATH = REPO_DIR / "parameters_run3_153.npz"
GDS_OUT = REPO_DIR / "baseline_4mode_scheme_b_iter266.gds"

# Physical Parameters (in nm)
GRID_NM = 20.0
FINE_GRID_NM = 2.5
CHORD_STEP_NM = 10.0
DBU_UM = 0.001  # 1 nm database unit
UNITS_PER_NM = 1.0 / (DBU_UM * 1000.0)

# Coordinate Ranges (in nm)
X_DESIGN_MIN, X_DESIGN_MAX = -3000.0, 3000.0
Y_DESIGN_MIN, Y_DESIGN_MAX = -3000.0, 3000.0

# Waveguide leads end at +/- 4500 nm, full padded field extends to +/- 5000 nm
X_LEAD_IN = -4500.0
X_LEAD_OUT = 4500.0

X_PADDED_MIN, X_PADDED_MAX = -5000.0, 5000.0
Y_PADDED_MIN, Y_PADDED_MAX = -4000.0, 4000.0

# 4 Single-mode Input Waveguides (width 500 nm, spacing 1000 nm, pitch 1500 nm)
IN_CENTRES_Y = [2250.0, 750.0, -750.0, -2250.0]
IN_WIDTH = 500.0

# 1 Multimode Output Waveguide (width 1800 nm, top aligned [1200, 3000])
OUT_Y_MIN = 1200.0
OUT_Y_MAX = 3000.0

def u(nm):
    return int(round(nm * UNITS_PER_NM))

def resample_ring(xy, step_nm):
    """Resample polygon vertices along equal arclength to avoid collinear/degenerate points."""
    if np.linalg.norm(xy[0] - xy[-1]) > 1e-4:
        raise ValueError(f"Open contour detected at field boundary! ({xy[0]} to {xy[-1]})")
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.r_[True, ds > 1e-6]
    xy = xy[keep]
    if len(xy) < 4:
        return xy
    length = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    total_len = length[-1]
    if total_len < 1e-4:
        return xy
    number = max(8, int(np.ceil(total_len / step_nm)))
    samples = np.linspace(0, total_len, number, endpoint=False)
    out = np.c_[np.interp(samples, length, xy[:, 0]), np.interp(samples, length, xy[:, 1])]
    return out

def build_full_field(rho_design):
    """Embed 6x6 um design region into extended field with padded boundaries to guarantee closed contours."""
    xs = np.arange(X_PADDED_MIN, X_PADDED_MAX + 0.1, GRID_NM)
    ys = np.arange(Y_PADDED_MIN, Y_PADDED_MAX + 0.1, GRID_NM)
    xx, yy = np.meshgrid(xs, ys)
    field = np.zeros(xx.shape, dtype=float)

    # 1. Input waveguide leads: x in [-4500, -3000]
    for yc in IN_CENTRES_Y:
        in_mask = (xx >= X_LEAD_IN) & (xx <= X_DESIGN_MIN) & (yy >= yc - IN_WIDTH / 2.0) & (yy <= yc + IN_WIDTH / 2.0)
        field[in_mask] = 1.0

    # 2. Output waveguide lead: x in [3000, 4500], y in [1200, 3000]
    out_mask = (xx >= X_DESIGN_MAX) & (xx <= X_LEAD_OUT) & (yy >= OUT_Y_MIN) & (yy <= OUT_Y_MAX)
    field[out_mask] = 1.0

    # 3. Core design region: x in [-3000, 3000], y in [-3000, 3000]
    ix0 = int(round((X_DESIGN_MIN - X_PADDED_MIN) / GRID_NM))
    iy0 = int(round((Y_DESIGN_MIN - Y_PADDED_MIN) / GRID_NM))
    nr, nc = rho_design.shape
    field[iy0:iy0 + nr, ix0:ix0 + nc] = rho_design

    return xs, ys, field

def extract_and_generate_gds():
    print("=" * 72)
    print("Extracting Smooth Polygon Skeleton (Scheme B) for Four-mode MUX @ Iter 266 (Ckpt 153)")
    print(f"Loading checkpoint: {NPZ_PATH}")
    data = np.load(NPZ_PATH)
    eps = data['eps']
    eps_min = float(data['eps_min'])
    eps_max = float(data['eps_max'])
    rho_phys = (eps - eps_min) / (eps_max - eps_min)
    rho_phys = np.clip(rho_phys.T, 0.0, 1.0)  # shape (301, 301): y, x

    # Build full field with padded leads
    xs, ys, field = build_full_field(rho_phys)

    # Gaussian smoothing with sigma = 20 nm
    sigma_nm = 20.0
    field_smooth = gaussian_filter(field, sigma=sigma_nm / GRID_NM, mode='constant', cval=0.0)

    # Spline interpolation to fine grid
    print("Interpolating field to fine 2.5 nm grid...")
    x_fine = np.arange(xs[0], xs[-1] + 0.01, FINE_GRID_NM)
    y_fine = np.arange(ys[0], ys[-1] + 0.01, FINE_GRID_NM)
    fit = RectBivariateSpline(ys, xs, field_smooth, kx=3, ky=3, s=0)
    fine_field = fit(y_fine, x_fine)

    # Find contours at level = 0.5
    print("Extracting level-set contours (rho = 0.5)...")
    contours = find_contours(fine_field, 0.5, fully_connected='low', positive_orientation='low')
    print(f"Found {len(contours)} contours.")

    # Convert to KLayout Region
    layout = db.Layout()
    layout.dbu = DBU_UM
    top = layout.create_cell("TOP")
    layer_si = layout.layer(13, 1)        # Silicon core
    layer_core = layout.layer(13, 2)      # Design region
    layer_bound = layout.layer(251, 16)   # Boundary

    core_region = db.Region()
    extracted_polygons = []

    for c in contours:
        # Convert index coordinates to physical nm
        c_nm = np.c_[
            np.interp(c[:, 1], np.arange(len(x_fine)), x_fine),
            np.interp(c[:, 0], np.arange(len(y_fine)), y_fine)
        ]
        # Resample
        c_res = resample_ring(c_nm, step_nm=CHORD_STEP_NM)
        pts = [db.Point(u(p[0]), u(p[1])) for p in c_res]
        poly = db.Polygon(pts)
        core_region.insert(poly)
        extracted_polygons.append(c_res)

    # Merge overlapping/adjacent polygons using even-odd boolean
    print("Merging polygons in KLayout...")
    core_region.merge()

    # Crop to design boundary leads [-4500, 4500] x [-3500, 3500]
    lead_box = db.Box(u(-4500.0), u(-3500.0), u(4500.0), u(3500.0))
    core_region &= db.Region(lead_box)

    top.shapes(layer_si).insert(core_region)

    # Design region marker (Layer 13/2)
    design_box = db.Box(u(X_DESIGN_MIN), u(Y_DESIGN_MIN), u(X_DESIGN_MAX), u(Y_DESIGN_MAX))
    top.shapes(layer_core).insert(design_box)

    # Boundary marker (Layer 251/16)
    bound_box = db.Box(u(X_PADDED_MIN), u(Y_PADDED_MIN), u(X_PADDED_MAX), u(Y_PADDED_MAX))
    top.shapes(layer_bound).insert(bound_box)

    layout.write(str(GDS_OUT))
    print(f"Saved GDS to: {GDS_OUT}")

    # DRC Assessment
    print("\n--- DRC Compliance Assessment (Threshold = 120 nm) ---")
    drc_roi = db.Box(u(X_DESIGN_MIN), u(Y_DESIGN_MIN), u(X_DESIGN_MAX), u(Y_DESIGN_MAX))
    drc_region = core_region & db.Region(drc_roi)
    t120 = u(120.0)

    w_err = drc_region.width_check(t120, ignore_angle=85)
    s_err = drc_region.space_check(t120, ignore_angle=85)

    print(f"Min Width (<120 nm) error count: {w_err.count()}")
    print(f"Min Space (<120 nm) error count: {s_err.count()}")
    print(f"Total DRC Violations: {w_err.count() + s_err.count()}")

    # Plot Layout Visualization
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Plot design area boundary
    ax.add_patch(Rectangle((X_DESIGN_MIN/1000, Y_DESIGN_MIN/1000), 
                           (X_DESIGN_MAX-X_DESIGN_MIN)/1000, 
                           (Y_DESIGN_MAX-Y_DESIGN_MIN)/1000, 
                           fill=False, edgecolor='blue', linestyle='--', linewidth=1.5, label='Design Region (6x6 μm)'))

    # Plot polygons from KLayout merged region
    num_polys = 0
    for poly in core_region.each():
        hull = np.array([[pt.x * DBU_UM, pt.y * DBU_UM] for pt in poly.each_point_hull()])
        ax.add_patch(MplPoly(hull, facecolor='#2b5c8f', edgecolor='black', linewidth=0.8, alpha=0.9))
        for h in range(poly.holes()):
            hole_pts = np.array([[pt.x * DBU_UM, pt.y * DBU_UM] for pt in poly.each_point_hole(h)])
            ax.add_patch(MplPoly(hole_pts, facecolor='white', edgecolor='black', linewidth=0.8))
        num_polys += 1

    # Port indicators
    for i, yc in enumerate(IN_CENTRES_Y):
        ax.plot([-3.0, -4.5], [yc/1000, yc/1000], color='crimson', linewidth=3)
        ax.text(-4.6, yc/1000, f'IN{i+1} (TE0)', color='crimson', fontsize=9, fontweight='bold', ha='right', va='center')

    ax.plot([3.0, 4.5], [2.1, 2.1], color='cyan', linewidth=8, alpha=0.6)
    ax.text(4.6, 2.1, 'OUT (1.8μm)\nTE0-TE3', color='navy', fontsize=9, fontweight='bold', ha='left', va='center')

    ax.set_xlim(-5.2, 5.5)
    ax.set_ylim(-3.8, 3.8)
    ax.set_xlabel('x (μm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('y (μm)', fontsize=11, fontweight='bold')
    ax.set_title(f'Four-mode MUX: Scheme B Smooth Polygon Skeleton @ Iter 266\n(β=38.34, Discreteness=92.7%, Binary Area=91.8%, Total Polygons={num_polys})', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.5)

    img_local = REPO_DIR / "scheme_b_iter266_skeleton.png"
    plt.savefig(img_local, dpi=300)
    plt.close()
    shutil.copy(img_local, ARTIFACT_DIR / "scheme_b_iter266_skeleton.png")
    print(f"Saved layout image to: {img_local}")

if __name__ == "__main__":
    extract_and_generate_gds()
