"""Extract smooth continuous polygon skeleton (Scheme B) for Four-mode MUX.

Reads: parameters_80.npz from EGADO-4mode
Generates:
1. baseline_4mode_scheme_b.gds (Layer 13/1 Core, Layer 13/2 Design Region, Layer 251/16 Boundary)
2. DRC compliance report (width & space checks with 120 nm threshold)
3. High-resolution layout visualization images
"""

import os
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

NPZ_PATH = REPO_DIR / "parameters_80.npz"
GDS_OUT = REPO_DIR / "baseline_4mode_scheme_b.gds"

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
    print("Extracting Smooth Polygon Skeleton (Scheme B) for Four-mode MUX")
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
    for i, contour in enumerate(contours):
        xy = np.c_[x_fine[0] + contour[:, 1] * FINE_GRID_NM,
                   y_fine[0] + contour[:, 0] * FINE_GRID_NM]
        xy = resample_ring(xy, CHORD_STEP_NM)
        pts = [db.Point(u(px), u(py)) for px, py in xy]
        if len(pts) >= 3:
            poly = db.Polygon(pts)
            core_region ^= db.Region(poly)  # XOR handles islands and holes automatically

    core_region.merge()

    # Insert into cell
    top.shapes(layer_si).insert(core_region)

    # Insert boundary boxes
    design_box = db.Box(u(X_DESIGN_MIN), u(Y_DESIGN_MIN), u(X_DESIGN_MAX), u(Y_DESIGN_MAX))
    bound_box = db.Box(u(X_LEAD_IN), u(Y_PADDED_MIN), u(X_LEAD_OUT), u(Y_PADDED_MAX))
    top.shapes(layer_core).insert(design_box)
    top.shapes(layer_bound).insert(bound_box)

    layout.write(str(GDS_OUT))
    print(f"GDS written successfully to: {GDS_OUT}")

    # DRC Checks (120 nm threshold)
    print("\nRunning 120 nm DRC checks on smooth skeleton...")
    roi_design = db.Region(design_box)
    inner_core = core_region & roi_design

    t_120 = u(120.0)
    width_errs = inner_core.width_check(t_120, ignore_angle=85)
    space_errs = inner_core.space_check(t_120, ignore_angle=85)

    n_width_errs = width_errs.count()
    n_space_errs = space_errs.count()
    print("Core Design Region (6x6 um2) DRC Results:")
    print(f"  Width < 120 nm violations: {n_width_errs}")
    print(f"  Space < 120 nm violations: {n_space_errs}")

    # Extract polygon coordinates for plotting
    polys_to_plot = []
    for poly in core_region.each():
        hull = [np.array([p.x * DBU_UM, p.y * DBU_UM]) for p in poly.each_point_hull()]
        polys_to_plot.append(np.array(hull))
        for h in range(poly.holes()):
            hole = [np.array([p.x * DBU_UM, p.y * DBU_UM]) for p in poly.each_point_hole(h)]
            polys_to_plot.append(np.array(hole))

    # Width/Space error bounding boxes for annotation
    w_boxes = [db.Box(e.bbox()).transformed(db.ICplxTrans(DBU_UM, 0, False, 0, 0)) for e in width_errs.each()]
    s_boxes = [db.Box(e.bbox()).transformed(db.ICplxTrans(DBU_UM, 0, False, 0, 0)) for e in space_errs.each()]

    # Visualization
    plot_layout(polys_to_plot, w_boxes, s_boxes, n_width_errs, n_space_errs)

    return n_width_errs, n_space_errs

def plot_layout(polys, w_boxes, s_boxes, n_w, n_s):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300, gridspec_kw={'width_ratios': [1.25, 1]})

    # Subplot 1: Full Device Layout
    ax1.set_facecolor("#f8f9fa")
    for p in polys:
        patch = MplPoly(p, closed=True, facecolor="#1a5276", edgecolor="#113349", linewidth=0.8, alpha=0.92)
        ax1.add_patch(patch)

    # Core region box
    rect_core = Rectangle((-3.0, -3.0), 6.0, 6.0, linewidth=2.0, edgecolor="#c0392b", facecolor="none", linestyle="--", label="Core Design Region (6.0x6.0 um2)")
    ax1.add_patch(rect_core)

    # Annotate 4 Inputs
    for i, yc in enumerate([2.25, 0.75, -0.75, -2.25], 1):
        ax1.annotate(f"IN {i} (TE0)\nWidth 0.5 um", xy=(-3.0, yc), xytext=(-4.2, yc),
                     arrowprops=dict(facecolor="#d35400", edgecolor="#ba4a00", width=1.5, headwidth=6),
                     fontsize=9, fontweight="bold", ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.2", fc="#fbeee6", ec="#d35400", lw=1))

    # Annotate Output
    ax1.annotate("Multimode OUT\n(TE0 ~ TE3)\nWidth 1.8 um", xy=(3.0, 2.1), xytext=(4.2, 2.1),
                 arrowprops=dict(facecolor="#8e44ad", edgecolor="#6c3483", width=1.5, headwidth=6),
                 fontsize=9, fontweight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.2", fc="#f4ecf7", ec="#8e44ad", lw=1))

    ax1.set_xlim(-4.8, 4.8)
    ax1.set_ylim(-3.8, 3.8)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X (um)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Y (um)", fontsize=11, fontweight="bold")
    ax1.set_title("Scheme B: Four-mode MUX Smooth GDS Layout (Iter 80)\nLayer 13/1 (Silicon Core, 220 nm, n=3.48)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=10, framealpha=0.9)

    # Subplot 2: Core Zoom & 120 nm DRC Inspection
    ax2.set_facecolor("#f8f9fa")
    for p in polys:
        patch = MplPoly(p, closed=True, facecolor="#1a5276", edgecolor="#113349", linewidth=1.0, alpha=0.92)
        ax2.add_patch(patch)

    rect_core2 = Rectangle((-3.0, -3.0), 6.0, 6.0, linewidth=2.2, edgecolor="#c0392b", facecolor="none", linestyle="--")
    ax2.add_patch(rect_core2)

    # Overlay DRC error markers if any
    for b in w_boxes:
        rect_w = Rectangle((b.left, b.bottom), b.width(), b.height(), edgecolor="#e74c3c", facecolor="#e74c3c", alpha=0.7, lw=1.5)
        ax2.add_patch(rect_w)
    for b in s_boxes:
        rect_s = Rectangle((b.left, b.bottom), b.width(), b.height(), edgecolor="#f39c12", facecolor="#f39c12", alpha=0.7, lw=1.5)
        ax2.add_patch(rect_s)

    ax2.set_xlim(-3.2, 3.2)
    ax2.set_ylim(-3.2, 3.2)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X (um)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Y (um)", fontsize=11, fontweight="bold")
    ax2.set_title(f"Core Design Region (6.0x6.0 um2) Detail\n120 nm DRC Violations: Width={n_w}, Space={n_s}", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    local_img = REPO_DIR / "scheme_b_4mode_skeleton.png"
    artifact_img = ARTIFACT_DIR / "scheme_b_4mode_skeleton.png"
    plt.savefig(local_img, dpi=300)
    plt.savefig(artifact_img, dpi=300)
    plt.close()
    print(f"Layout plots saved to:\n  {local_img}\n  {artifact_img}")

if __name__ == '__main__':
    extract_and_generate_gds()
