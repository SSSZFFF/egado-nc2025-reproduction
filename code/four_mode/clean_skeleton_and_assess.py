"""Clean small holes and small islands from the smooth skeleton.

Investigates area thresholds:
- Eliminates small islands (area < threshold)
- Fills small holes inside polygons (area < threshold)
Assesses DRC compliance before and after cleaning.
"""

from pathlib import Path
import json
import numpy as np
import klayout.db as db
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Rectangle

REPO_DIR = Path(r"G:\codex_project\inverse_design_sun\EGADO-4mode")
ARTIFACT_DIR = Path(r"C:\Users\USER\.gemini\antigravity\brain\39ad5e21-1313-4872-992d-8dc8c32f9b61")
GDS_IN = REPO_DIR / "baseline_4mode_scheme_b.gds"

DBU_UM = 0.001
UNITS_PER_NM = 1.0 / (DBU_UM * 1000.0)

def u(nm):
    return int(round(nm * UNITS_PER_NM))

def clean_region(region, island_thresh_um2=0.05, hole_thresh_um2=0.05):
    """Filter out small outer islands and small inner holes."""
    cleaned = db.Region()
    removed_islands = []
    filled_holes = []

    for poly in region.each_merged():
        outer_pts = list(poly.each_point_hull())
        outer_poly = db.Polygon(outer_pts)
        outer_area = outer_poly.area() * DBU_UM * DBU_UM

        if outer_area < island_thresh_um2:
            removed_islands.append((outer_area, outer_poly.bbox()))
            continue  # Drop this small island

        # Create new polygon with only valid, sufficiently large holes
        new_poly = db.Polygon(outer_pts)
        for h in range(poly.holes()):
            hole_pts = list(poly.each_point_hole(h))
            hole_poly = db.Polygon(hole_pts)
            hole_area = hole_poly.area() * DBU_UM * DBU_UM
            if hole_area < hole_thresh_um2:
                filled_holes.append((hole_area, hole_poly.bbox()))
                # Do not insert this hole -> filled
            else:
                new_poly.insert_hole(hole_pts)

        cleaned.insert(new_poly)

    cleaned.merge()
    return cleaned, removed_islands, filled_holes

def run_checks(r, roi_box):
    roi = db.Region(roi_box)
    inner = r & roi
    t120 = u(120.0)
    w_err = inner.width_check(t120, ignore_angle=85)
    s_err = inner.space_check(t120, ignore_angle=85)
    return w_err.count(), s_err.count(), w_err, s_err

def main():
    layout = db.Layout()
    layout.read(str(GDS_IN))
    top = layout.top_cell()
    layer_si = layout.layer(13, 1)
    layer_core = layout.layer(13, 2)
    layer_bound = layout.layer(251, 16)

    orig_region = db.Region(top.begin_shapes_rec(layer_si))
    orig_region.merge()

    design_box = db.Box(u(-3000), u(-3000), u(3000), u(3000))

    w_orig, s_orig, _, _ = run_checks(orig_region, design_box)
    print("=" * 72)
    print(f"Original Raw Skeleton (Iter 80): Width violations = {w_orig}, Space violations = {s_orig}")

    # Test cleaning with area threshold = 0.05 um2 (~220 nm x 220 nm)
    for thresh in [0.03, 0.05, 0.10, 0.15]:
        cleaned, rem_islands, fill_holes = clean_region(orig_region, island_thresh_um2=thresh, hole_thresh_um2=thresh)
        w_c, s_c, _, _ = run_checks(cleaned, design_box)
        print(f"Threshold {thresh:.2f} um2: Removed islands = {len(rem_islands)}, Filled holes = {len(fill_holes)} | Width violations = {w_c}, Space violations = {s_c}")

    # Use 0.12 um2 (corresponds to ~350 nm size, removes sub-resolution holes 0~4)
    best_thresh = 0.12
    cleaned_best, rem_islands, fill_holes = clean_region(orig_region, island_thresh_um2=best_thresh, hole_thresh_um2=best_thresh)
    w_clean, s_clean, w_errs, s_errs = run_checks(cleaned_best, design_box)

    # Save cleaned GDS
    out_gds = REPO_DIR / "baseline_4mode_cleaned.gds"
    layout_out = db.Layout()
    layout_out.dbu = DBU_UM
    top_out = layout_out.create_cell("TOP")
    top_out.shapes(layout_out.layer(13, 1)).insert(cleaned_best)
    top_out.shapes(layout_out.layer(13, 2)).insert(design_box)
    top_out.shapes(layout_out.layer(251, 16)).insert(db.Box(u(-4500), u(-4000), u(4500), u(4000)))
    layout_out.write(str(out_gds))
    print(f"\nCleaned GDS saved to: {out_gds}")
    print(f"Cleaned (threshold={best_thresh} um2): Width = {w_clean}, Space = {s_clean}")

    # Plot Comparison
    plot_comparison(orig_region, cleaned_best, rem_islands, fill_holes, w_orig, s_orig, w_clean, s_clean)

def plot_comparison(orig_r, clean_r, rem_islands, fill_holes, w_orig, s_orig, w_c, s_c):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), dpi=300)

    # Left: Original Raw
    ax1.set_facecolor("#f8f9fa")
    for poly in orig_r.each():
        hull = [np.array([p.x * DBU_UM, p.y * DBU_UM]) for p in poly.each_point_hull()]
        ax1.add_patch(MplPoly(np.array(hull), closed=True, facecolor="#1a5276", edgecolor="#113349", lw=0.8, alpha=0.9))
        for h in range(poly.holes()):
            hole = [np.array([p.x * DBU_UM, p.y * DBU_UM]) for p in poly.each_point_hole(h)]
            ax1.add_patch(MplPoly(np.array(hole), closed=True, facecolor="#f8f9fa", edgecolor="#c0392b", lw=1.2))

    # Mark filled holes and removed islands
    for area, b in fill_holes:
        rect = Rectangle((b.left*DBU_UM, b.bottom*DBU_UM), b.width()*DBU_UM, b.height()*DBU_UM,
                         edgecolor="#e74c3c", facecolor="none", lw=2.0, linestyle="--")
        ax1.add_patch(rect)
        ax1.text((b.left+b.right)*0.5*DBU_UM, (b.top)*DBU_UM + 0.1, f"Hole {area*1e6:.0f}nm²",
                 color="#c0392b", fontsize=7.5, fontweight="bold", ha="center")

    for area, b in rem_islands:
        rect = Rectangle((b.left*DBU_UM, b.bottom*DBU_UM), b.width()*DBU_UM, b.height()*DBU_UM,
                         edgecolor="#e67e22", facecolor="none", lw=2.0, linestyle=":")
        ax1.add_patch(rect)

    ax1.add_patch(Rectangle((-3.0, -3.0), 6.0, 6.0, lw=2.0, edgecolor="#c0392b", facecolor="none", linestyle="--"))
    ax1.set_xlim(-3.2, 3.2)
    ax1.set_ylim(-3.2, 3.2)
    ax1.set_aspect("equal")
    ax1.set_xlabel("X (μm)", fontweight="bold")
    ax1.set_ylabel("Y (μm)", fontweight="bold")
    ax1.set_title(f"Before Area Cleaning: Sub-resolution Holes & Islands\n120 nm DRC: Width={w_orig}, Space={s_orig}", fontweight="bold", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Right: Cleaned Layout
    ax2.set_facecolor("#f8f9fa")
    for poly in clean_r.each():
        hull = [np.array([p.x * DBU_UM, p.y * DBU_UM]) for p in poly.each_point_hull()]
        ax2.add_patch(MplPoly(np.array(hull), closed=True, facecolor="#1a5276", edgecolor="#113349", lw=0.8, alpha=0.9))
        for h in range(poly.holes()):
            hole = [np.array([p.x * DBU_UM, p.y * DBU_UM]) for p in poly.each_point_hole(h)]
            ax2.add_patch(MplPoly(np.array(hole), closed=True, facecolor="#f8f9fa", edgecolor="#113349", lw=0.8))

    ax2.add_patch(Rectangle((-3.0, -3.0), 6.0, 6.0, lw=2.0, edgecolor="#c0392b", facecolor="none", linestyle="--"))
    ax2.set_xlim(-3.2, 3.2)
    ax2.set_ylim(-3.2, 3.2)
    ax2.set_aspect("equal")
    ax2.set_xlabel("X (μm)", fontweight="bold")
    ax2.set_ylabel("Y (μm)", fontweight="bold")
    ax2.set_title(f"After Area Cleaning (Filtered Sub-resolution Defects)\n120 nm DRC: Width={w_c}, Space={s_c} (Clean Solid Skeletons)", fontweight="bold", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    local_img = REPO_DIR / "scheme_b_cleaning_comparison.png"
    artifact_img = ARTIFACT_DIR / "scheme_b_cleaning_comparison.png"
    plt.savefig(local_img, dpi=300)
    plt.savefig(artifact_img, dpi=300)
    plt.close()
    print(f"Comparison plot saved to:\n  {local_img}\n  {artifact_img}")

if __name__ == '__main__':
    main()
