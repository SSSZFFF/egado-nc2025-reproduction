"""Clean small boundary slivers from Scheme B skeleton @ Iter 266.

Filters out:
- Outer islands with area < 0.08 um^2
- Inner holes with area < 0.08 um^2
Assesses DRC compliance before and after cleaning.
Generates baseline_4mode_cleaned_iter266.gds.
"""

from pathlib import Path
import json
import shutil
import numpy as np
import klayout.db as db
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly, Rectangle

REPO_DIR = Path(r"G:\codex_project\inverse_design_sun\EGADO-4mode")
ARTIFACT_DIR = Path(r"C:\Users\USER\.gemini\antigravity\brain\39ad5e21-1313-4872-992d-8dc8c32f9b61")
GDS_IN = REPO_DIR / "baseline_4mode_scheme_b_iter266.gds"
GDS_OUT = REPO_DIR / "baseline_4mode_cleaned_iter266.gds"

DBU_UM = 0.001
UNITS_PER_NM = 1.0 / (DBU_UM * 1000.0)

def u(nm):
    return int(round(nm * UNITS_PER_NM))

def clean_region(region, island_thresh_um2=0.08, hole_thresh_um2=0.08):
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
    cell = layout.top_cell()
    layer_si = layout.layer(13, 1)
    layer_core = layout.layer(13, 2)
    layer_bound = layout.layer(251, 16)

    raw_region = db.Region(cell.shapes(layer_si))
    roi_box = db.Box(u(-3000.0), u(-3000.0), u(3000.0), u(3000.0))

    w_before, s_before, _, _ = run_checks(raw_region, roi_box)
    print(f"Raw Skeleton (Iter 266): Polygons = {raw_region.count()}, Width err = {w_before}, Space err = {s_before}")

    cleaned_region, removed_islands, filled_holes = clean_region(raw_region, island_thresh_um2=0.08, hole_thresh_um2=0.08)
    w_after, s_after, _, _ = run_checks(cleaned_region, roi_box)
    print(f"Cleaned Skeleton (Iter 266): Polygons = {cleaned_region.count()}, Width err = {w_after}, Space err = {s_after}")
    print(f"Removed {len(removed_islands)} tiny islands, filled {len(filled_holes)} tiny holes.")

    # Save Cleaned GDS
    out_layout = db.Layout()
    out_layout.dbu = DBU_UM
    out_cell = out_layout.create_cell("TOP")
    out_cell.shapes(out_layout.layer(13, 1)).insert(cleaned_region)
    out_cell.shapes(out_layout.layer(13, 2)).insert(roi_box)
    bound_box = db.Box(u(-5000.0), u(-4000.0), u(5000.0), u(4000.0))
    out_cell.shapes(out_layout.layer(251, 16)).insert(bound_box)
    out_layout.write(str(GDS_OUT))
    print(f"Saved Cleaned GDS to: {GDS_OUT}")

    # Plot Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    for ax, r, title in zip(axes, [raw_region, cleaned_region], ['Raw Skeleton (Iter 266)', 'Cleaned Skeleton (Iter 266)']):
        ax.add_patch(Rectangle((-3.0, -3.0), 6.0, 6.0, fill=False, edgecolor='blue', linestyle='--', linewidth=1.5))
        for poly in r.each():
            hull = np.array([[pt.x * DBU_UM, pt.y * DBU_UM] for pt in poly.each_point_hull()])
            ax.add_patch(MplPoly(hull, facecolor='#2b5c8f', edgecolor='black', linewidth=0.8, alpha=0.9))
            for h in range(poly.holes()):
                hp = np.array([[pt.x * DBU_UM, pt.y * DBU_UM] for pt in poly.each_point_hole(h)])
                ax.add_patch(MplPoly(hp, facecolor='white', edgecolor='black', linewidth=0.8))
        ax.set_xlim(-5.0, 5.0)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)

    img_local = REPO_DIR / "scheme_b_iter266_cleaning_comparison.png"
    plt.tight_layout()
    plt.savefig(img_local, dpi=300)
    plt.close()
    shutil.copy(img_local, ARTIFACT_DIR / "scheme_b_iter266_cleaning_comparison.png")
    print(f"Saved cleaning comparison to: {img_local}")

if __name__ == "__main__":
    main()
