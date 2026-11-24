"""Setup Scheme B inputs for 4-mode device:
1. Extract candidate cells on 50x50 120nm grid based on 120nm DRC violations.
2. Run fast geometric pre-optimization to obtain 0-DRC baseline geometry.
3. Export fixed_background.gds, initial.gds, and initial_state.npz.
"""

import sys
import time
from pathlib import Path
import json
import numpy as np
import klayout.db as db

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

REPO_DIR = Path(__file__).resolve().parent
SOURCE_GDS = REPO_DIR / "baseline_4mode_true_iter266.gds"
OUT_DIR = REPO_DIR / "scheme_b_dbs_inputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROWS = COLS = 50
CELL_NM = 120
ORIGIN_NM = -3000
DBU_UM = 0.001
UNITS_PER_NM = 1.0 / (DBU_UM * 1000.0)

def u(nm):
    return int(round(nm * UNITS_PER_NM))

ROI_BOX = db.Box(u(-3000), u(-3000), u(3000), u(3000))
ROI_REGION = db.Region(ROI_BOX)

def cell_box(row, col):
    x0 = ORIGIN_NM + col * CELL_NM
    y0 = ORIGIN_NM + row * CELL_NM
    return db.Box(u(x0), u(y0), u(x0 + CELL_NM), u(y0 + CELL_NM))

def cell_boxes(mask):
    r = db.Region()
    for row, col in np.argwhere(mask):
        r.insert(cell_box(row, col))
    r.merge()
    return r

def read_source_layout():
    layout = db.Layout()
    layout.read(str(SOURCE_GDS))
    top = layout.top_cell()
    layer_si = layout.layer(13, 1)
    core = db.Region(top.begin_shapes_rec(layer_si))
    core.merge()
    return layout, top, core

def count_violations(region, target_nm=120):
    """Ultra-fast C++ DRC violation count."""
    thr = u(target_nm)
    design_reg = region & ROI_REGION
    w_count = design_reg.width_check(thr, ignore_angle=85).count()
    s_count = design_reg.space_check(thr, ignore_angle=85).count()
    return w_count, s_count

def extract_marker_boxes(region, target_nm=120):
    """Extracts bounding boxes of DRC violations."""
    thr = u(target_nm)
    design_reg = region & ROI_REGION
    width = design_reg.width_check(thr, ignore_angle=85)
    space = design_reg.space_check(thr, ignore_angle=85)
    
    marker_boxes = []
    for errors in (width, space):
        for ep in errors.each():
            bb = ep.bbox()
            marker_boxes.append((bb.left * 1.0, bb.bottom * 1.0, bb.right * 1.0, bb.top * 1.0))
    return marker_boxes

def get_candidate_mask(marker_boxes, padding_nm=0):
    m = np.zeros((ROWS, COLS), dtype=bool)
    for a, b, c, d in marker_boxes:
        a_nm, b_nm, c_nm, d_nm = a * DBU_UM * 1000, b * DBU_UM * 1000, c * DBU_UM * 1000, d * DBU_UM * 1000
        j0 = max(0, int(np.floor((a_nm - ORIGIN_NM - padding_nm - 0.5) / CELL_NM)))
        j1 = min(COLS, int(np.ceil((c_nm - ORIGIN_NM + padding_nm + 0.5) / CELL_NM)))
        i0 = max(0, int(np.floor((b_nm - ORIGIN_NM - padding_nm - 0.5) / CELL_NM)))
        i1 = min(ROWS, int(np.ceil((d_nm - ORIGIN_NM + padding_nm + 0.5) / CELL_NM)))
        if i1 > i0 and j1 > j0:
            m[i0:i1, j0:j1] = True
    return m

def form_silicon(fixed_bg, mask, state):
    si = db.Region()
    for row, col in np.argwhere(mask & state):
        si.insert(cell_box(row, col))
    si.merge()
    r = fixed_bg | si
    r.merge()
    return r

def main():
    print("=== Step 1: Loading Continuous Skeleton GDS ===", flush=True)
    layout, top, source_core = read_source_layout()
    counts = count_violations(source_core, 120)
    print(f"Raw skeleton 120nm violations: Width={counts[0]}, Space={counts[1]}", flush=True)
    
    # 1. Candidate mask from DRC violations
    markers = extract_marker_boxes(source_core, 120)
    mask = get_candidate_mask(markers, padding_nm=0)
    print(f"Initial candidate cells touched by violations: {mask.sum()} / 2500 ({mask.sum()/2500*100:.1f}%)", flush=True)
    
    # Initial state by silicon area majority rule
    state = np.zeros((ROWS, COLS), dtype=bool)
    for r, c in np.argwhere(mask):
        box = cell_box(r, c)
        cell_reg = db.Region(box)
        inter = source_core & cell_reg
        state[r, c] = (inter.area() >= (0.5 * box.area()))
        
    cand_boxes = cell_boxes(mask)
    fixed_bg = source_core - cand_boxes
    
    current_geom = form_silicon(fixed_bg, mask, state)
    current_counts = count_violations(current_geom, 120)
    print(f"Filled initial majority state violations: Width={current_counts[0]}, Space={current_counts[1]} (Total={sum(current_counts)})", flush=True)
    
    print("\n=== Step 2: Bounded Geometric Search for 0-DRC Baseline ===", flush=True)
    t0 = time.time()
    best_counts = current_counts
    best_state = state.copy()
    best_mask = mask.copy()
    
    for round_idx in range(1, 15):
        if sum(current_counts) == 0:
            print(f"Achieved 0 DRC violations at round {round_idx}!", flush=True)
            break
            
        cur_markers = extract_marker_boxes(current_geom, 120)
        affected = get_candidate_mask(cur_markers, padding_nm=10) & mask
        flips = 0
        
        # Test flips on affected candidate cells
        for r, c in np.argwhere(affected):
            trial_state = state.copy()
            trial_state[r, c] = not trial_state[r, c]
            trial_geom = form_silicon(fixed_bg, mask, trial_state)
            t_counts = count_violations(trial_geom, 120)
            
            if sum(t_counts) < sum(current_counts):
                state = trial_state
                current_geom = trial_geom
                current_counts = t_counts
                flips += 1
                if sum(current_counts) < sum(best_counts):
                    best_counts = current_counts
                    best_state = state.copy()
                    best_mask = mask.copy()
                if sum(current_counts) == 0:
                    break
                    
        print(f"  Round {round_idx}: Violations = {current_counts} (Total={sum(current_counts)}), Flips = {flips}", flush=True)
        if flips == 0:
            # Expand mask slightly near remaining violations
            additions = get_candidate_mask(cur_markers, padding_nm=10) & ~mask
            if additions.any() and mask.sum() + additions.sum() <= 400:
                print(f"  Expanding mask by {additions.sum()} cells...", flush=True)
                for r, c in np.argwhere(additions):
                    box = cell_box(r, c)
                    inter = source_core & db.Region(box)
                    state[r, c] = (inter.area() >= (0.5 * box.area()))
                mask |= additions
                cand_boxes = cell_boxes(mask)
                fixed_bg = source_core - cand_boxes
                current_geom = form_silicon(fixed_bg, mask, state)
                current_counts = count_violations(current_geom, 120)
            else:
                break
                
    final_geom = form_silicon(source_core - cell_boxes(best_mask), best_mask, best_state)
    final_counts = count_violations(final_geom, 120)
    print(f"\nGeometric search completed in {time.time()-t0:.1f}s.", flush=True)
    print(f"Best violations: Width={final_counts[0]}, Space={final_counts[1]}, Candidates={best_mask.sum()}", flush=True)
    
    # Prune unnecessary candidate cells that don't violate DRC if reverted to continuous curve
    print("Pruning redundant candidate cells...", flush=True)
    fixed_base = source_core - cell_boxes(best_mask)
    pruned = 0
    for r, c in np.argwhere(best_mask):
        trial_mask = best_mask.copy()
        trial_mask[r, c] = False
        t_fixed = source_core - cell_boxes(trial_mask)
        trial_geom = form_silicon(t_fixed, trial_mask, best_state)
        p_counts = count_violations(trial_geom, 120)
        if sum(p_counts) <= sum(final_counts):
            best_mask = trial_mask
            final_counts = p_counts
            pruned += 1
            
    print(f"Pruned {pruned} candidate cells. Final frozen candidate count: {best_mask.sum()}", flush=True)
    
    final_cand_boxes = cell_boxes(best_mask)
    final_bg = source_core - final_cand_boxes
    final_geom = form_silicon(final_bg, best_mask, best_state)
    final_counts = count_violations(final_geom, 120)
    print(f"Final Candidate Device Violations: Width={final_counts[0]}, Space={final_counts[1]}", flush=True)
    
    # Step 3: Export GDS and NPZ
    print("\n=== Step 3: Exporting Scheme B Artifacts ===", flush=True)
    out_initial_gds = OUT_DIR / "initial.gds"
    out_bg_gds = OUT_DIR / "fixed_background.gds"
    
    # Write initial.gds
    l_init = db.Layout()
    l_init.dbu = DBU_UM
    t_init = l_init.create_cell("TOP")
    t_init.shapes(l_init.layer(13, 1)).insert(final_geom)
    l_init.write(str(out_initial_gds))
    
    # Write fixed_background.gds
    l_bg = db.Layout()
    l_bg.dbu = DBU_UM
    t_bg = l_bg.create_cell("TOP")
    t_bg.shapes(l_bg.layer(13, 1)).insert(final_bg)
    l_bg.write(str(out_bg_gds))
    
    # Save NPZ
    order = np.array(sorted(map(tuple, np.argwhere(best_mask)), key=lambda rc: (rc[1], -rc[0])), dtype=np.int64)
    np.savez_compressed(
        OUT_DIR / "initial_state.npz",
        candidate_mask=best_mask,
        silicon_state=best_state,
        order_rc=order
    )
    print(f"Successfully saved to {OUT_DIR}:", flush=True)
    print(f"  - initial.gds ({out_initial_gds.stat().st_size} bytes)", flush=True)
    print(f"  - fixed_background.gds ({out_bg_gds.stat().st_size} bytes)", flush=True)
    print(f"  - initial_state.npz ({best_mask.sum()} candidate cells)", flush=True)

if __name__ == "__main__":
    main()
