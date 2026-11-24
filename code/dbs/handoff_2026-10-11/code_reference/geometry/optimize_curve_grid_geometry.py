"""Bounded geometric two-state search for the curve/grid interface trial.

Surrogate checks reproduce WGF03/04/05 geometry and use conservative native
width/space checks. Final acceptance always requires a fresh official report.
"""

from pathlib import Path
from itertools import combinations
import json
import hashlib
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import klayout.db as db

HERE = Path(__file__).resolve().parent
OUT = HERE / "curve_then_grid_trial"
manifest = json.loads((OUT / "grid_parameters.json").read_text(encoding="utf-8"))
layout = db.Layout()
layout.read(manifest["source_gds"])
DBU = layout.dbu
UPNM = 1 / (DBU * 1000)
source = db.Region(layout.top_cell().begin_shapes_rec(layout.layer(13, 1)))
source.merge()
other = {(a, b): db.Region(layout.top_cell().begin_shapes_rec(layout.layer(a, b)))
         for a, b in ((13, 2), (251, 16))}
LIMIT_SECONDS = 90


def box(row, col):
    return db.Box(*[int(round(x * UPNM)) for x in
                    (-1800 + col * 120, -1800 + row * 120,
                     -1800 + (col + 1) * 120, -1800 + (row + 1) * 120)])


def boxes(mask):
    r = db.Region()
    for row, col in np.argwhere(mask):
        r.insert(box(row, col))
    r.merge()
    return r


def majority(row, col):
    b = box(row, col)
    return (source & db.Region(b)).area() / b.area() >= 0.5


def form(mask, state):
    si = db.Region()
    for row, col in np.argwhere(mask & state):
        si.insert(box(row, col))
    si.merge()
    r = (source - boxes(mask)) | si
    r.merge()
    return r


def assess(r, target):
    threshold = int(round(target * UPNM))
    width = r.width_check(threshold, ignore_angle=85)
    space = r.space_check(threshold, ignore_angle=85)
    marker_boxes = []
    for errors in (width, space):
        for ep in errors.each():
            bb = ep.bbox()
            marker_boxes.append((bb.left, bb.bottom, bb.right, bb.top))
    notch = angle = small_area = 0
    for polygon in r.each_merged():
        if polygon.area() * DBU ** 2 < 0.2:
            small_area += 1
            bb = polygon.bbox()
            marker_boxes.append((bb.left, bb.bottom, bb.right, bb.top))
        rings = [list(polygon.each_point_hull())]
        rings += [list(polygon.each_point_hole(i)) for i in range(polygon.holes())]
        for ring in rings:
            p = np.array([(v.x, v.y) for v in ring], dtype=float)
            incoming, outgoing = p - np.roll(p, 1, axis=0), np.roll(p, -1, axis=0) - p
            cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
            alpha = 180 + np.degrees(np.arctan2(cross, (incoming * outgoing).sum(axis=1)))
            convex, concave = (alpha >= 0) & (alpha <= 160), (alpha >= 200) & (alpha <= 360)
            length = np.linalg.norm(outgoing, axis=1) / UPNM
            bad_edge = (((convex & np.roll(concave, -1)) | (concave & np.roll(convex, -1)))
                        & (length >= 6 - 1e-6) & (length < 100 - 1e-6))
            bad_angle = (alpha < 84.9) | (alpha > 270.1)
            notch += int(bad_edge.sum())
            angle += int(bad_angle.sum())
            for i in np.flatnonzero(bad_edge):
                a, b = p[i], p[(i + 1) % len(p)]
                marker_boxes.append((min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])))
            for i in np.flatnonzero(bad_angle):
                marker_boxes.append((p[i, 0], p[i, 1], p[i, 0], p[i, 1]))
    counts = (width.count(), space.count(), notch, angle, small_area)
    return counts, marker_boxes


def marker_mask(marker_boxes, padding_nm=0):
    m = np.zeros((30, 30), bool)
    for a, b, c, d in marker_boxes:
        a, b, c, d = np.array([a, b, c, d]) / UPNM
        j0 = max(0, int(np.floor((a + 1800 - padding_nm - .5) / 120)))
        j1 = min(30, int(np.ceil((c + 1800 + padding_nm + .5) / 120)))
        i0 = max(0, int(np.floor((b + 1800 - padding_nm - .5) / 120)))
        i1 = min(30, int(np.ceil((d + 1800 + padding_nm + .5) / 120)))
        if i1 > i0 and j1 > j0:
            m[i0:i1, j0:j1] = True
    return m


results = {}
for name, target in (("sp90_90nm", 90), ("target_120nm", 120)):
    t0 = time.perf_counter()
    input_arrays = np.load(OUT / f"{name}_cells.npz", allow_pickle=False)
    mask, state = input_arrays["candidate_mask"].copy(), input_arrays["silicon_state"].copy()
    original_mask = mask.copy()
    cache = {}
    history = []

    def get(mask, state):
        key = np.packbits(mask).tobytes() + np.packbits(state & mask).tobytes()
        if key not in cache:
            r = form(mask, state)
            counts, marks = assess(r, target)
            delta = (r ^ source).area() * DBU ** 2
            cache[key] = (sum(counts), delta, int(mask.sum())), counts, marks, r
        return cache[key]

    current = get(mask, state)
    best = current[0], mask.copy(), state.copy()
    print(name, "initial", current[1], "cells", mask.sum(), flush=True)
    for iteration in range(10):
        if current[0][0] == 0 or time.perf_counter() - t0 > LIMIT_SECONDS:
            break
        affected = marker_mask(current[2], 20) & mask
        flips = 0
        for row, col in np.argwhere(affected):
            if time.perf_counter() - t0 > LIMIT_SECONDS:
                break
            trial = state.copy()
            trial[row, col] = ~trial[row, col]
            candidate = get(mask, trial)
            if candidate[0] < current[0]:
                state, current = trial, candidate
                flips += 1
                if current[0][0] == 0:
                    break
        history.append({"round": iteration + 1, "counts": list(current[1]),
                        "cells": int(mask.sum()), "flips": flips})
        print(name, "round", iteration + 1, current[1], "cells", mask.sum(), "flips", flips, flush=True)
        if current[0] < best[0]:
            best = current[0], mask.copy(), state.copy()
        if current[0][0] == 0 or time.perf_counter() - t0 > LIMIT_SECONDS:
            break
        additions = marker_mask(current[2], 20) & ~mask
        if additions.any() and mask.sum() + additions.sum() <= 200:
            for row, col in np.argwhere(additions):
                state[row, col] = majority(row, col)
            mask |= additions
            current = get(mask, state)
            continue
        if not flips:
            positions = [tuple(p) for p in np.argwhere(affected)][:20]
            winner = None
            for first, second in combinations(positions, 2):
                if time.perf_counter() - t0 > LIMIT_SECONDS:
                    break
                trial = state.copy()
                for row, col in (first, second):
                    trial[row, col] = ~trial[row, col]
                candidate = get(mask, trial)
                if candidate[0] < current[0] and (winner is None or candidate[0] < winner[0][0]):
                    winner = candidate, trial
                    if candidate[0][0] == 0:
                        break
            if winner is None:
                break
            current, state = winner

    _, mask, state = best
    current = get(mask, state)
    # Recover any fine curves whose restoration does not reintroduce errors.
    if current[0][0] == 0:
        for row, col in np.argwhere(mask):
            if time.perf_counter() - t0 > LIMIT_SECONDS:
                break
            trial_mask = mask.copy()
            trial_mask[row, col] = False
            candidate = get(trial_mask, state)
            if candidate[0][0] == 0:
                mask, current = trial_mask, candidate
    gds = OUT / f"{name}_optimized.gds"
    if gds.exists():
        raise FileExistsError(gds)
    l = db.Layout()
    l.dbu = DBU
    cell = l.create_cell(name.upper() + "_OPTIMIZED")
    cell.shapes(l.layer(13, 1)).insert(current[3])
    for layer, r in other.items():
        cell.shapes(l.layer(*layer)).insert(r)
    l.write(str(gds))
    actual_l = db.Layout()
    actual_l.read(str(gds))
    actual = db.Region(actual_l.top_cell().begin_shapes_rec(actual_l.layer(13, 1)))
    actual.merge()
    delta_write = (actual ^ current[3]).area() * DBU ** 2
    assert delta_write < 1e-6
    post_counts, _ = assess(actual, target)
    output = {
        "target_nm": target, "gds": str(gds), "gds_sha256": hashlib.sha256(gds.read_bytes()).hexdigest().upper(),
        "initial_cells": int(original_mask.sum()), "final_cells": int(mask.sum()),
        "surrogate_counts_width_space_notch_angle_area": list(post_counts),
        "requires_official_drc_confirmation": True,
        "changed_area_percent_of_design": (actual ^ source).area() * DBU ** 2 / 12.96 * 100,
        "outside_candidate_difference_um2": ((actual ^ source) - boxes(mask)).area() * DBU ** 2,
        "curve_area_outside_cells_percent": 100 * (900 - mask.sum()) / 900,
        "gds_readback_difference_um2": delta_write,
        "geometry_evaluations": len(cache), "elapsed_seconds": time.perf_counter() - t0,
        "history": history, "optical_simulations": 0,
    }
    results[name] = output
    np.savez_compressed(OUT / f"{name}_optimized_cells.npz", candidate_mask=mask, silicon_state=state)
    print(name, "FINAL", output, flush=True)
(OUT / "optimization_parameters.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
