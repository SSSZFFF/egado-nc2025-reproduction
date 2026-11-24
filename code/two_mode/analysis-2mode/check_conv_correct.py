"""REDO of the digitisation with the CORRECT pixel_size.

get_conversion_matrix(analog_file, target_folder, pixel_size, mesh_size, buff)
must be called with pixel_size = the DIGITAL pixel side (120 nm here), exactly
as config_for_eg_a2d does (pixel_size=self.rect_size_x).  Passing the 20 nm mesh
spacing instead makes scale = int(20/20) = 1, which skips pooling entirely and
leaves the matrix at the 180x180 analog resolution.
"""
import os
import sys

import numpy as np

repo = r"G:\codex_project\inverse_design_sun\EGADO-2mode"
sys.path.insert(0, repo)
os.chdir(repo)

from func_get_conversion_matrix import get_conversion_matrix  # noqa: E402

work = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_digitize_correct")
os.makedirs(work, exist_ok=True)

ckpt = r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_final.npz"
z = np.load(ckpt)
rect_size = 120e-9            # <- the digital pixel side, as config_for_eg_a2d passes
scale = int(np.ceil(rect_size * 1e9) / 20)
print("digital pixel %.0f nm, mesh 20 nm -> scale = %d" % (rect_size * 1e9, scale))
print("design region %.1f x %.1f um -> expect %d x %d = %d digital pixels" % (
    (z["x"][-1] - z["x"][0]) * 1e6, (z["y"][-1] - z["y"][0]) * 1e6,
    int((z["x"][-1] - z["x"][0]) / rect_size), int((z["y"][-1] - z["y"][0]) / rect_size),
    int((z["x"][-1] - z["x"][0]) / rect_size) ** 2))
print()

print("%-6s %-10s %-12s %-12s %-12s %-10s" % (
    "D", "matrix", "deposit", "edge/TBD", "etch", "unclassified"))
res = {}
for D in (0, 2, 4):
    m = get_conversion_matrix(analog_file=ckpt, target_folder=work,
                              pixel_size=rect_size, mesh_size=20, buff=D)
    res[D] = m
    n = m.size
    dep = int(np.sum(m < 0.01))
    edg = int(np.sum((m >= 0.49) & (m <= 0.51)))
    etc = int(np.sum(m > 0.99))
    unc = n - dep - edg - etc
    print("%-6d %-10s %-12s %-12s %-12s %-10s" % (
        D, str(m.shape),
        "%d (%.1f%%)" % (dep, 100.0 * dep / n),
        "%d (%.1f%%)" % (edg, 100.0 * edg / n),
        "%d (%.1f%%)" % (etc, 100.0 * etc / n),
        "%d (%.1f%%)" % (unc, 100.0 * unc / n)))

print()
print("PAPER (Supplementary Note 5), two-mode MUX at D = 0: 71.22% direct conversion")
m0 = res[0]
print("ours at D = 0: %.2f%%  (%d of %d pixels are TBD)" % (
    100.0 * np.sum((m0 >= 0.49) & (m0 <= 0.51)) / m0.size,
    int(np.sum((m0 >= 0.49) & (m0 <= 0.51))), m0.size))
np.save(os.path.join(work, "cm_D0.npy"), res[0])
np.save(os.path.join(work, "cm_D2.npy"), res[2])
np.save(os.path.join(work, "cm_D4.npy"), res[4])
print("\nsaved corrected matrices to", work)
