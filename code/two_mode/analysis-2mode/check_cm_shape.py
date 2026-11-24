"""Verify the shape of get_conversion_matrix's return value.

pattern_mapping() should fold the 180x180 analog pattern down to 30x30 digital
pixels, so the conversion matrix must be (pixels_per_col, pixels_per_row) =
(30, 30) -- anything else breaks EG-ATD's pixel indexing.
"""
import os
import sys

import numpy as np

repo = r"G:\codex_project\inverse_design_sun\EGADO-2mode"
sys.path.insert(0, repo)
os.chdir(repo)

from func_get_conversion_matrix import get_conversion_matrix, pattern_mapping, convol  # noqa: E402
from skimage import feature  # noqa: E402

work = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shape_check")
os.makedirs(work, exist_ok=True)

z = np.load(r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_final.npz")
eps = z["eps"]
an = eps.T[:-1, :-1]
an = -1 * np.flip(an, axis=0)
an = (an - an.min()) / (an.max() - an.min())

pixel_size = np.ceil((z["x"][1] - z["x"][0]) * 1e9)
scale = int(pixel_size / 20)
print("analog pattern     :", an.shape)
print("pixel_size (nm)    :", pixel_size, " mesh 20 nm  -> scale =", scale)

edge = feature.canny(an)
print("edge map           :", edge.shape, " edges =", int(edge.sum()))

for D in (0, 2, 4):
    conv = convol(edge, scale + D, scale, D // 2)
    mapped = pattern_mapping(an, conv, 0.5, scale)
    print("  D=%d : conv_result %s   mapped_result %s" % (D, conv.shape, mapped.shape))

print()
m = get_conversion_matrix(analog_file=r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_final.npz",
                          target_folder=work, pixel_size=z["x"][1] - z["x"][0],
                          mesh_size=20, buff=2)
print("get_conversion_matrix returned:", m.shape, m.dtype)
print("saved npz key shapes:")
s = np.load(os.path.join(work, "conversion_matrix.npz"))
for k in s.files:
    print("   %-20s %s" % (k, s[k].shape))
print()
print("EXPECTED for a 3.6um design region at 120 nm pixels: (30, 30) = 900 pixels")
