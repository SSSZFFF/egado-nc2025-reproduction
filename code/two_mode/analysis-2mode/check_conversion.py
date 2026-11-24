"""Check that EG-ATD's conversion matrix comes out 30x30 for the 2-mode device.

get_conversion_matrix() reads a TO checkpoint npz, so we synthesise a plausible
one at the 2-mode geometry (3.6 x 3.6 um on the 20 nm TO mesh) and run the real
function for buff = 0, 2, 4 (the paper's dilation coefficients D).

usage:  python check_conversion.py <repo>
"""
import os
import sys

import numpy as np

repo = os.path.abspath(sys.argv[1])
sys.path.insert(0, repo)
os.chdir(repo)

from func_get_conversion_matrix import get_conversion_matrix  # noqa: E402

work = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_conv_check")
os.makedirs(work, exist_ok=True)

# 20 nm TO mesh over 3.6 um -> 181 x 181 points; 220 nm Si slab sampled at 3 z
x = np.linspace(-1.8e-6, 1.8e-6, int(3600 / 20) + 1)
y = np.linspace(-1.8e-6, 1.8e-6, int(3600 / 20) + 1)
z = np.linspace(-110e-9, 110e-9, 3)

eps_bg, eps_wg = 1.4440 ** 2, 3.4764 ** 2

# a synthetic "optimised" pattern: smooth blobs, so Canny finds real edges
rng = np.random.default_rng(0)
blob = np.zeros((len(x), len(y)))
for _ in range(14):
    cx, cy = rng.uniform(-1.4e-6, 1.4e-6, 2)
    r = rng.uniform(0.25e-6, 0.7e-6)
    blob += np.exp(-(((x[:, None] - cx) ** 2 + (y[None, :] - cy) ** 2) / (2 * r ** 2)))
blob = np.clip(blob / blob.max(), 0, 1)
eps = eps_bg + (eps_wg - eps_bg) * blob

npz = os.path.join(work, "synthetic_TO.npz")
np.savez(npz, x=x, y=y, z=z, eps=eps, params=blob,
         eps_min=eps_bg, eps_max=eps_wg, beta=1.0)
print("synthetic checkpoint:", npz)
print("  grid x/y/z =", len(x), len(y), len(z))

print()
exp = (int(3600 / 20) // 6, int(3600 / 20) // 6)   # 30 x 30 pixels
print("expected conversion matrix shape (pixels_per_col, pixels_per_row) =", exp)
print()
ok = True
for buff in (0, 2, 4):
    print("--- buff = {}  (paper's D = {}) ---".format(buff, buff))
    try:
        cm = get_conversion_matrix(analog_file=npz, target_folder=work,
                                   pixel_size=120e-9, mesh_size=20, buff=buff)
    except Exception as exc:
        print("   FAILED: {}: {}".format(type(exc).__name__, exc))
        ok = False
        continue
    print("   shape = {}   values = {}".format(cm.shape, sorted(set(np.unique(cm).tolist()))))
    n_gray = int(np.sum((cm > 0.01) & (cm < 0.99)))
    print("   non-binary (edge) cells = {} / {} = {:.1f}%".format(
        n_gray, cm.size, 100.0 * n_gray / cm.size))
    if cm.shape != exp:
        print("   !! shape mismatch, expected", exp)
        ok = False

print()
print("RESULT:", "all buff values give the expected 30x30 matrix" if ok else "SOME CHECKS FAILED")
