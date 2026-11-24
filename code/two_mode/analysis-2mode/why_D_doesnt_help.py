"""Why does enlarging D not improve performance?  A mechanistic check.

The paper reports that going D = 0 -> 2 -> 4 (TBD 238 -> 308 -> 365 on our
device) gives no significant performance gain.  Hypothesis: the pixels that a
larger D adds are further from the material boundary, hence systematically LESS
ambiguous -- exactly the pixels whose material assignment the analog pattern
already determines.

We test that by comparing the ambiguity distribution of
    TBD(D=0),  TBD(D=2) \\ TBD(D=0),  TBD(D=4) \\ TBD(D=2)
where ambiguity A_i = 1 - |2 p_i - 1| and p_i is the silicon fraction of the
120 nm cell.
"""
import os
import sys

import numpy as np

MESH_NM, PIXEL_NM = 20.0, 120.0
S = 6
repo = r"G:\codex_project\inverse_design_sun\EGADO-2mode"
sys.path.insert(0, repo)
os.chdir(repo)
from func_get_conversion_matrix import get_conversion_matrix  # noqa: E402

work = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_amb")
os.makedirs(work, exist_ok=True)
ckpt = r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_final.npz"

z = np.load(ckpt)
an = z["eps"].T[:-1, :-1]
an = -1 * np.flip(an, axis=0)
rho = (an - an.min()) / (an.max() - an.min())
ny, nx = rho.shape
NC = nx // S

p = rho.reshape(NC, S, NC, S).mean(axis=(1, 3))
A = 1.0 - np.abs(2 * p - 1)

tbd = {}
for D in (0, 2, 4):
    m = get_conversion_matrix(analog_file=ckpt, target_folder=work,
                              pixel_size=PIXEL_NM * 1e-9, mesh_size=int(MESH_NM), buff=D)
    tbd[D] = (m >= 0.49) & (m <= 0.51)

groups = [("TBD(D=0)", tbd[0]),
          ("added by D=2", tbd[2] & ~tbd[0]),
          ("added by D=4", tbd[4] & ~tbd[2])]

print("A_i = 1-|2p_i-1|   (0 = cell is pure Si or pure oxide, 1 = exactly 50/50)")
print()
print("%-16s %6s %8s %8s %8s %8s %8s" % ("group", "n", "mean A", "median", "A<0.1", "A<0.25", "A>0.5"))
for name, g in groups:
    if g.sum() == 0:
        print("%-16s %6d" % (name, 0))
        continue
    a = A[g]
    print("%-16s %6d %8.3f %8.3f %7.1f%% %7.1f%% %7.1f%%"
          % (name, g.sum(), a.mean(), np.median(a),
             100 * (a < 0.1).mean(), 100 * (a < 0.25).mean(), 100 * (a > 0.5).mean()))

print()
print("Interpretation guide:")
print("  if 'added by D=2/4' has a clearly LOWER mean A than TBD(D=0), then a larger D")
print("  spends FDTD solves on cells the analog pattern already decides -- which would")
print("  explain the paper's own observation that more TBD brings no gain.")
print()
base = A[tbd[0]].mean()
for name, g in groups[1:]:
    if g.sum():
        print("  mean A: %-14s %.3f  vs  TBD(D=0) %.3f   (ratio %.2f)"
              % (name, A[g].mean(), base, A[g].mean() / base))
