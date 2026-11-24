"""Visual check of the quantisation ladder, to explain the 60 -> 120 nm cliff."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MESH_NM = 20.0
CKPT = r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_final.npz"
FOM = {20.0: 1.950978, 40.0: 1.931392, 60.0: 1.860536, 120.0: 1.404275}
DRC = {20.0: 859, 40.0: 264, 60.0: 127, 120.0: 8}

z = np.load(CKPT)
an = z["eps"].T[:-1, :-1]
an = -1 * np.flip(an, axis=0)
rho = (an - an.min()) / (an.max() - an.min())
nx, ny = rho.shape


def quantise(pitch_nm, rule="mean"):
    k = int(round(pitch_nm / MESH_NM))
    n = nx // k
    out = np.zeros((nx, ny), dtype=bool)
    for I in range(n):
        for J in range(n):
            blk = rho[I * k:(I + 1) * k, J * k:(J + 1) * k]
            v = blk[(k - 1) // 2, (k - 1) // 2] if rule == "centre" else blk.mean()
            out[I * k:(I + 1) * k, J * k:(J + 1) * k] = v < 0.5
    return out


def to_display(m):
    return m[::-1, :].T


panels = [("analog (20 nm grid, continuous)", rho < 0.5, None, None),
          ("20 nm  (binary20)", quantise(20.0), 20.0, None),
          ("40 nm", quantise(40.0), 40.0, None),
          ("60 nm", quantise(60.0), 60.0, None),
          ("120 nm  centre rule", quantise(120.0, "centre"), 120.0, None),
          ("120 nm  mean rule", quantise(120.0, "mean"), 120.0, None)]

fig, axes = plt.subplots(1, 6, figsize=(21, 4.1))
for ax, (title, B, pitch, _) in zip(axes, panels):
    ax.imshow(to_display(B), origin="lower", cmap="bone_r", interpolation="nearest",
              extent=[-1.8, 1.8, -1.8, 1.8])
    sub = "silicon %.1f%%" % (100 * B.mean())
    if pitch is not None:
        sub += "\nS T = %.4f   DRC %d" % (FOM[pitch], DRC[pitch])
    ax.set_title("%s\n%s" % (title, sub), fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    if ax is axes[0]:
        ax.set_ylabel("y (um)")

fig.suptitle("Boundary-pitch ladder: the optical cliff sits between 60 nm and 120 nm", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.9])
out = r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_pitch_ladder.png"
plt.savefig(out, dpi=140)
print("wrote", out)

print()
print("%-10s %10s %12s %12s %12s" % ("pitch", "vars", "Si frac", "cells vs 20nm", "edge cells"))
B20 = quantise(20.0)
for p in (20.0, 40.0, 60.0, 120.0):
    B = quantise(p)
    print("%-10.0f %10d %11.1f%% %11.2f%% %11s"
          % (p, int((nx * MESH_NM / p) ** 2), 100 * B.mean(),
             100 * (B != B20).mean(), ""))

# how much of the silicon EDGE does each pitch displace?
from scipy import ndimage
edge20 = B20 & ~ndimage.binary_erosion(B20)
print()
print("perimeter length (20 nm cells): %d" % int(edge20.sum()))
for p in (20.0, 40.0, 60.0, 120.0):
    B = quantise(p)
    e = B & ~ndimage.binary_erosion(B)
    # symmetric difference against the fine pattern, restricted to a 2-cell band
    band = ndimage.binary_dilation(edge20, iterations=6)
    print("  %6.0f nm : perimeter %6d cells,  |dA| inside a 120 nm band around the"
          " 20 nm edge = %5.1f%%"
          % (p, int(e.sum()), 100 * (B != B20)[band].mean()))
