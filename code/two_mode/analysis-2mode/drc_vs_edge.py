"""Manufacturability-guided candidate selection vs EGADO's edge-guided selection.

Implements the criterion from the discussion:

    D(B)    = total DRC violation area of the binary pattern B
    B_i^0   = B with digital pixel i forced to oxide
    B_i^1   = B with digital pixel i forced to silicon
    dD_i    = D(B) - min( D(B_i^0), D(B_i^1) )
    TBD_M   = { i : dD_i > 0 }        <- flipping i can REPAIR a violation

The DRC score uses morphological opening, which is the standard way to express a
minimum-feature-size rule:
    silicon thinner than MFS  = B  & ~opening(B,  disk(MFS/2))
    oxide   thinner than MFS  = ~B & ~opening(~B, disk(MFS/2))
D is the total area (in 20 nm mesh cells) of those two masks.

Everything is done in the same index space as get_conversion_matrix
(m[a,b]: a = y (reversed), b = x) so the sets are directly comparable.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from skimage.morphology import binary_opening, disk

MESH_NM = 20.0
PIXEL_NM = 120.0
SCALE = int(PIXEL_NM / MESH_NM)          # 6
MFS_NM = 120.0
R = int(round(MFS_NM / MESH_NM / 2))     # disk radius in mesh cells = 3

repo = r"G:\codex_project\inverse_design_sun\EGADO-2mode"
sys.path.insert(0, repo)
os.chdir(repo)
from func_get_conversion_matrix import get_conversion_matrix  # noqa: E402

work = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_drc")
os.makedirs(work, exist_ok=True)

ckpt = r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_final.npz"
z = np.load(ckpt)
an = z["eps"].T[:-1, :-1]
an = -1 * np.flip(an, axis=0)
an = (an - an.min()) / (an.max() - an.min())
B = an < 0.5                              # True = silicon
ny, nx = B.shape
NC = nx // SCALE
print("binary pattern %s  silicon %.1f%%   digital grid %dx%d"
      % (B.shape, 100 * B.mean(), NC, NC))

SE = disk(R)


def violation(mask):
    """Total area (mesh cells) that is thinner than the MFS, silicon + oxide."""
    thin_si = mask & ~binary_opening(mask, SE)
    thin_ox = ~mask & ~binary_opening(~mask, SE)
    return int(thin_si.sum()) + int(thin_ox.sum())


D0 = violation(B)
print("D(B) initial = %d mesh cells (%.0f nm^2) = %.2f%% of the design area"
      % (D0, D0 * MESH_NM ** 2, 100.0 * D0 / B.size))

# ---------------------------------------------------------------- dD_i sweep
print("\nsweeping 2 x %d pixel flips ..." % (NC * NC))
dD = np.zeros((NC, NC), dtype=int)
for i in range(NC):
    for j in range(NC):
        sl = (slice(i * SCALE, (i + 1) * SCALE), slice(j * SCALE, (j + 1) * SCALE))
        b1 = B.copy(); b1[sl] = True
        b0 = B.copy(); b0[sl] = False
        dD[i, j] = D0 - min(violation(b1), violation(b0))

TBD_M = dD > 0
print("  TBD_M (flip can repair a violation) : %d of %d (%.1f%%)"
      % (TBD_M.sum(), dD.size, 100 * TBD_M.mean()))
print("  dD distribution: " + ", ".join(
    "dD=%d:%d" % (v, int((dD == v).sum())) for v in sorted(set(dD.ravel().tolist()))[:10]))

# ---------------------------------------------------------------- EGADO sets
print("\n=== EGADO edge+D criterion (correct pixel_size = 120 nm) ===")
tbd_e = {}
for D in (0, 2, 4):
    m = get_conversion_matrix(analog_file=ckpt, target_folder=work,
                              pixel_size=PIXEL_NM * 1e-9, mesh_size=int(MESH_NM), buff=D)
    t = (m >= 0.49) & (m <= 0.51)
    tbd_e[D] = t
    print("  D=%d : %3d TBD | overlap with TBD_M %3d | TBD_M not in edge %3d | edge that repair nothing %3d"
          % (D, t.sum(), int((t & TBD_M).sum()), int((TBD_M & ~t).sum()), int((t & ~TBD_M).sum())))

# ---------------------------------------------------------------- figure
def to_display(m):
    return m[::-1, :].T


fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
ext = [-1.8, 1.8, -1.8, 1.8]

ax = axes[0]
ax.imshow(to_display(B), origin="lower", cmap="bone_r", extent=ext, interpolation="nearest")
ax.set_title("TO pattern (binarised)\n%.1f%% silicon" % (100 * B.mean()), fontsize=10)
ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")

ax = axes[1]
ax.imshow(to_display(dD), origin="lower", cmap="viridis", extent=ext, interpolation="nearest")
ax.set_title("$\\Delta D_i$ = violation removed by flipping pixel i\n"
             "(0 = flipping it repairs nothing)", fontsize=10)
ax.set_xlabel("x (um)")
plt.colorbar(ax.images[0], ax=ax, fraction=0.046)

E0 = tbd_e[0]
cls = np.zeros((NC, NC), dtype=int)
cls[TBD_M & ~E0] = 1          # manufacturability only
cls[TBD_M & E0] = 2           # both
cls[~TBD_M & E0] = 3          # edge only
cmap = ListedColormap(["#f0f0f0", "#d62728", "#7b3294", "#1f77b4"])
ax = axes[2]
ax.imshow(to_display(cls), origin="lower", cmap=cmap,
          norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4), extent=ext, interpolation="nearest")
ax.set_title("TBD_M vs EGADO D=0\nM-only %d, both %d, E-only %d"
             % (int((TBD_M & ~E0).sum()), int((TBD_M & E0).sum()), int((~TBD_M & E0).sum())),
             fontsize=10)
ax.set_xlabel("x (um)")
handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="k", lw=0.4) for c in
           ["#f0f0f0", "#d62728", "#7b3294", "#1f77b4"]]
fig.legend(handles, ["neither (direct map)", "manufacturability only",
                     "both criteria agree", "EGADO edge only (repairs nothing)"],
           loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

ax = axes[3]
labels = ["EGADO D=0", "EGADO D=2", "EGADO D=4", "TBD_M\n(manufacturability)"]
counts = [int(tbd_e[0].sum()), int(tbd_e[2].sum()), int(tbd_e[4].sum()), int(TBD_M.sum())]
colors = ["#1f77b4", "#1f77b4", "#1f77b4", "#d62728"]
bars = ax.bar(labels, counts, color=colors)
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, c + 4, str(c), ha="center", fontsize=10)
ax.set_ylabel("TBD pixels  (= 2 FDTD solves each)")
ax.set_title("candidate-set size", fontsize=10)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Manufacturability-guided vs edge-guided TBD selection  (2-mode MUX)", fontsize=12)
plt.tight_layout(rect=[0, 0.06, 1, 0.93])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2mode_drc_vs_edge.png")
plt.savefig(out, dpi=150)
print("\nwrote", out)

np.save(os.path.join(work, "dD.npy"), dD)
