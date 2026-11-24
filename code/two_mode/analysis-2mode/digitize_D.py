"""Digitise the TO result with several dilation coefficients D and plot them.

D is the paper's "dilation coefficient"; in this codebase it is the EG_ATD
parameter `buff` (get_conversion_matrix's kernel is scale+buff with buff//2
padding, so larger D -> wider decision area around every edge).

usage:  python digitize_D.py <repo> <checkpoint.npz> <out.png> [D ...]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

repo = os.path.abspath(sys.argv[1])
ckpt = os.path.abspath(sys.argv[2])
out_png = os.path.abspath(sys.argv[3])
Ds = [int(x) for x in sys.argv[4:]] or [0, 2, 4]

sys.path.insert(0, repo)
os.chdir(repo)

from func_get_conversion_matrix import get_conversion_matrix  # noqa: E402

work = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_digitize")
os.makedirs(work, exist_ok=True)

# reference: the analog (projected) pattern as the conversion matrix sees it
z = np.load(ckpt)
eps = z["eps"]
analog = (eps - eps.min()) / (eps.max() - eps.min())

CLASS_COLORS = ["#1a1a1a", "#d62728", "#f2f2f2", "#ffb000"]
CLASS_NAMES = ["deposit / Si  (<0.01)", "edge / decision (0.49-0.51)",
               "etch / hole  (>0.99)", "unclassified"]
cmap = ListedColormap(CLASS_COLORS)
norm_b = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)


def classify(m):
    """Mirror func_MDM_inverse_design's classification.

    It reads:  col > 0.99 -> etch_convert (flip to state 1 = hole)
               col < 0.01 -> deposit_convert (flip to state 0 = Si)
    so the conversion matrix is inverted w.r.t. eps, matching the project's
    "1 = etched, 0 = deposited" pixel convention.
    """
    c = np.full(m.shape, 3, dtype=int)
    c[m < 0.01] = 0
    c[(m >= 0.49) & (m <= 0.51)] = 1
    c[m > 0.99] = 2
    return c


print("checkpoint:", os.path.basename(ckpt))
print("analog grid %s, eps %.4f .. %.4f" % (analog.shape, eps.min(), eps.max()))

# CRITICAL: get_conversion_matrix's `pixel_size` is the DIGITAL pixel side, not
# the mesh spacing -- config_for_eg_a2d passes pixel_size=self.rect_size_x.
# Inside, scale = int(ceil(pixel_size*1e9)/mesh_size), so 120 nm -> scale 6 and
# the matrix comes out at the 30x30 digital resolution.  Passing 20 nm instead
# silently gives scale=1 and an unconverted 180x180 matrix.
RECT_SIZE_NM = 120.0
print("digital pixel %.0f nm, mesh 20 nm -> scale = %d" % (
    RECT_SIZE_NM, int(RECT_SIZE_NM / 20)))
print()

results = {}
for D in Ds:
    print("--- D = %d ---" % D)
    m = get_conversion_matrix(analog_file=ckpt, target_folder=work,
                              pixel_size=RECT_SIZE_NM * 1e-9, mesh_size=20, buff=D)
    results[D] = m
    c = classify(m)
    n = m.size
    print("   matrix %s" % (m.shape,))
    for k, name in enumerate(CLASS_NAMES):
        cnt = int(np.sum(c == k))
        print("     %-32s %4d  %5.1f%%" % (name, cnt, 100.0 * cnt / n))
    print("     -> direct conversion rate %.2f%%" % (100.0 * (n - np.sum(c == 1)) / n))
    print()

# ------------------------------------------------------------------ plot
#
# Orientation, established by checking the boundary columns against the raw
# eps grid:
#   * eps[i, j] has i = x index, j = y index -> display as-is.
#   * get_conversion_matrix does  analog.T[:-1,:-1]  then  -1*flip(axis=0),
#     so its matrix m[a, b] holds the class at  x = x[b],  y = y[N-1-a].
#     The flip is there to match the project's pixel numbering (n = i*cols + j
#     with i running along -y); to plot it in physical orientation we undo it.
def to_display(m):
    return m[::-1, :].T          # -> [x index, y index], y ascending


def edge_si_runs(col, vals, thr):
    out, s = [], None
    v = col > thr
    for i in range(len(v)):
        if v[i] and s is None:
            s = i
        if not v[i] and s is not None:
            out.append((vals[s], vals[i - 1]))
            s = None
    if s is not None:
        out.append((vals[s], vals[-1]))
    return [(round(a * 1e6, 2), round(b * 1e6, 2)) for a, b in out
            if (b - a) > 0.15e-6]


fig, axes = plt.subplots(1, len(Ds) + 1, figsize=(3.5 * (len(Ds) + 1), 4.2))

ax = axes[0]
im = ax.imshow(analog, origin="lower", cmap="bone_r", vmin=0, vmax=1,
               extent=[-1.8, 1.8, -1.8, 1.8])
ax.set_title("analog pattern from TO\n(3.6 x 3.6 um, 181 x 181 @20 nm)", fontsize=10)
ax.set_xlabel("x (um)")
ax.set_ylabel("y (um)")
plt.colorbar(im, ax=ax, fraction=0.046)

print("orientation check (silicon/deposit runs at the boundaries):")
print("  raw eps  left edge x=-1.8 :", edge_si_runs(analog[0, :], z["y"], 0.5))
print("  raw eps right edge x=+1.8 :", edge_si_runs(analog[-1, :], z["y"], 0.5))

for i, D in enumerate(Ds, start=1):
    ax = axes[i]
    c = classify(results[D])
    disp = to_display(c)
    ax.imshow(disp, origin="lower", cmap=cmap, norm=norm_b,
              extent=[-1.8, 1.8, -1.8, 1.8], interpolation="nearest")
    n = c.size
    ax.set_title("D = %d\nedge %d (%.0f%%)   unclassified %d (%.1f%%)" % (
        D, int(np.sum(c == 1)), 100.0 * np.sum(c == 1) / n,
        int(np.sum(c == 3)), 100.0 * np.sum(c == 3) / n), fontsize=10)
    ax.set_xlabel("x (um)")
    print("  D=%d  left  edge x=-1.8 : %s" % (
        D, edge_si_runs(disp[0, :] < 0.5, np.linspace(-1.8, 1.8, disp.shape[1]), 0.5)))
    print("  D=%d  right edge x=+1.8 : %s" % (
        D, edge_si_runs(disp[-1, :] < 0.5, np.linspace(-1.8, 1.8, disp.shape[1]), 0.5)))

handles = [plt.Rectangle((0, 0), 1, 1, fc=col, ec="k", lw=0.4)
           for col in CLASS_COLORS]
fig.legend(handles, CLASS_NAMES, loc="lower center", ncol=4, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Digitisation of the 2-mode MUX for different dilation coefficients D "
             "(= EG_ATD `buff`)", fontsize=12)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(out_png, dpi=150)
print("wrote", out_png)
