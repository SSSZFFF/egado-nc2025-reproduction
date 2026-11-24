"""Manufacturability-aware digitisation analysis.

Idea under test: instead of marking EVERY edge as TBD (Canny edge map dilated by
D), only mark the digital cells whose content is genuinely ambiguous -- i.e. the
120 nm cell is a MIXTURE of silicon and hole, so mapping it to one material is
lossy.  Cells that are uniformly Si or uniformly hole map faithfully and can be
kept as they are.

Also reports how much of the analog pattern violates the minimum feature size
(MFS): the paper's devices are quoted at MFS = 120 nm, while this run used
min_feature_size = 0.

usage:  python manufacturability.py <repo> <checkpoint.npz>
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import ndimage
import numpy as np

repo = os.path.abspath(sys.argv[1])
ckpt = os.path.abspath(sys.argv[2])
sys.path.insert(0, repo)
os.chdir(repo)

MESH_NM = 20.0
PIXEL_NM = 120.0
SCALE = int(PIXEL_NM / MESH_NM)
MFS_NM = 120.0

z = np.load(ckpt)
eps = z["eps"]

# same transform get_conversion_matrix applies, so indices line up
an = eps.T[:-1, :-1]
an = -1 * np.flip(an, axis=0)
an = (an - an.min()) / (an.max() - an.min())

si = an < 0.5                      # True = silicon (value 0 in the pixel convention)
ny, nx = si.shape
print("analog grid %s   silicon %.1f%%" % (si.shape, 100 * si.mean()))
print("digital grid %d x %d = %d cells of %.0f nm" % (nx // SCALE, ny // SCALE,
                                                      (nx // SCALE) * (ny // SCALE), PIXEL_NM))

# ---- per-cell composition -------------------------------------------------
frac = si.reshape(ny // SCALE, SCALE, nx // SCALE, SCALE).mean(axis=(1, 3))
clean_si = frac > 0.999
clean_hole = frac < 0.001
mixed = ~(clean_si | clean_hole)
print()
print("=== cell composition (the proposed criterion) ===")
print("  uniform silicon  (>0.999 Si) : %4d  (%.1f%%)" % (clean_si.sum(), 100 * clean_si.mean()))
print("  uniform hole     (<0.001 Si) : %4d  (%.1f%%)" % (clean_hole.sum(), 100 * clean_hole.mean()))
print("  MIXED  -> TBD                : %4d  (%.1f%%)" % (mixed.sum(), 100 * mixed.mean()))
print("  direct conversion rate       : %.2f%%" % (100 * (1 - mixed.mean())))

# relax the tolerance: treat a cell as clean if it is >=99% one material
for tol in (0.99, 0.95, 0.90, 0.80):
    cl = (frac > tol) | (frac < 1 - tol)
    print("  tolerance %.2f -> mixed %4d, conversion %.2f%%" % (
        tol, (~cl).sum(), 100 * cl.mean()))

# ---- how the Canny+D criterion compares ----------------------------------
print()
print("=== Canny + D criterion, for comparison ===")
sys.path.insert(0, repo)
from func_get_conversion_matrix import get_conversion_matrix  # noqa: E402
work = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_mfcheck")
os.makedirs(work, exist_ok=True)

canny_tbd = {}
for D in (0, 2, 4):
    m = get_conversion_matrix(analog_file=ckpt, target_folder=work,
                              pixel_size=PIXEL_NM * 1e-9, mesh_size=int(MESH_NM), buff=D)
    tbd = (m >= 0.49) & (m <= 0.51)
    canny_tbd[D] = tbd
    inter = int((tbd & mixed).sum())
    print("  D=%d : %4d TBD (%.2f%%) | overlap with mixed %4d | mixed not covered %4d | TBD that are clean %4d"
          % (D, tbd.sum(), 100 * tbd.mean(), inter, int((mixed & ~tbd).sum()),
             int((tbd & ~mixed).sum())))

# ---- minimum feature size ------------------------------------------------
print()
print("=== minimum feature size (MFS = %.0f nm) ===" % MFS_NM)
# local width via distance transform, sampled on the medial axis INSIDE silicon
dt = ndimage.distance_transform_edt(si)
inside = si & (dt > 0)
axis = np.zeros_like(si, dtype=bool)
for ax in (0, 1):
    axis |= (dt == ndimage.maximum_filter1d(dt, 3, axis=ax))
axis &= inside
widths = 2 * dt[axis] * MESH_NM
print("  silicon medial-axis samples: %d points (inside silicon only)" % widths.size)
for q in (0.5, 1, 5, 25, 50, 75, 95):
    print("     %5.1fth percentile width : %6.1f nm" % (q, np.percentile(widths, q)))
print("  narrowest silicon neck : %.1f nm  -> %s"
      % (widths.min(), "BELOW MFS" if widths.min() < MFS_NM else "ok"))
print("  medial points narrower than MFS : %.2f%%" % (100.0 * np.mean(widths < MFS_NM)))
print("  medial points narrower than 2*MFS: %.2f%%" % (100.0 * np.mean(widths < 2 * MFS_NM)))

# connected components
lbl, n = ndimage.label(si)
objs = ndimage.find_objects(lbl)
slabs = ndimage.sum(si, lbl, range(1, n + 1)) * MESH_NM ** 2
thin = 0
for i, sl in enumerate(objs):
    h = (sl[0].stop - sl[0].start) * MESH_NM
    w = (sl[1].stop - sl[1].start) * MESH_NM
    if h < MFS_NM or w < MFS_NM:
        thin += 1
print()
print("  silicon islands : %d, area %.0f .. %.0f nm^2" % (n, slabs.min(), slabs.max()))
print("    islands whose bounding box is < MFS in either direction: %d" % thin)
lbl2, n2 = ndimage.label(~si)
sz2 = ndimage.sum(~si, lbl2, range(1, n2 + 1)) * MESH_NM ** 2
lbl2, n2 = ndimage.label(~si)
sz2 = ndimage.sum(~si, lbl2, range(1, n2 + 1)) * MESH_NM ** 2
print("  hole islands    : %d, area %.0f .. %.0f nm^2" % (n2, sz2.min(), sz2.max()))
print("    holes < MFS^2 (%.0f nm^2)     : %d" % (MFS_NM ** 2, int((sz2 < MFS_NM ** 2).sum())))
print("    holes < 4*MFS^2               : %d" % int((sz2 < 4 * MFS_NM ** 2).sum()))
print("    narrowest hole equivalent width: %.1f nm"
      % (2 * ndimage.distance_transform_edt(~si)[~si].max() * MESH_NM /
         max(1, ndimage.distance_transform_edt(~si)[~si].max())))


# ---- figure --------------------------------------------------------------
def to_display(m):
    return m[::-1, :].T


fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
ext = [-1.8, 1.8, -1.8, 1.8]

ax = axes[0]
ax.imshow(to_display(si), origin="lower", cmap="bone_r", extent=ext, interpolation="nearest")
ax.set_title("analog pattern (binarised)\n%.1f%% silicon" % (100 * si.mean()), fontsize=10)
ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")

ax = axes[1]
im = ax.imshow(to_display(frac), origin="lower", cmap="coolwarm", vmin=0, vmax=1,
               extent=ext, interpolation="nearest")
ax.set_title("silicon fraction per 120 nm cell\n0 = pure hole, 1 = pure silicon", fontsize=10)
ax.set_xlabel("x (um)")
plt.colorbar(im, ax=ax, fraction=0.046)

ax = axes[2]
cls = np.where(clean_si, 0, np.where(clean_hole, 2, 1))
cmap = ListedColormap(["#1a1a1a", "#d62728", "#f2f2f2"])
ax.imshow(to_display(cls), origin="lower", cmap=cmap,
          norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), extent=ext, interpolation="nearest")
ax.set_title("composition criterion\nuniform Si %d, uniform hole %d, MIXED/TBD %d"
             % (clean_si.sum(), clean_hole.sum(), mixed.sum()), fontsize=10)
ax.set_xlabel("x (um)")
handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="k", lw=0.4) for c in ["#1a1a1a", "#d62728", "#f2f2f2"]]
fig.legend(handles, ["uniform silicon (keep)", "MIXED -> TBD (needs DBS)", "uniform hole (keep)"],
           loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Manufacturability-aware digitisation of the 2-mode MUX", fontsize=12)
plt.tight_layout(rect=[0, 0.06, 1, 0.94])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2mode_manufacturability.png")
plt.savefig(out, dpi=150)
print("\nwrote", out)
