"""Step 1: sanity-check the DRC metric D(B) against known feature widths.

Builds synthetic binary patterns with a known feature width w and asks whether
D(B) reports a violation.  The metric must switch from FAIL to PASS at w = MFS
(120 nm here).

Key subtlety being tested: a DISCRETE disk structuring element of radius r only
lets features of width >= 2r+1 *pixels* survive an opening, whereas the
continuous rule is width >= 2r.  At 20 nm sampling, disk(3) therefore tests
>= 7 cells = 140 nm, not 120 nm.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.morphology import binary_opening, disk

MESH_NM = 20.0
MFS_NM = 120.0
N = 120                                  # canvas 120 x 120 cells = 2.4 x 2.4 um
WIDTHS_NM = [80, 100, 120, 140, 160, 180]
WIDTHS = [w // int(MESH_NM) for w in WIDTHS_NM]


def make(kind, w):
    """w in mesh cells. Returns boolean array, True = silicon."""
    B = np.zeros((N, N), dtype=bool)
    c = N // 2
    if kind == "silicon line":            # a horizontal Si wire in oxide
        B[c - w // 2: c - w // 2 + w, :] = True
    elif kind == "oxide gap":             # oxide slit through silicon
        B[:] = True
        B[c - w // 2: c - w // 2 + w, :] = False
    elif kind == "silicon island":        # square Si island in oxide
        B[c - w // 2: c - w // 2 + w, c - w // 2: c - w // 2 + w] = True
    elif kind == "oxide hole":            # square hole in silicon
        B[:] = True
        B[c - w // 2: c - w // 2 + w, c - w // 2: c - w // 2 + w] = False
    elif kind == "diagonal neck":         # Si everywhere except two diagonal slits
        B[:] = True
        yy, xx = np.mgrid[0:N, 0:N]
        for s in (+1, -1):
            d = np.abs((yy - c) - s * (xx - c)) / np.sqrt(2)
            B[d < w / 2.0] = False
    return B


def violation(mask, r):
    thin_si = mask & ~binary_opening(mask, disk(r))
    thin_ox = ~mask & ~binary_opening(~mask, disk(r))
    return int(thin_si.sum() + thin_ox.sum())


KINDS = ["silicon line", "oxide gap", "silicon island", "oxide hole", "diagonal neck"]
RADII = [2, 3, 4]

print("MFS = %.0f nm, mesh = %.0f nm  ->  a feature of width w occupies w/20 cells"
      % (MFS_NM, MESH_NM))
print("continuous rule: pass iff w >= 2r;  discrete rule: pass iff w >= 2r+1 cells")
print()

for r in RADII:
    print("=== disk radius r = %d cells = %.0f nm  (continuous threshold %.0f nm, "
          "discrete threshold %.0f nm) ==="
          % (r, r * MESH_NM, 2 * r * MESH_NM, (2 * r + 1) * MESH_NM))
    hdr = "  %-16s" % "structure"
    for w in WIDTHS_NM:
        hdr += "%8d" % w
    print(hdr + "   (nm  ->  D(B) violation cells)")
    for kind in KINDS:
        row = "  %-16s" % kind
        for w in WIDTHS:
            d = violation(make(kind, w), r)
            row += "%8s" % ("PASS" if d == 0 else str(d))
        print(row)
    print()

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
r_check = 3
for ax, kind in zip(axes.ravel(), KINDS):
    ds = [violation(make(kind, w), r_check) for w in WIDTHS]
    ax.bar([str(w) for w in WIDTHS_NM], ds,
           color=["#d62728" if d else "#2ca02c" for d in ds])
    ax.set_title("%s\n(r=%d cells)" % (kind, r_check), fontsize=10)
    ax.set_xlabel("feature width (nm)")
    ax.set_ylabel("D(B)  violation cells")
    ax.axvline(2.5, color="k", ls=":", lw=1)
    ax.text(2.55, ax.get_ylim()[1] * 0.9, "MFS = 120 nm", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
axes.ravel()[-1].axis("off")
fig.suptitle("D(B) sanity check: green = reported manufacturable, red = violation", fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_drc_sanity.png"
plt.savefig(out, dpi=140)
print("wrote", out)
