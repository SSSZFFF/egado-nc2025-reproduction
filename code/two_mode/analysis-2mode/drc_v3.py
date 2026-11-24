"""D(B) v3: Manhattan (square) structuring element.

v1 (disk opening) failed on corners and had a 2r+1 threshold.
v2 (distance-transform ridge) failed because max-EDT cannot separate a 5-cell
   from a 6-cell line at 20 nm sampling, and square-island corner pixels still
   qualify as 2-D local maxima.

v3 uses a SQUARE structuring element of side k cells.  This is the natural
discrete form of a Manhattan minimum-width / minimum-space rule -- which is what
foundries actually check -- and it does not round off axis-aligned corners:

    a feature survives opening by square(k)  <=>  it is at least k cells wide
    in BOTH axis directions somewhere

so a k-cell line passes, a (k-1)-cell line does not, and a square island of side
k passes with no corner penalty.

Known limitation (documented, not hidden): a square SE is more permissive than a
disk on 45-degree and curved boundaries, because a diagonal strip of
perpendicular width k has axis-aligned extent k*sqrt(2) > k.
"""
import numpy as np
from skimage.morphology import binary_opening, square

MESH_NM = 20.0
N = 120


def make(kind, w):
    B = np.zeros((N, N), dtype=bool)
    c = N // 2
    if kind == "silicon line":
        B[c - w // 2: c - w // 2 + w, :] = True
    elif kind == "oxide gap":
        B[:] = True
        B[c - w // 2: c - w // 2 + w, :] = False
    elif kind == "silicon island":
        B[c - w // 2: c - w // 2 + w, c - w // 2: c - w // 2 + w] = True
    elif kind == "oxide hole":
        B[:] = True
        B[c - w // 2: c - w // 2 + w, c - w // 2: c - w // 2 + w] = False
    elif kind == "diagonal strip":        # Si strip at 45 deg, perpendicular width w
        yy, xx = np.mgrid[0:N, 0:N]
        d = np.abs((yy - c) - (xx - c)) / np.sqrt(2.0)
        B[d < w / 2.0] = True
    return B


def violation_v3(B, k):
    thin_si = B & ~binary_opening(B, square(k))
    thin_ox = ~B & ~binary_opening(~B, square(k))
    return int(thin_si.sum() + thin_ox.sum())


print("=== v3 thresholds: k = 5, 6, 7 cells (= 100, 120, 140 nm) ===")
KINDS = ["silicon line", "oxide gap", "silicon island", "oxide hole"]
WIDTHS_NM = [80, 100, 120, 140, 160, 180]
for k in (5, 6, 7):
    print()
    print("  k = %d cells = %.0f nm" % (k, k * MESH_NM))
    hdr = "    %-16s" % "structure"
    for w in WIDTHS_NM:
        hdr += "%8d" % w
    print(hdr + "   (nm -> violation cells)")
    for kind in KINDS:
        row = "    %-16s" % kind
        for wnm in WIDTHS_NM:
            d = violation_v3(make(kind, wnm // 20), k)
            row += "%8s" % ("PASS" if d == 0 else str(d))
        print(row)

print()
print("=== with k = 6 (MFS = 120 nm), which is the configuration we want ===")
print("  %-16s %s" % ("structure", "verdict per width (pass = manufacturable)"))
ok = True
for kind in KINDS:
    row = "  %-16s" % kind
    for wnm in WIDTHS_NM:
        d = violation_v3(make(kind, wnm // 20), 6)
        v = d == 0
        if (wnm >= 120) != v:
            ok = False
            v = "MISMATCH"
        row += "%8s" % (v if isinstance(v, str) else ("PASS" if v else "fail"))
    print(row)
print()
print("  rectilinear rules behave correctly for every width: %s" % ok)

print()
print("=== documented limitation: diagonal features ===")
for wnm in (100, 120, 140, 160):
    d = violation_v3(make("diagonal strip", wnm // 20), 6)
    print("  45-deg strip of perpendicular width %3d nm -> violation %5d cells  (%s)"
          % (wnm, d, "pass" if d == 0 else "flagged"))
print("  a diagonal strip needs an axis-aligned extent >= 6 cells, i.e. a")
print("  perpendicular width >= 6/sqrt(2) ~ 4.24 cells = 85 nm, to pass.")
print("  -> for patterns with diagonal boundaries v3 is PERMISSIVE by up to 1/sqrt(2).")
print("  -> mitigate by checking both the pattern and its 45-deg rotation (do both).")
