"""Why is our D=0 conversion rate 95.9% when the paper reports 71.22%?

Measure the edge density of our analog pattern and how it responds to the Canny
parameters, then compare with what the paper's numbers imply.
"""
import os
import sys

import numpy as np
from scipy import ndimage
from skimage import feature

sys.path.insert(0, r"G:\codex_project\inverse_design_sun\EGADO-2mode")
os.chdir(r"G:\codex_project\inverse_design_sun\EGADO-2mode")

z = np.load(r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_final.npz")
eps = z["eps"]

# exactly what get_conversion_matrix does
an = eps.T[:-1, :-1]
an = -1 * np.flip(an, axis=0)
an = (an - an.min()) / (an.max() - an.min())

print("analog_pattern fed to Canny: %s   range %.3f .. %.3f" % (an.shape, an.min(), an.max()))
print("  binary fraction (<0.01 or >0.99): %.2f%%" % (100 * (np.mean(an < 0.01) + np.mean(an > 0.99))))
print("  silicon fraction (<0.01)        : %.2f%%" % (100 * np.mean(an < 0.01)))

print()
print("=== Canny edge density vs parameters ===")
ed = feature.canny(an)
print("  DEFAULT sigma=1.0 low=0.1 high=0.2 -> %d edges = %.2f%%" % (ed.sum(), 100 * ed.sum() / ed.size))
rows = []
for sig in (1.0, 2.0, 3.0):
    for lo, hi in ((0.2, 0.4), (0.1, 0.2), (0.05, 0.15), (0.01, 0.05)):
        e = feature.canny(an, sigma=sig, low_threshold=lo, high_threshold=hi)
        rows.append((sig, lo, hi, int(e.sum()), 100.0 * e.sum() / e.size))
        print("  sigma=%.1f  thr=(%.2f, %.2f) -> %5d = %5.2f%%" % (sig, lo, hi, e.sum(), 100 * e.sum() / e.size))

print()
print("=== pattern morphology (binary view) ===")
si = an < 0.01
lbl, n = ndimage.label(si)
sizes = ndimage.sum(si, lbl, range(1, n + 1))
print("  silicon: %.1f%% of area, %d connected domains, largest %.1f%% of silicon"
      % (100 * si.mean(), n, 100 * sizes.max() / si.sum()))
lbl2, n2 = ndimage.label(~si)
sizes2 = ndimage.sum(~si, lbl2, range(1, n2 + 1))
print("  holes  : %.1f%% of area, %d connected domains, largest %.1f%% of holes"
      % (100 * (~si).mean(), n2, 100 * sizes2.max() / (~si).sum()))
per = ndimage.binary_dilation(si) & ~si
print("  boundary pixels (1-px dilation ring): %d = %.2f%% of image"
      % (per.sum(), 100 * per.sum() / per.size))

print()
print("=== what the paper's numbers imply ===")
print("  paper 2-mode D=0 : 71.22%%  -> 28.78%% of the 900 digital pixels are TBD")
print("                     = 259 TBD pixels")
print("  ours   2-mode D=0 : 95.90%%  -> 4.10%%  ->  37 TBD pixels")
print()
print("  For a 6x6 stride-6 pooling, a digital pixel is TBD iff its 6x6 window")
print("  contains >=1 edge pixel.  259 TBD windows need a much denser edge map")
print("  than 37 do.")
for dens in (2, 5, 10, 20, 40):
    p_hit = 1 - (1 - dens / 100.0) ** 36
    print("     edge density %5.1f%%  ->  %.0f%% of windows hit  ->  conversion %.1f%%"
          % (dens, 100 * p_hit, 100 * (1 - p_hit)))
need = 1 - (1 - 259.0 / 900) ** (1 / 36.0)
print("  => the paper's 71.22%% implies an edge density around %.1f%%" % (100 * need))
print("     we measure %.2f%% with the code's default Canny settings" % (100 * ed.sum() / ed.size))
