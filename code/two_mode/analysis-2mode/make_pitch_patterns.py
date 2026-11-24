"""Build repaired patterns for each pitch and export them as .npy.

The repair does NOT have to be optimal: a crude repair gives a LOWER BOUND on
what is achievable inside the MFS-feasible set, which is all we need to decide
whether direction 3 is worth pursuing.

Three variants per pitch:
  raw      the plain quantisation
  min      KLayout-greedy repair (smallest possible local change)
  morph    greedy, then a global morphological smoothing as an escalation
"""
import os

import numpy as np

import mfs_lib as M

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_pitch")
os.makedirs(OUT, exist_ok=True)

z = np.load(r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_final.npz")
# Work directly in the SIMULATION's own grid (eps is (nx, ny) = (181, 181)),
# not the 180x180 convention used by get_conversion_matrix, otherwise the
# pattern cannot be broadcast onto x_geo / y_geo.
raw_eps = z["eps"]
rho = (raw_eps - raw_eps.min()) / (raw_eps.max() - raw_eps.min())
nx, ny = rho.shape
MESH_NM = 20.0
print("grid %dx%d" % (nx, ny))


def quantise(pitch_nm):
    k = int(round(pitch_nm / MESH_NM))
    n = nx // k
    out = np.zeros((nx, ny), dtype=bool)
    for I in range(n):
        for J in range(n):
            out[I * k:(I + 1) * k, J * k:(J + 1) * k] = \
                rho[I * k:(I + 1) * k, J * k:(J + 1) * k].mean() > 0.5
    return out


from skimage.morphology import binary_closing, binary_opening, disk  # noqa: E402

print("%-8s %-8s %-18s %-18s %s" % ("pitch", "variant", "DRC before", "DRC after", "Si %"))
records = []
for pitch in (120.0, 60.0, 40.0):
    raw = quantise(pitch)
    c0 = M.counts(raw)
    np.save(os.path.join(OUT, "p%03.0f_raw.npy" % pitch), raw)
    print("%-8.0f %-8s %-18s %-18s %.1f" % (pitch, "raw", c0, "-", 100 * raw.mean()))
    records.append((pitch, "raw", c0, c0, 100 * raw.mean()))

    rep, iters, ok = M.project(raw, max_iter=25)
    c1 = M.counts(rep)
    np.save(os.path.join(OUT, "p%03.0f_min.npy" % pitch), rep)
    print("%-8.0f %-8s %-18s %-18s %.1f   clean=%s (%d rounds)"
          % (pitch, "min", c0, c1, 100 * rep.mean(), ok, iters))
    records.append((pitch, "min", c0, c1, 100 * rep.mean()))

    mo = binary_closing(binary_opening(rep, disk(3)), disk(3))
    c2 = M.counts(mo)
    np.save(os.path.join(OUT, "p%03.0f_morph.npy" % pitch), mo)
    print("%-8.0f %-8s %-18s %-18s %.1f   clean=%s"
          % (pitch, "morph", c1, c2, 100 * mo.mean(), sum(c2) == 0))
    records.append((pitch, "morph", c1, c2, 100 * mo.mean()))
    print()

np.save(os.path.join(OUT, "records.npy"), np.array(records, dtype=object), allow_pickle=True)
print("wrote patterns to", OUT)

