"""Render the final 2-mode MUX pattern from the TO checkpoint.

usage:  python plot_result.py <npz> <convergence.txt> <out.png>
"""
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

npz, conv, out_png = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])

z = np.load(npz)
params = z["params"]
eps = z["eps"]
emin, emax = float(z["eps_min"]), float(z["eps_max"])
norm = (eps - eps.min()) / (eps.max() - eps.min())

# convergence history
rows = []
for line in open(conv, errors="replace"):
    p = [x.strip() for x in line.split(",")]
    if len(p) < 6:
        continue
    try:
        rows.append((int(p[0]), float(p[1]), float(p[-4]), float(p[-3])))
    except ValueError:
        pass
it = np.array([r[0] for r in rows])
fom = np.array([r[1] for r in rows])
beta = np.array([r[2] for r in rows])
disc = np.array([r[3] for r in rows])

fig = plt.figure(figsize=(13.5, 4.6))

# --- final projected pattern -------------------------------------------
ax = fig.add_subplot(1, 3, 1)
ax.imshow(norm.T, origin="lower", cmap="bone", vmin=0, vmax=1,
          extent=[-1.8, 1.8, -1.8, 1.8])
ax.set_title("final device  (eps, projected)\n"
             "{:.1f}% binary   disc = {:.4f}".format(
                 100 * (np.mean(norm > 0.99) + np.mean(norm < 0.01)), disc[-1]), fontsize=10)
ax.set_xlabel("x (um)")
ax.set_ylabel("y (um)")

# --- raw density --------------------------------------------------------
ax = fig.add_subplot(1, 3, 2)
im = ax.imshow(params.T, origin="lower", cmap="bone", vmin=0, vmax=1,
               extent=[-1.8, 1.8, -1.8, 1.8])
ax.set_title("raw density parameter\n(still smooth - the beta projection is what bins it)",
             fontsize=10)
ax.set_xlabel("x (um)")
fig.colorbar(im, ax=ax, fraction=0.046)

# --- convergence --------------------------------------------------------
ax = fig.add_subplot(1, 3, 3)
ax.plot(it, fom, color="#c0392b", lw=1.2, label="FOM  (= $\\Sigma$ loss)")
ax.set_xlabel("iteration")
ax.set_ylabel("FOM", color="#c0392b")
ax.tick_params(axis="y", labelcolor="#c0392b")
ax.set_yscale("log")
ax2 = ax.twinx()
ax2.plot(it, disc, color="#2471a3", lw=1.2, label="discreteness")
ax2.axhline(0.99, color="#2471a3", ls=":", lw=1)
ax2.set_ylabel("discreteness", color="#2471a3")
ax2.tick_params(axis="y", labelcolor="#2471a3")
ax2.set_ylim(0.4, 1.02)
ax.set_title("convergence: {} iters, {} beta rounds\nfinal disc {:.4f} (target 0.99)".format(
    len(rows), len(np.unique(beta)), disc[-1]), fontsize=10)

plt.tight_layout()
plt.savefig(out_png, dpi=150)
print("wrote", out_png)

print("\nnormalised eps distribution:")
for lo, hi, lab in [(0, 0.01, "etch     (<0.01)"), (0.01, 0.49, "low grey "),
                    (0.49, 0.51, "edge band"), (0.51, 0.99, "high grey"),
                    (0.99, 1.01, "deposit  (>0.99)")]:
    n = int(np.sum((norm >= lo) & (norm < hi)))
    print("  %-18s %6d  %6.2f%%" % (lab, n, 100.0 * n / norm.size))
print("  unclassified (grey, outside the edge band): %.2f%%" % (
    100.0 * (np.sum((norm > 0.01) & (norm < 0.99)) - np.sum((norm >= 0.49) & (norm <= 0.51))) / norm.size))
