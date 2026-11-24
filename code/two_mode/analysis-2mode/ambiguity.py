"""Quantization-ambiguity analysis (no FDTD).

Reframes the candidate-set question the way the paper actually poses it:

    20 nm analog pattern -> 120 nm digital pixel is a POOLING / requantisation.
    A digital cell that straddles a material boundary cannot be mapped to one
    material without representation error -> TBD.

Per 120 nm cell i (6x6 = 36 analog cells):

    p_i = N_Si / 36                          silicon fraction
    A_i = 2*min(p_i, 1-p_i) = 1-|2p_i-1|     ambiguity, 0 = certain, 1 = 50/50
    Q_i^0 = sum (rho_j - 0)^2                quantisation error if forced oxide
    Q_i^1 = sum (rho_j - 1)^2                quantisation error if forced silicon
    C_i   = |Q_i^1 - Q_i^0| / (Q_i^1+Q_i^0)  confidence (1 - A_i for binary rho)

Candidate set:  C(tau) = { i : A_i > tau }.

Baseline for comparison is EGADO D=0 (238 TBD) -- D=2/4 are the author's own
deliberate enlargements of the search window and are not the right baseline.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

MESH_NM = 20.0
PIXEL_NM = 120.0
S = int(PIXEL_NM / MESH_NM)          # 6

repo = r"G:\codex_project\inverse_design_sun\EGADO-2mode"
sys.path.insert(0, repo)
os.chdir(repo)
from func_get_conversion_matrix import get_conversion_matrix  # noqa: E402

work = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_amb")
os.makedirs(work, exist_ok=True)
ckpt = r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_final.npz"

z = np.load(ckpt)
eps = z["eps"]
an = eps.T[:-1, :-1]
an = -1 * np.flip(an, axis=0)
rho = (an - an.min()) / (an.max() - an.min())      # the array digitisation sees
ny, nx = rho.shape
NC = nx // S
print("analog grid %s  ->  digital %dx%d = %d cells" % (rho.shape, NC, NC, NC * NC))
print("rho: binary fraction (<0.01 or >0.99) = %.2f%%" % (100 * (np.mean(rho < 0.01) + np.mean(rho > 0.99))))

# ---- per-cell statistics -------------------------------------------------
p = rho.reshape(NC, S, NC, S).mean(axis=(1, 3))          # mean rho = Si fraction
p_si = (rho < 0.5).reshape(NC, S, NC, S).mean(axis=(1, 3))  # binarised Si fraction
A = 1.0 - np.abs(2 * p - 1)
Q0 = (rho ** 2).reshape(NC, S, NC, S).sum(axis=(1, 3))
Q1 = ((rho - 1) ** 2).reshape(NC, S, NC, S).sum(axis=(1, 3))
C = np.abs(Q1 - Q0) / (Q1 + Q0 + 1e-12)

mixed = (p > 1e-6) & (p < 1 - 1e-6)
print()
print("=== cell composition ===")
print("  uniform silicon (p>0.999999) : %4d" % int((p >= 1 - 1e-6).sum()))
print("  uniform oxide   (p<0.000001) : %4d" % int((p <= 1e-6).sum()))
print("  mixed                        : %4d" % int(mixed.sum()))

print()
print("=== the key question: how ambiguous are the mixed cells? ===")
Am = A[mixed]
print("  A over the %d mixed cells:" % mixed.sum())
for lo, hi in [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 0.5),
               (0.5, 0.65), (0.65, 0.8), (0.8, 0.9), (0.9, 1.01)]:
    n = int(((Am >= lo) & (Am < hi)).sum())
    bar = "#" * int(60 * n / max(1, mixed.sum()))
    print("    A in [%.2f,%.2f) : %4d  %5.1f%%  %s" % (lo, hi, n, 100.0 * n / mixed.sum(), bar))
print("  A median %.3f, mean %.3f, max %.3f" % (np.median(Am), Am.mean(), Am.max()))

print()
print("=== threshold sweep:  N_TBD(tau) = |{A_i > tau}| ===")
print("  %-8s %-8s %s" % ("tau", "N_TBD", "vs EGADO D=0 (238)"))
taus = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
Nt = []
for t in taus:
    n = int((A > t + 1e-12).sum())
    Nt.append(n)
    print("  %-8.2f %-8d %+.1f%%" % (t, n, 100.0 * (n - 238) / 238.0))

print()
print("=== Canny D=0 baseline and what it selects ===")
m0 = get_conversion_matrix(analog_file=ckpt, target_folder=work,
                           pixel_size=PIXEL_NM * 1e-9, mesh_size=int(MESH_NM), buff=0)
E0 = (m0 >= 0.49) & (m0 <= 0.51)
print("  Canny D=0 TBD      : %d" % E0.sum())
print("  mixed              : %d" % mixed.sum())
print("  overlap            : %d" % int((E0 & mixed).sum()))
print("  Canny but NOT mixed: %d   <- these are pruned by the ambiguity criterion" % int((E0 & ~mixed).sum()))
print("  mixed but NOT Canny: %d" % int((mixed & ~E0).sum()))
if (E0 & ~mixed).sum():
    print("  their A values     : %s" % np.round(np.sort(A[E0 & ~mixed]), 4).tolist())
    print("  their p values     : %s" % np.round(np.sort(p[E0 & ~mixed]), 4).tolist())
if (mixed & ~E0).sum():
    print("  mixed-not-Canny A  : %s" % np.round(np.sort(A[mixed & ~E0]), 4).tolist())

# ------------------------------------------------------------------ figure
def to_display(m):
    return m[::-1, :].T


fig = plt.figure(figsize=(17, 7.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.42, wspace=0.28)

ax = fig.add_subplot(gs[0, 0])
ax.hist(p.ravel(), bins=np.linspace(0, 1, 61), color="#888888", edgecolor="none")
ax.hist(p[mixed].ravel(), bins=np.linspace(0, 1, 61), color="#d62728", edgecolor="none")
ax.set_yscale("log")
ax.set_xlabel("$p_i$  (silicon fraction of a 120 nm cell)")
ax.set_ylabel("cells (log)")
ax.set_title("$p_i$ distribution\nred = the %d mixed cells" % mixed.sum(), fontsize=10)
for x in (0.1, 0.25, 0.5):
    ax.axvline(x, color="k", ls=":", lw=0.8)
    ax.text(x, ax.get_ylim()[1] * 0.4, "%.2f" % x, fontsize=7, rotation=90)

ax = fig.add_subplot(gs[0, 1])
ax.hist(A[mixed], bins=np.linspace(0, 1, 41), color="#d62728", edgecolor="none")
ax.set_xlabel("$A_i = 1-|2p_i-1|$   (0 = certain, 1 = 50/50)")
ax.set_ylabel("mixed cells")
ax.set_title("ambiguity of the mixed cells\nmedian %.3f, >0.5: %d, >0.2: %d"
             % (np.median(Am), int((Am > 0.5).sum()), int((Am > 0.2).sum())), fontsize=10)

ax = fig.add_subplot(gs[0, 2])
ax.plot(taus, Nt, "o-", color="#d62728", label=r"$N_{TBD}(\tau)$")
ax.axhline(238, color="#1f77b4", ls="--", label="EGADO D=0 (238)")
ax.axhline(308, color="#1f77b4", ls=":", label="EGADO D=2 (308)")
ax.axhline(205, color="#2ca02c", ls="-.", label="mixed = 205")
ax.set_xlabel(r"ambiguity threshold $\tau$")
ax.set_ylabel("candidate pixels")
ax.set_title("threshold sweep", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

for k, (mask, title) in enumerate([
        (E0, "EGADO D=0  (%d)" % E0.sum()),
        (mixed, "mixed  (%d)" % mixed.sum()),
        (A > 0.5, r"$A_i>0.5$  (%d)" % int((A > 0.5).sum()))]):
    ax = fig.add_subplot(gs[1, k])
    ax.imshow(to_display(A), origin="lower", cmap="magma", vmin=0, vmax=1,
              extent=[-1.8, 1.8, -1.8, 1.8], interpolation="nearest", alpha=0.85)
    ov = mask & ~(A > 0.5) if k != 2 else mask
    ax.contour(to_display(mask).astype(float), levels=[0.5], colors=["#00ff00"], linewidths=1.4,
               extent=[-1.8, 1.8, -1.8, 1.8])
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Quantization-ambiguity analysis of the 2-mode MUX TO result", fontsize=13)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "2mode_ambiguity.png")
plt.savefig(out, dpi=140, bbox_inches="tight")
print("\nwrote", out)
