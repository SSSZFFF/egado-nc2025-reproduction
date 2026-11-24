"""Consolidated summary of the boundary-parameterisation study."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (label, sum T, MFS violations, feasible?)
DATA = [
    ("analog TO\n(continuous)",            1.965576, None, None),
    ("20 nm grid\nraw",                    1.950978, 859, False),
    ("40 nm grid\nraw",                    1.931392, 264, False),
    ("60 nm grid\nraw",                    1.860536, 127, False),
    ("40 nm grid\nMFS-repaired",           1.524147,   8, True),
    ("60 nm grid\nMFS-repaired",           1.552486,  10, True),
    ("120 nm lattice\ncentre rule (EGADO)", 1.404275,  2, True),
    ("120 nm lattice\narea-mean rule",      1.555496,  2, True),
]

fig, ax = plt.subplots(figsize=(11.5, 6.0))
for i, (lab, fom, viol, feas) in enumerate(DATA):
    if feas is None:
        c, m = "#7f7f7f", "s"
    elif feas:
        c, m = "#2ca02c", "o"
    else:
        c, m = "#d62728", "X"
    ax.scatter(i, fom, s=190, c=c, marker=m, zorder=3, edgecolor="k", linewidth=0.6)
    txt = "%.4f" % fom
    if viol is not None:
        txt += "\n%d violations" % viol
    ax.annotate(txt, (i, fom), textcoords="offset points", xytext=(0, 16),
                ha="center", fontsize=8.5)

ax.axhspan(1.52, 1.56, color="#2ca02c", alpha=0.10, zorder=1)
ax.axhline(1.404275, color="#1f77b4", ls=":", lw=1.2, zorder=1)
ax.text(7.35, 1.412, "EGADO (centre rule)", fontsize=8, color="#1f77b4", ha="right")

ax.set_xticks(range(len(DATA)))
ax.set_xticklabels([d[0] for d in DATA], fontsize=9)
ax.set_ylabel("mode-matched FOM  ($\\Sigma T$)")
ax.set_title("Boundary parameterisation at MFS = 120 nm\n"
             "red X = violates the rule, green O = compliant, grey = reference",
             fontsize=12)
ax.set_ylim(1.35, 2.03)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out = r"G:\codex_project\inverse_design_sun\analysis-2mode\2mode_pitch_summary.png"
plt.savefig(out, dpi=150)
print("wrote", out)

print()
print("%-32s %10s %12s %s" % ("design", "sum T", "violations", "compliant"))
for lab, fom, viol, feas in DATA:
    print("%-32s %10.6f %12s %s"
          % (lab.replace("\n", " "), fom, viol if viol is not None else "-",
             {True: "yes", False: "NO", None: "-"}[feas]))
print()
print("spread of the four compliant designs: %.4f .. %.4f  (%.3f dB)"
      % (min(d[1] for d in DATA if d[3]), max(d[1] for d in DATA if d[3]),
         10 * np.log10(max(d[1] for d in DATA if d[3]) / min(d[1] for d in DATA if d[3]))))
