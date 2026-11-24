"""Measure the INITIAL optical FoM of handover schemes A and B.

The package documents A's initial state and B's initial state geometrically but
contains no optical result for either (status: "...no_FDTD_or_optical_DBS_has_been_run").
This produces the two reference numbers a handover party needs to sanity-check
their own unified model against.

Self-check: A's initialisation is EGADO's centre rule, whose sum T was measured
independently in this project as 1.404275.  If A reproduces that, the row/column
mapping used here is correct and B's number can be trusted.

usage:  python initial_fom.py <repo> <checkpoint.npz> <pkg_dir>
"""
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(sys.argv[1])
CKPT = os.path.abspath(sys.argv[2])
PKG = os.path.abspath(sys.argv[3])
S = 6                                  # 120 nm / 20 nm
sys.path.insert(0, REPO)
os.chdir(REPO)
from func_MDM_inverse_design import MDM_InverseDesign, lumapi  # noqa: E402

T0 = time.time()


def log(m):
    print("[%7.1fs] %s" % (time.time() - T0, m), flush=True)


design = MDM_InverseDesign(
    pitch=120e-9, shape="rect", rect_size=120e-9, radius=45e-9,
    z_min=-110e-9, z_max=110e-9,
    design_region_x=3.6e-6, design_region_y=3.6e-6,
    num_in_wg=2, num_out_wg=1,
    input_wg_width=0.5e-6, in_wg_pitch=1.3e-6, in_mode_list=[1, 1],
    output_wg_width=0.9e-6, out_mode_list=[1, 2],
    n_wg="Si (Silicon) - Palik", n_bg="SiO2 (Glass) - Palik",
    wavelength=1.55e-6, wavelength_range=40e-9,
    offset_x=0, offset_y=0, buff=2,
    self_define_materials=True, processor='CPU', hide_cad=True)
base = design.config_for_lsf(dir_name=design.cur_path, lsf_filename="base_TO.lsf")
base = design._force_self_define_materials(base, design.self_define_materials)
src_script = ('for (i=1:mode_num){setnamed("Source_TE_mode"+num2str(i)+"_1550", "enabled", (i==%d));}'
              'setglobalmonitor("frequency points", 11);')


def attach_expansion(f, mon, mon_exp, mode):
    f.eval('select("%s"); set("override global monitor settings", false);' % mon)
    mtype = f.getnamed(mon, "monitor type")
    f.eval("addmodeexpansion;")
    f.eval('set("name","%s");' % mon_exp)
    f.eval('setexpansion("%s","%s");' % (mon_exp, mon))
    f.eval('setnamed("%s","auto update before analysis",true);' % mon_exp)
    f.eval('setnamed("%s","override global monitor settings",false);' % mon_exp)
    f.eval('setnamed("%s","theta",0); setnamed("%s","phi",0);'
           'setnamed("%s","rotation offset",0);' % (mon_exp, mon_exp, mon_exp))
    props = ["monitor type", "x", "y", "z"]
    if mtype == "2D X-normal":
        props += ["y span", "z span"]
    for p in props:
        f.setnamed(mon_exp, p, f.getnamed(mon, p))
    f.select(mon_exp)
    f.setnamed(mon_exp, "mode selection", "user select")
    f.updatemodes(mode)


z = np.load(CKPT)
x, y, zg = z["x"], z["y"], z["z"]
emin, emax = float(z["eps_min"]), float(z["eps_max"])
nx, ny = len(x), len(y)
log("simulation grid %dx%d" % (nx, ny))


def expand(cells30):
    """30 x 30 digital cells -> nx x ny silicon mask.

    cells30 is indexed [row_y_up, column_x_right], i.e. [y, x], while the
    simulation array is indexed [x, y].  Getting this transposed produces a
    device that does not conduct at all (sum T ~ 0.04), which is what the A
    self-check is here to catch.
    """
    out = np.zeros((nx, ny), dtype=bool)
    n = cells30.shape[0]
    for i in range(nx):                      # i indexes x
        c = min(i // S, n - 1)
        for j in range(ny):                  # j indexes y
            r = min(j // S, n - 1)
            out[i, j] = cells30[r, c]
    return out


outdir = os.path.join(design.cur_path, "analysis")
os.makedirs(outdir, exist_ok=True)
scratch = os.path.join(outdir, "_close.fsp")

print()
print("%-24s %10s %10s %10s %10s" % ("scheme", "T1 (TE0)", "T2 (TE1)", "sum T", "J = sum/2"))
summary = []
for tag, sub, statefile in (
        ("A_D0", r"inputs\A_D0", "initial_state.npz"),
        ("B_curve_DRC120", r"inputs\B_curve_DRC120", "initial_state_CORRECTED.npz"),
        ("B_curve_DRC120_ORIG", r"inputs\B_curve_DRC120", "initial_state.npz")):
    st = np.load(os.path.join(PKG, sub, statefile))
    cells = st["silicon_state"]
    eps2d = np.where(expand(cells), emax, emin)
    log("%s: %d cells, silicon %.1f%%" % (tag, int(cells.sum()), 100 * cells.mean()))
    Ts = []
    for src in (1, 2):
        fsp = os.path.join(outdir, "initial_%s_src%d.fsp" % (tag, src))
        f = lumapi.FDTD(hide=design.hide_cad)
        try:
            f.eval(base + src_script % src)
            full = np.broadcast_to(eps2d[:, :, None], (nx, ny, len(zg))).copy()
            f.putv("x_geo", x)
            f.putv("y_geo", y)
            f.putv("z_geo", zg)
            f.putv("eps_geo", full)
            f.eval("addimport; importnk2(sqrt(eps_geo),x_geo,y_geo,z_geo);")
            attach_expansion(f, "fom%d" % src, "fom%d_mode_exp" % src, src)
            f.eval('save("%s");' % fsp.replace("\\", "/"))
            f.eval('addjob("%s");' % fsp.replace("\\", "/"))
            f.eval("runjobs;")
            f.eval('load("%s");' % fsp.replace("\\", "/"))
            ds = f.getresult("fom%d_mode_exp" % src, "expansion for fom%d_mode_exp" % src)
            Ts.append(float(np.mean(np.array(ds["T_forward"]))))
        except Exception as exc:
            log("  %s src%d FAILED: %s: %s" % (tag, src, type(exc).__name__, exc))
            Ts.append(float("nan"))
        finally:
            try:
                f.save(scratch)
            except Exception:
                pass
            try:
                f.close()
            except Exception:
                pass
    s = float(np.nansum(Ts))
    summary.append((tag, Ts[0], Ts[1], s, s / 2.0))
    log("%-24s %10.6f %10.6f %10.6f %10.6f" % (tag, Ts[0], Ts[1], s, s / 2.0))

print()
print("=== initial-state reference values ===")
for tag, t1, t2, s, J in summary:
    print("  %-24s T1=%.6f  T2=%.6f  sum T=%.6f   J=%.6f" % (tag, t1, t2, s, J))
print()
print("  self-check: A is EGADO's centre rule; the value measured independently in")
print("  this project for that rule was sum T = 1.404275, i.e. J = 0.702138.")
np.save(os.path.join(outdir, "initial_scheme_fom.npy"), np.array(summary, dtype=object),
        allow_pickle=True)
log("saved analysis/initial_scheme_fom.npy")
