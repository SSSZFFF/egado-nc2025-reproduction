"""How much does the 120 nm lattice itself cost?   (mode-matched FOM)

All four variants come from the SAME TO checkpoint and are simulated identically:

  analog           the raw TO permittivity (continuous, 20 nm grid)
  binary20         thresholded Si/SiO2 on the 20 nm grid -- 20 nm boundary freedom
  pixel120_centre  EGADO's direct mapping: the 120 nm cell takes the value at the
                   centre of its 6x6 block
  pixel120_mean    better quantiser: the 120 nm cell takes the block MEAN

(binary20 vs pixel120_*) isolates the lattice cost: same source, same
binarisation, both DRC-clean -- only the boundary quantisation differs.

The FOM is the MODE-MATCHED transmission, computed the way lumopt does it:
an expansion monitor is attached to each power monitor and T_forward is read
from `getresult(<exp monitor>, "expansion for <exp monitor>")`.

Three machine-specific facts learned the hard way:
  * a bare `run;` blocks forever in an API session -- use save/addjob/runjobs
  * lumopt adds the expansion monitors at run time, so a plain base-script
    session has only the 2D power monitors `fom1` / `fom2`
  * the expansion result is named after the EXPANSION monitor, not "Output"

usage:  python compare_quantization.py <repo> <checkpoint.npz>
"""
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(sys.argv[1])
CKPT = os.path.abspath(sys.argv[2])
S = 6
sys.path.insert(0, REPO)
os.chdir(REPO)
from func_MDM_inverse_design import MDM_InverseDesign, lumapi  # noqa: E402

T0 = time.time()


def log(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


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
    """Replicate lumopt ModeMatch.add_mode_expansion_monitor (modematch.py:140)."""
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
    elif mtype == "2D Y-normal":
        props += ["x span", "z span"]
    elif mtype == "2D Z-normal":
        props += ["x span", "y span"]
    for p in props:                                   # synchronise geometry
        f.setnamed(mon_exp, p, f.getnamed(mon, p))
    f.select(mon_exp)
    f.setnamed(mon_exp, "mode selection", "user select")
    f.updatemodes(mode)                               # 1-based mode index
    return mtype


# ------------------------------------------------------------------ patterns
z = np.load(CKPT)
x, y, zg = z["x"], z["y"], z["z"]
emin, emax = float(z["eps_min"]), float(z["eps_max"])
norm = (z["eps"] - z["eps"].min()) / (z["eps"].max() - z["eps"].min())
nx, ny = norm.shape
NI, NJ = nx // S, ny // S
sil20 = norm < 0.5

cell_c = np.zeros((NI, NJ), dtype=bool)
cell_m = np.zeros((NI, NJ), dtype=bool)
for I in range(NI):
    for J in range(NJ):
        blk = norm[I * S:(I + 1) * S, J * S:(J + 1) * S]
        cell_c[I, J] = blk[(S - 1) // 2, (S - 1) // 2] < 0.5
        cell_m[I, J] = blk.mean() < 0.5


def expand(cells):
    e = np.zeros((nx, ny), dtype=bool)
    for I in range(NI):
        for J in range(NJ):
            e[I * S:(I + 1) * S, J * S:(J + 1) * S] = cells[I, J]
    return e


VARIANTS = [("analog", z["eps"].astype(float)),
            ("binary20", np.where(sil20, emin, emax)),
            ("pixel120_centre", np.where(expand(cell_c), emin, emax)),
            ("pixel120_mean", np.where(expand(cell_m), emin, emax))]
log("patterns built; centre vs mean differ in %d/%d cells"
    % (int((cell_c != cell_m).sum()), NI * NJ))

outdir = os.path.join(design.cur_path, "analysis")
os.makedirs(outdir, exist_ok=True)
scratch = os.path.join(outdir, "_close.fsp")

results, power = {}, {}
for name, eps2d in VARIANTS:
    Ts, Ps = [], []
    for src in (1, 2):
        fsp = os.path.join(outdir, "fom_%s_src%d.fsp" % (name, src))
        f = lumapi.FDTD(hide=design.hide_cad)
        try:
            f.eval(base + src_script % src)
            full = np.broadcast_to(eps2d[:, :, None], (len(x), len(y), len(zg))).copy()
            f.putv("x_geo", x)
            f.putv("y_geo", y)
            f.putv("z_geo", zg)
            f.putv("eps_geo", full)
            f.eval("addimport; importnk2(sqrt(eps_geo),x_geo,y_geo,z_geo);")

            mon, mon_exp = "fom%d" % src, "fom%d_mode_exp" % src
            mtype = attach_expansion(f, mon, mon_exp, src)      # mode i for channel i
            if src == 1 and name == "analog":
                log("    expansion monitor attached to %s (%s, mode %d)" % (mon, mtype, src))

            f.eval('save("%s");' % fsp.replace("\\", "/"))
            f.eval('addjob("%s");' % fsp.replace("\\", "/"))
            f.eval("runjobs;")
            f.eval('load("%s");' % fsp.replace("\\", "/"))

            rname = "expansion for " + mon_exp
            ds = f.getresult(mon_exp, rname)
            Tf = np.array(ds["T_forward"])
            if src == 1 and name == "analog":
                log("    %s keys=%s  T_forward shape=%s" % (rname, list(ds.keys()), Tf.shape))
            Ts.append(float(np.mean(Tf)))
            ds2 = f.getresult(mon, "T")
            Ps.append(float(np.mean(np.array(ds2["T"]))))
        except Exception as exc:
            log("  %s src%d FAILED: %s: %s" % (name, src, type(exc).__name__, exc))
            Ts.append(float("nan"))
            Ps.append(float("nan"))
        finally:
            try:
                f.save(scratch)
            except Exception:
                pass
            try:
                f.close()
            except Exception:
                pass
    results[name] = Ts
    power[name] = Ps
    log("%-16s  T1 = %.6f  T2 = %.6f  sum = %.6f   (power %.4f / %.4f)"
        % (name, Ts[0], Ts[1], np.nansum(Ts), Ps[0], Ps[1]))

# ------------------------------------------------------------------ report
print()
print("=== mode-matched FOM ===")
print("  %-16s %10s %10s %10s" % ("variant", "T1 (TE0)", "T2 (TE1)", "sum T"))
for name, _ in VARIANTS:
    print("  %-16s %10.6f %10.6f %10.6f"
          % (name, results[name][0], results[name][1], np.nansum(results[name])))

a, b = np.nansum(results["analog"]), np.nansum(results["binary20"])
print()
print("=== loss decomposition (mode-matched) ===")
print("  binarisation (analog -> binary20)      : %+.4f dB  (%.2f%%)"
      % (-10 * np.log10(b / a), 100 * (1 - b / a)))
for k in ("pixel120_centre", "pixel120_mean"):
    c = np.nansum(results[k])
    print("  120 nm lattice (binary20 -> %-14s): %+.4f dB  (%.2f%%)"
          % (k + ")", -10 * np.log10(c / b), 100 * (1 - c / b)))
print()
print("  per channel, lattice cost with the centre quantiser:")
for i, lab in enumerate(("TE0 (ch1)", "TE1 (ch2)")):
    print("     %-10s : %+.4f dB" % (lab, -10 * np.log10(results["pixel120_centre"][i]
                                                       / results["binary20"][i])))
print("  recovered by the area-mean quantiser (free, no FDTD):")
for i, lab in enumerate(("TE0 (ch1)", "TE1 (ch2)")):
    print("     %-10s : %+.4f dB" % (lab, -10 * np.log10(results["pixel120_mean"][i]
                                                       / results["pixel120_centre"][i])))
np.save(os.path.join(outdir, "quantization_fom.npy"),
        np.array([results[k] for k, _ in VARIANTS]))
np.save(os.path.join(outdir, "quantization_power.npy"),
        np.array([power[k] for k, _ in VARIANTS]))
log("saved analysis/quantization_fom.npy")
