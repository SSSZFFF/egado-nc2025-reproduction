"""Simulate the repaired pitch patterns and report the mode-matched FOM.

usage:  python sim_pitch_patterns.py <repo> <checkpoint.npz> <pattern.npy> [...]
Each pattern is a boolean array on the 20 nm grid (True = silicon).
"""
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(sys.argv[1])
CKPT = os.path.abspath(sys.argv[2])
PATTERNS = [os.path.abspath(p) for p in sys.argv[3:]]
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

z = np.load(CKPT)
x, y, zg = z["x"], z["y"], z["z"]
emin, emax = float(z["eps_min"]), float(z["eps_max"])
nx, ny = z["eps"].shape if z["eps"].ndim == 2 else z["eps"].T[:-1, :-1].shape


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


outdir = os.path.join(design.cur_path, "analysis")
os.makedirs(outdir, exist_ok=True)
scratch = os.path.join(outdir, "_close.fsp")

print()
print("%-28s %10s %10s %10s" % ("pattern", "T1 (TE0)", "T2 (TE1)", "sum T"))
summary = []
for path in PATTERNS:
    B = np.load(path)
    if B.shape != (nx, ny):
        B = B.T
    # B is True = SILICON, i.e. HIGH permittivity -> emax
    eps2d = np.where(B, emax, emin)
    Ts = []
    for src in (1, 2):
        stem = os.path.splitext(os.path.basename(path))[0]
        fsp = os.path.join(outdir, "%s_src%d.fsp" % (stem, src))
        f = lumapi.FDTD(hide=design.hide_cad)
        try:
            f.eval(base + src_script % src)
            full = np.broadcast_to(eps2d[:, :, None], (len(x), len(y), len(zg))).copy()
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
            log("  %s src%d FAILED: %s: %s" % (stem, src, type(exc).__name__, exc))
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
    stem = os.path.splitext(os.path.basename(path))[0]
    summary.append((stem, Ts[0], Ts[1], np.nansum(Ts)))
    log("%-28s %10.6f %10.6f %10.6f" % (stem, Ts[0], Ts[1], np.nansum(Ts)))

np.save(os.path.join(outdir, "pitch_repaired_fom.npy"), np.array(summary, dtype=object),
        allow_pickle=True)
log("saved analysis/pitch_repaired_fom.npy")
