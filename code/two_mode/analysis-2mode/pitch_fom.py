"""FOM vs boundary pitch, with the DRC violation count reported alongside.

We already have the two endpoints:
    20 nm  : FOM 1.951   859 DRC violations   (the raw TO output)
   120 nm : FOM 1.404     8 DRC violations   (EGADO's lattice)

This fills in 40 nm and 60 nm so the trade-off curve is complete: how much
optical performance does each extra step of boundary freedom buy, and how many
manufacturing-rule violations does it cost?

No MFS projection is applied here on purpose -- the projection is itself a
research problem (and the morphological heuristics tried so far do not converge
against KLayout's exact rules).  Reporting (violations, FOM) pairs keeps the
measurement honest and tells us whether building a proper projection is worth it.

usage:  python pitch_fom.py <repo> <checkpoint.npz> <pitch_nm> [pitch_nm ...]
"""
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(sys.argv[1])
CKPT = os.path.abspath(sys.argv[2])
PITCHES = [float(a) for a in sys.argv[3:]]
MESH_NM = 20.0
sys.path.insert(0, REPO)
os.chdir(REPO)
from func_MDM_inverse_design import MDM_InverseDesign, lumapi  # noqa: E402
from compare_quantization import attach_expansion  # noqa: E402

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
norm = (z["eps"] - z["eps"].min()) / (z["eps"].max() - z["eps"].min())
nx, ny = norm.shape

outdir = os.path.join(design.cur_path, "analysis")
os.makedirs(outdir, exist_ok=True)
scratch = os.path.join(outdir, "_close.fsp")


def quantise(pitch_nm):
    k = int(round(pitch_nm / MESH_NM))
    n = nx // k
    cells = np.zeros((n, n), dtype=bool)
    for I in range(n):
        for J in range(n):
            cells[I, J] = norm[I * k:(I + 1) * k, J * k:(J + 1) * k].mean() < 0.5
    out = np.zeros((nx, ny), dtype=bool)
    for I in range(n):
        for J in range(n):
            out[I * k:(I + 1) * k, J * k:(J + 1) * k] = cells[I, J]
    return out


print()
print("%-10s %10s %10s %10s %10s" % ("pitch", "vars", "T1 (TE0)", "T2 (TE1)", "sum T"))
summary = []
for pitch in PITCHES:
    B = quantise(pitch)
    eps2d = np.where(B, emin, emax)
    Ts = []
    for src in (1, 2):
        fsp = os.path.join(outdir, "pitch_%03.0f_src%d.fsp" % (pitch, src))
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
            log("  pitch %.0f src%d FAILED: %s: %s" % (pitch, src, type(exc).__name__, exc))
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
    nvar = int((nx * MESH_NM / pitch) ** 2)
    summary.append((pitch, nvar, Ts[0], Ts[1], np.nansum(Ts)))
    log("%-10.0f %10d %10.6f %10.6f %10.6f" % (pitch, nvar, Ts[0], Ts[1], np.nansum(Ts)))

print()
print("=== summary (DRC counts at 120 nm from pitch_study.py) ===")
DRC = {120.0: 8, 60.0: 127, 40.0: 264, 20.0: 859}
print("%-8s %9s %9s %9s %12s %10s" % ("pitch", "vars", "sum T", "vs 120nm", "DRC viol", "dB/viol"))
base_t = [s[4] for s in summary if s[0] == 120.0]
ref = base_t[0] if base_t else 1.404275
for pitch, nvar, t1, t2, tot in summary:
    d = DRC.get(pitch)
    print("%-8.0f %9d %9.6f %+8.4f dB %12s %10s"
          % (pitch, nvar, tot, 10 * np.log10(tot / ref),
             d if d is not None else "?", "%.2f" % (d / (10 * np.log10(tot / ref) * 100) * 100)
             if d and tot != ref else "-"))
np.save(os.path.join(outdir, "pitch_fom.npy"), np.array(summary))
log("saved analysis/pitch_fom.npy")
