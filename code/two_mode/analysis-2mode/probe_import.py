"""Minimal probe: can we inject an arbitrary permittivity pattern and run it?

Replicates lumopt's TopologyOptimization3DLayered.update_geometry():
    putv x_geo / y_geo / z_geo / eps_geo
    addimport; importnk2(sqrt(eps_geo), x_geo, y_geo, z_geo)

Prints a marker after every step so a hang can be localised.
"""
import os
import sys
import time

import numpy as np

REPO = r"F:\codex_project\inverse_design_sun\EGADO-2mode"
CKPT = os.path.join(REPO, r"TO\x3600_y3600_f0239_m0000_0\parameters_248.npz")

sys.path.insert(0, REPO)
os.chdir(REPO)


def step(msg):
    print("[%7.1fs] %s" % (time.time() - T0, msg), flush=True)


T0 = time.time()
from func_MDM_inverse_design import MDM_InverseDesign, lumapi  # noqa: E402
step("imported module")

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
step("design object built")

s = design.config_for_lsf(dir_name=design.cur_path, lsf_filename="base_TO.lsf")
s = design._force_self_define_materials(s, design.self_define_materials)
step("base script built (%d chars)" % len(s))

z = np.load(CKPT)
x, y, zg = z["x"], z["y"], z["z"]
norm = (z["eps"] - z["eps"].min()) / (z["eps"].max() - z["eps"].min())
eps2d = np.where(norm < 0.5, float(z["eps_min"]), float(z["eps_max"]))
full = np.broadcast_to(eps2d[:, :, None], (len(x), len(y), len(zg))).copy()
step("pattern ready %s, silicon %.1f%%" % (full.shape, 100 * np.mean(norm >= 0.5)))

scratch = os.path.join(design.cur_path, "analysis", "_probe_close.fsp")
f = lumapi.FDTD(hide=True)
step("session opened")
try:
    f.eval(s + 'for (i=1:mode_num){setnamed("Source_TE_mode"+num2str(i)+"_1550", "enabled", (i==1));}'
              'setglobalmonitor("frequency points", 11);')
    step("base script evaluated")

    # what is selected right now?  (guards against a stray delete hitting it)
    f.eval('sel=get("name");')
    step("currently selected object: %s" % f.getv("sel"))

    f.putv("x_geo", x)
    f.putv("y_geo", y)
    f.putv("z_geo", zg)
    f.putv("eps_geo", full)
    step("putv done")

    f.eval("addimport;")
    f.eval('nm=get("name"); tp=get("type");')
    step("addimport OK -> new object name=%s type=%s" % (f.getv("nm"), f.getv("tp")))

    f.eval("importnk2(sqrt(eps_geo),x_geo,y_geo,z_geo);")
    step("importnk2 OK")

    f.eval('select("FDTD"); run;')
    step("RUN finished")

    t = f.get_mode_transmission("fom1_mode_exp")[0, 1, :]
    step("T1 = %.6f  (mean over %d wavelengths)" % (float(np.mean(t)), len(t)))
except Exception as exc:
    import traceback
    step("FAILED: %s: %s" % (type(exc).__name__, exc))
    traceback.print_exc()
finally:
    try:
        f.save(scratch)
        step("saved scratch")
    except Exception as exc:
        step("save failed: %s" % exc)
    try:
        f.close()
        step("closed")
    except Exception as exc:
        step("close failed: %s" % exc)
step("done")
