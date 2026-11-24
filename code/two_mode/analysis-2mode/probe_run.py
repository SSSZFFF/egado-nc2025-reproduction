"""Is `run` broken in general right now, or only when the Import object exists?

Two cases, each with an explicit wall-clock report:
   plain  : base script only
   import : base script + addimport/importnk2 with the TO pattern

Whichever finishes fast without an engine log means `run` silently did nothing.
"""
import os
import sys
import time

import numpy as np

REPO = r"F:\codex_project\inverse_design_sun\EGADO-2mode"
CKPT = os.path.join(REPO, r"TO\x3600_y3600_f0239_m0000_0\parameters_248.npz")
case = sys.argv[1] if len(sys.argv) > 1 else "plain"

sys.path.insert(0, REPO)
os.chdir(REPO)
from func_MDM_inverse_design import MDM_InverseDesign, lumapi  # noqa: E402

T0 = time.time()
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
s = design.config_for_lsf(dir_name=design.cur_path, lsf_filename="base_TO.lsf")
s = design._force_self_define_materials(s, design.self_define_materials)

scratch = os.path.join(design.cur_path, "analysis", "_probe_close.fsp")
print("case =", case, flush=True)
f = lumapi.FDTD(hide=True)
try:
    f.eval(s + 'for (i=1:mode_num){setnamed("Source_TE_mode"+num2str(i)+"_1550", "enabled", (i==1));}')
    print("[%6.1fs] script evaluated" % (time.time() - T0), flush=True)

    if case == "import":
        z = np.load(CKPT)
        x, y, zg = z["x"], z["y"], z["z"]
        norm = (z["eps"] - z["eps"].min()) / (z["eps"].max() - z["eps"].min())
        eps2d = np.where(norm < 0.5, float(z["eps_min"]), float(z["eps_max"]))
        full = np.broadcast_to(eps2d[:, :, None], (len(x), len(y), len(zg))).copy()
        f.putv("x_geo", x); f.putv("y_geo", y); f.putv("z_geo", zg); f.putv("eps_geo", full)
        f.eval("addimport; importnk2(sqrt(eps_geo),x_geo,y_geo,z_geo);")
        print("[%6.1fs] import added" % (time.time() - T0), flush=True)

    t0 = time.time()
    if case != "plain":
        # the project's own mechanism (cost_function): save -> addjob -> runjobs.
        # A bare `run;` blocks forever in this API session; runjobs works.
        fsp = os.path.join(design.cur_path, "analysis", "_probe_job.fsp")
        f.eval('save("%s");' % fsp.replace("\\", "/"))
        print("[%6.1fs] saved fsp" % (time.time() - T0), flush=True)
        f.eval('addjob("%s");' % fsp.replace("\\", "/"))
        f.eval("runjobs;")
        print("[%6.1fs] runjobs returned after %.1f s" % (time.time() - T0, time.time() - t0), flush=True)
        f.eval('load("%s");' % fsp.replace("\\", "/"))
        print("[%6.1fs] reloaded results" % (time.time() - T0), flush=True)
    else:
        f.eval('select("FDTD"); run;')
        print("[%6.1fs] run returned after %.1f s" % (time.time() - T0, time.time() - t0), flush=True)

    # splayout's modeapi uses:  getresult(name, "expansion for Output") -> data.T_forward
    f.eval('data = getresult("fom1_mode_exp","expansion for Output");')
    f.eval('Tfw = data.T_forward;')
    T = np.array(f.getv("Tfw"))
    print("          T_forward shape %s   mean %.6f" % (T.shape, float(np.mean(T))), flush=True)
except Exception as exc:
    print("FAILED: %s: %s" % (type(exc).__name__, exc), flush=True)
finally:
    try:
        f.save(scratch)
    except Exception:
        pass
    try:
        f.close()
    except Exception:
        pass
print("done", flush=True)
