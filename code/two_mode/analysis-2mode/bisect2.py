"""Bisect the new mesh overrides using the real in-process build path.

A failing case returns in ~3 s; a working one takes minutes, so start with the
cheap (expected-to-fail) variants.

usage:  python bisect2.py <repo> <variant>
"""
import os
import re
import sys
import time
import glob

repo = os.path.abspath(sys.argv[1])
variant = sys.argv[2]

sys.path.insert(0, repo)
os.chdir(repo)

from func_MDM_inverse_design import MDM_InverseDesign, lumapi  # noqa: E402

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
script = design.config_for_lsf(dir_name=design.cur_path, lsf_filename="base_TO.lsf")

DROP = {
    "none":      [],
    "design":    ["source_mesh1", "source_mesh2"],
    "source":    ["design_mesh"],
    "noxy_z":    ["source_mesh1", "source_mesh2"],
    "noz_both":  [],
    "fom_only":  ["design_mesh", "source_mesh1", "source_mesh2"],
}
# variants that additionally strip the z override from the kept meshes
STRIP_Z = {"noz_both": ["design_mesh"], "noxy_z": []}
KEEP_Z_OFF = {"noxy_z": ["design_mesh"]}

drop = DROP[variant]
before = set(glob.glob(os.path.join(repo, "*.log")))
fdtd = lumapi.FDTD(hide=True)
try:
    fdtd.eval(script)
    for nm in drop:
        try:
            fdtd.eval('select("{}"); delete;'.format(nm))
        except Exception as exc:
            print("  delete {} failed ({})".format(nm, exc))
    for nm in KEEP_Z_OFF.get(variant, []):
        fdtd.eval('select("{}"); set("override z mesh", false);'.format(nm))
    fdtd.eval('select("FDTD");')
    t0 = time.time()
    try:
        fdtd.eval("run;")
        raised = None
    except Exception as exc:
        raised = exc
    dt = time.time() - t0
    new_logs = sorted(set(glob.glob(os.path.join(repo, "*.log"))) - before)
    grid = ""
    for p in reversed(new_logs):
        txt = open(p, errors="replace").read()
        g = re.findall(r"Simulation size in gridpoints: (\d+ x \d+ x \d+)", txt)
        if g:
            grid = g[0]
            break
    for p in new_logs:
        try:
            os.remove(p)
        except Exception:
            pass
    print("variant={:<10} {:>7.1f} s  {}".format(
        variant, dt, ("RAISED " + str(raised)) if raised else ("grid=" + grid if grid else "NO ENGINE LOG")))
finally:
    try:
        fdtd.close()
    except Exception:
        pass
