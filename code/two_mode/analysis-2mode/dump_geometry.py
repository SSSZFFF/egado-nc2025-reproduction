"""Dump the geometry that base_TO.lsf produces, for one repo / configuration.

Run as a subprocess (one repo per process) so that the two copies of
func_MDM_inverse_design never collide in sys.modules.

usage:
    python dump_geometry.py <repo> <num_in_wg> <design_x_um> <design_y_um> <out_w_um> <pitch_um|none> [out.json]

Writes the JSON to the given path (default geometry.json); stdout carries the
noisy Lumerical start-up banner, so the file is the reliable channel.
"""
import json
import os
import sys

repo = sys.argv[1]
num_in_wg = int(sys.argv[2])
design_x = float(sys.argv[3]) * 1e-6
design_y = float(sys.argv[4]) * 1e-6
out_w = float(sys.argv[5]) * 1e-6
pitch_arg = sys.argv[6]
pitch = None if pitch_arg.lower() == "none" else float(pitch_arg) * 1e-6
out_path = sys.argv[7] if len(sys.argv) > 7 else "geometry.json"
# resolve before the chdir(repo) below, otherwise it lands inside the repo
out_path = os.path.abspath(out_path)

sys.path.insert(0, repo)
os.chdir(repo)

from func_MDM_inverse_design import MDM_InverseDesign, lumapi  # noqa: E402

kw = dict(pitch=120e-9, shape="rect", rect_size=120e-9, radius=45e-9,
          z_min=-110e-9, z_max=110e-9,
          design_region_x=design_x, design_region_y=design_y,
          num_in_wg=num_in_wg, num_out_wg=1,
          input_wg_width=0.5e-6, in_mode_list=[1] * num_in_wg,
          output_wg_width=out_w, out_mode_list=list(range(1, num_in_wg + 1)),
          n_wg="Si (Silicon) - Palik", n_bg="SiO2 (Glass) - Palik",
          wavelength=1.55e-6, wavelength_range=40e-9,
          offset_x=0, offset_y=0, buff=0,
          self_define_materials=True, processor='CPU', hide_cad=True)
if pitch is not None:
    kw["in_wg_pitch"] = pitch

design = MDM_InverseDesign(**kw)
script = design.config_for_lsf(dir_name=design.cur_path, lsf_filename="base_TO.lsf")

out = {"repo": repo, "leftover_placeholders": [ln.strip() for ln in script.splitlines() if "=none" in ln],
       "in_wg_pitch_substituted": None, "objects": {}}

for ln in script.splitlines():
    s = ln.strip()
    if s.startswith("in_wg_pitch="):
        out["in_wg_pitch_substituted"] = s

fdtd = lumapi.FDTD(hide=True)
try:
    fdtd.eval(script)

    def q(obj, prop):
        fdtd.eval('select("{}"); qv = get("{}");'.format(obj, prop))
        v = fdtd.getv("qv")
        try:
            return float(v)
        except (TypeError, ValueError):
            return [float(z) for z in v]

    names = ["FDTD", "opt_fields", "global_index", "cladding", "output wg1", "fom_mesh"]
    names += ["input wg {}".format(i) for i in range(1, num_in_wg + 1)]
    names += ["Source_TE_mode{}_1550".format(i) for i in range(1, num_in_wg + 1)]
    names += ["fom{}".format(i) for i in range(1, num_in_wg + 1)]

    for n in names:
        entry = {}
        for prop in ("x", "y", "x span", "y span", "z span"):
            try:
                entry[prop] = q(n, prop)
            except Exception as exc:
                entry[prop] = "ERR:{}".format(type(exc).__name__)
        out["objects"][n] = entry
finally:
    try:
        fdtd.close()
    except Exception:
        pass

with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("wrote", out_path)
