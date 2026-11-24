"""Dump the properties of named mesh objects, and the global mesh accuracy.

usage:  python probe_mesh.py <repo> <fsp|base_XXX.lsf> [names...]
"""
import os
import sys

repo = os.path.abspath(sys.argv[1])
kind = sys.argv[2]
extra_names = sys.argv[3:]

sys.path.insert(0, repo)
os.chdir(repo)

from func_MDM_inverse_design import MDM_InverseDesign, lumapi  # noqa: E402

fdtd = lumapi.FDTD(hide=True)

if kind.lower().endswith(".fsp"):
    fdtd.load(os.path.abspath(kind))
    print("loaded", os.path.basename(kind))
    probe = ["mesh", "mesh_1", "mesh_2", "fom_mesh", "design_mesh"] + extra_names
else:
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
    script = design.config_for_lsf(dir_name=design.cur_path, lsf_filename=kind)
    fdtd.eval(script)
    print("built from", kind)
    probe = ["mesh", "fom_mesh", "mesh_1"] + extra_names


def exists(name):
    try:
        fdtd.eval('select("{}"); t = get("type");'.format(name))
        return str(fdtd.getv("t"))
    except Exception:
        return None


def q(name, prop):
    fdtd.eval('select("{}"); v = get("{}");'.format(name, prop))
    v = fdtd.getv("v")
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


print()
print("=== named mesh objects ===")
found = []
for nm in probe:
    t = exists(nm)
    if t is None:
        continue
    if t.lower() != "mesh":
        print("  {:<14} exists but type = {}".format(nm, t))
        continue
    found.append(nm)
    print("  {}  (type=mesh)".format(nm))
    print("     position   x={:8.3f}  y={:8.3f}  z={:8.3f}  um".format(
        q(nm, "x") * 1e6, q(nm, "y") * 1e6, q(nm, "z") * 1e6))
    print("     span       x={:8.3f}  y={:8.3f}  z={:8.3f}  um".format(
        q(nm, "x span") * 1e6, q(nm, "y span") * 1e6, q(nm, "z span") * 1e6))
    ov = [("x", "override x mesh", "dx"), ("y", "override y mesh", "dy"), ("z", "override z mesh", "dz")]
    for ax, flag, dname in ov:
        on = bool(q(nm, flag))
        print("     override {} : {:<5}  d{} = {:6.1f} nm".format(
            ax, str(on), ax, q(nm, dname) * 1e9))
if not found:
    print("  (none found)")

print()
print("=== global FDTD settings ===")
fdtd.eval('select("FDTD"); ma = get("mesh accuracy"); st = get("simulation time");'
          ' xa = get("x span"); ya = get("y span"); za = get("z span");'
          ' dxm = get("mesh cells per wavelength");')
print("  mesh accuracy          :", fdtd.getv("ma"))
print("  mesh cells/wavelength  :", fdtd.getv("dxm"))
print("  simulation time        : {:.1f} fs".format(float(fdtd.getv("st")) * 1e15))
print("  FDTD span x/y/z        : {:.2f} x {:.2f} x {:.2f} um".format(
    float(fdtd.getv("xa")) * 1e6, float(fdtd.getv("ya")) * 1e6, float(fdtd.getv("za")) * 1e6))

try:
    fdtd.close()
except Exception:
    pass
