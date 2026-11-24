"""Generate the TO base .fsp files exactly the way the project does it.

Mirrors config_for_eg_a2d_fdtd_file():
    base_script  = config_for_lsf(...)
    base_script  = _force_self_define_materials(base_script, self_define_materials)
    for each mode i:
        temp_fdtd = lumapi.FDTD(hide=hide_cad)
        temp_fdtd.putv("current_source", i+1)
        temp_fdtd.eval(base_script + 'for(...){setnamed("Source_...", "enabled", (i==current_source));}'
                                    'setglobalmonitor("frequency points", 11);')
        temp_fdtd.save(<target>/TE_mode_<i>_TO.fsp)

usage:  python make_fsp.py <repo> [suffix]
"""
import os
import sys

repo = os.path.abspath(sys.argv[1])
suffix = sys.argv[2] if len(sys.argv) > 2 else "TO"

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

target_folder = os.path.join(design.cur_path, "{}_base".format(suffix))
os.makedirs(target_folder, exist_ok=True)

base_script = design.config_for_lsf(dir_name=design.cur_path, lsf_filename="base_TO.lsf")
base_script = design._force_self_define_materials(base_script, design.self_define_materials)

print("saving into:", target_folder)
made = []
for i in range(design.mode_num):
    temp_fdtd = lumapi.FDTD(hide=design.hide_cad)
    temp_fdtd.putv("current_source", i + 1)
    script = (
        'for (i=1:mode_num){setnamed("Source_TE_mode"+num2str(i)+"_1550", "enabled", (i==current_source));}'
        'setglobalmonitor("frequency points", 11);')
    temp_fdtd.eval(base_script + script)
    fdtd_file = "TE_mode_{}_{}.fsp".format(i, suffix)
    out = os.path.join(target_folder, fdtd_file)
    temp_fdtd.save(out)
    temp_fdtd.close()
    made.append(out)
    print("  saved {:42s} {:.2f} MB".format(fdtd_file, os.path.getsize(out) / 1e6))

# ---- verify what landed inside the saved files -------------------------
# NOTE: every session here is closed via save(<scratch>)+close(), never a bare
# close() - a session with unsaved changes pops a modal "Save as..." dialog on
# the user's desktop (and that dialog then blocks the session).
import tempfile
scratch_dir = os.path.join(tempfile.gettempdir(), "egado_scratch")
os.makedirs(scratch_dir, exist_ok=True)

print()
print("=== contents of each saved file ===")
for out in made:
    fdtd = lumapi.FDTD(hide=True)
    try:
        fdtd.load(out)
        srcs = []
        for k in range(design.mode_num):
            nm = "Source_TE_mode{}_1550".format(k + 1)
            try:
                fdtd.eval('select("{}"); en=get("enabled");'.format(nm))
                srcs.append("{}:{}".format(k + 1, fdtd.getv("en")))
            except Exception:
                srcs.append("{}:MISSING".format(k + 1))
        print(" ", os.path.basename(out))
        print("      sources (enabled) :", "  ".join(srcs))
        for nm in ("design_mesh", "source_mesh1", "source_mesh2", "fom_mesh"):
            try:
                fdtd.eval('select("{}"); t=get("type");'
                          ' xs=get("x span"); ys=get("y span"); zs=get("z span");'
                          ' ox=get("override x mesh"); dxv=get("dx");'.format(nm))
                print("      {:<14} type={:<5} span=({:.3f},{:.3f},{:.3f}) um  ovX={} dx={:.0f} nm".format(
                    nm, str(fdtd.getv("t")), float(fdtd.getv("xs")) * 1e6,
                    float(fdtd.getv("ys")) * 1e6, float(fdtd.getv("zs")) * 1e6,
                    bool(fdtd.getv("ox")), float(fdtd.getv("dxv")) * 1e9))
            except Exception:
                print("      {:<14} MISSING".format(nm))
    finally:
        # save first so close() has nothing to prompt about
        try:
            fdtd.save(os.path.join(scratch_dir, "_makefsp_verify_{}.fsp".format(os.getpid())))
        except Exception:
            pass
        try:
            fdtd.close()
        except Exception:
            pass

# remove the earlier hand-made file if it is still around
stale = os.path.join(design.cur_path, "2mode_base_TO.fsp")
if os.path.isfile(stale):
    os.remove(stale)
    print("\nremoved the earlier hand-saved 2mode_base_TO.fsp")
print("done.")
