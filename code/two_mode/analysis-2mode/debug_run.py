"""Debug: why does `run` finish in 3 s once the new meshes are present?"""
import os
import re
import sys
import glob
import traceback

repo = os.path.abspath(sys.argv[1])
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

fdtd = lumapi.FDTD(hide=True)
try:
    print("--- evaluating base_TO.lsf ---")
    fdtd.eval(script)
    print("    eval OK")

    print("--- listing mesh objects ---")
    for nm in ("design_mesh", "source_mesh1", "source_mesh2", "fom_mesh"):
        try:
            fdtd.eval('select("{}"); t=get("type"); en=get("enabled");'
                      ' xs=get("x span"); ys=get("y span"); zs=get("z span");'.format(nm))
            print("    {:<14} type={} enabled={} span=({:.3f},{:.3f},{:.3f}) um".format(
                nm, fdtd.getv("t"), fdtd.getv("en"), float(fdtd.getv("xs")) * 1e6,
                float(fdtd.getv("ys")) * 1e6, float(fdtd.getv("zs")) * 1e6))
        except Exception as exc:
            print("    {:<14} ERROR {}".format(nm, exc))

    print("--- checking sources ---")
    for nm in ("Source_TE_mode1_1550", "Source_TE_mode2_1550"):
        fdtd.eval('select("{}"); en=get("enabled");'.format(nm))
        print("    {} enabled={}".format(nm, fdtd.getv("en")))

    print("--- running ---")
    try:
        fdtd.eval("run;")
        print("    run returned")
    except Exception:
        print("    run RAISED:")
        traceback.print_exc()

    print("--- session state after run ---")
    try:
        fdtd.eval('select("FDTD"); lay=get("layout mode");')
        print("    layout mode:", fdtd.getv("lay"))
    except Exception as exc:
        print("    query failed:", exc)

    print("--- any log files? ---")
    for p in sorted(glob.glob(os.path.join(repo, "*.log")), key=os.path.getmtime)[-3:]:
        print("   ", os.path.basename(p), os.path.getsize(p), "bytes")
finally:
    try:
        fdtd.close()
    except Exception:
        pass
