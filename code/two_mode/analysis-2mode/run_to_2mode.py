"""Run ONE iteration of the 2-mode TO through the real entry point.

This exercises the project's own session handling end to end (lumopt creates,
saves and closes its own .fsp files), so nothing here can raise a Save-as
dialog.  Used to check that the new design_mesh / source_mesh overrides are
accepted by the pipeline.

usage:  python run_to_2mode.py <repo> <max_iter> [epoch]
"""
import os
import sys
import time

repo = os.path.abspath(sys.argv[1])
max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 1
epoch = int(sys.argv[3]) if len(sys.argv) > 3 else 0

sys.path.insert(0, repo)
os.chdir(repo)

from func_MDM_inverse_design import MDM_InverseDesign  # noqa: E402

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

print("=" * 72, flush=True)
print("2-mode TO: max_iter={} epoch={}".format(max_iter, epoch), flush=True)
print("started at", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
print("=" * 72, flush=True)

t0 = time.time()
workdir = design.TO(initial_cond=[0.99, 0.01], initial_type="perlin",
                    epoch=epoch, resume=False, filter_R=240e-9, max_iter=max_iter)
dt = time.time() - t0

print("=" * 72, flush=True)
print("TO finished in {:.1f} s ({:.2f} min)".format(dt, dt / 60.0), flush=True)
print("working dir:", workdir, flush=True)
