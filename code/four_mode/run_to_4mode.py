import os
import sys
import time
import glob

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

from func_MDM_inverse_design import MDM_InverseDesign

args = [a for a in sys.argv[1:]]
resume_file = None
if "--resume" in args:
    i = args.index("--resume")
    resume_file = args[i + 1]
    del args[i:i + 2]

max_iter = int(args[0]) if len(args) > 0 else 400
epoch = int(args[1]) if len(args) > 1 else 0

print("=" * 72, flush=True)
print("Four-mode MDM MUX Topology Optimization (TO)", flush=True)
print(f"Directory: {REPO}", flush=True)
if resume_file:
    print(f"TO RESUME: max_iter={max_iter} epoch={epoch}", flush=True)
    print(f"Checkpoint: {resume_file}", flush=True)
else:
    print(f"TO FRESH RUN: max_iter={max_iter} epoch={epoch}", flush=True)
    print("Initial Pattern: All-Si (50x50 pixels, 120 nm)", flush=True)
print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print("=" * 72, flush=True)

inverse_design = MDM_InverseDesign(
    pitch=120e-9,
    shape="rect",
    rect_size=120e-9,
    radius=45e-9,
    z_min=-110e-9,
    z_max=110e-9,
    design_region_x=6.0e-6,
    design_region_y=6.0e-6,
    num_in_wg=4,
    num_out_wg=1,
    input_wg_width=0.5e-6,
    in_wg_pitch=1.5e-6,
    in_mode_list=[1, 1, 1, 1],
    output_wg_width=1.8e-6,
    out_mode_list=[1, 2, 3, 4],
    n_wg="Si (Silicon) - Palik",
    n_bg="SiO2 (Glass) - Palik",
    wavelength=1.55e-6,
    wavelength_range=40e-9,
    offset_x=0,
    offset_y=0,
    buff=2,
    self_define_materials=True,
    processor="GPU",
    hide_cad=True,
)

t0 = time.time()
if resume_file:
    workdir = inverse_design.TO(
        initial_type="file",
        initial_file=resume_file,
        epoch=epoch,
        resume=False,
        filter_R=240e-9,
        max_iter=max_iter,
        num_threads=1,
    )
else:
    # All-Si initial pattern (Fig. S10(a)): uniform ones everywhere
    TO_initial_pattern = [[0.0, 1.0]]
    workdir = inverse_design.TO(
        initial_cond=TO_initial_pattern[0],
        initial_type="uniform",
        epoch=epoch,
        resume=False,
        filter_R=240e-9,
        max_iter=max_iter,
        num_threads=1,
    )
dt = time.time() - t0

print("=" * 72, flush=True)
print(f"TO finished in {dt:.1f} s ({dt/3600.0:.2f} h)", flush=True)
print(f"Working dir: {workdir}", flush=True)
if os.path.isdir(workdir):
    files = sorted(os.listdir(workdir))
    print(f"Files saved: {len(files)}", flush=True)
