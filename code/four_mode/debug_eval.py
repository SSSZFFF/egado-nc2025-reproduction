import os
import sys

LUMERICAL_API = r"D:\Program Files\ANSYS Inc\v251\Lumerical\api\python"
if LUMERICAL_API not in sys.path:
    sys.path.insert(0, LUMERICAL_API)

import lumapi

fdtd = lumapi.FDTD(hide=True)

with open("base_TO.lsf", "r", encoding="utf-8") as f:
    code = f.read()

replacements = {
    "opt_size_x=none": "opt_size_x=6e-6",
    "opt_size_y=none": "opt_size_y=6e-6",
    "num_in_wg=none": "num_in_wg=4",
    "num_out_wg=none": "num_out_wg=1",
    "lam_c=none": "lam_c=1.55e-6",
    "out_wg_width=none": "out_wg_width=1.8e-6",
    "in_wg_width=none": "in_wg_width=0.5e-6",
    "in_wg_pitch=none": "in_wg_pitch=1.5e-6",
    "wavelength_span=none": "wavelength_span=40e-9",
    "in_mode_list=none": "in_mode_list=[1, 1, 1, 1]",
    "out_mode_list=none": "out_mode_list=[1, 2, 3, 4]",
    "mode_num=none": "mode_num=4",
}
for k, v in replacements.items():
    code = code.replace(k, v)

# Evaluate in chunks or line-by-line
lines = code.splitlines()
cur_block = ""
in_block = False

for idx, line in enumerate(lines, 1):
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    
    cur_block += line + "\n"
    # Check if inside a block like for or if
    if "{" in stripped:
        in_block = True
    if "}" in stripped:
        in_block = False
    
    if not in_block:
        try:
            fdtd.eval(cur_block)
            cur_block = ""
        except Exception as e:
            print(f"FAILED around line {idx}:")
            print(f"Block:\n{cur_block}")
            print(f"Error: {e}")
            break

fdtd.close()
