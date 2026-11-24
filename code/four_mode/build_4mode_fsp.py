import os
import sys
import json

# Setup lumapi path if available on Windows
lumapi_paths = [
    r"D:\Program Files\ANSYS Inc\v251\Lumerical\api\python",
    r"C:\Program Files\Lumerical\v241\api\python",
    r"C:\Program Files\Lumerical\v202\api\python",
]
for p in lumapi_paths:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    import lumapi
except ImportError:
    lumapi = None
    print("[WARN] lumapi not directly importable on this environment.")

def build_4mode_to_fsp(output_dir="."):
    """Builds the 4-mode TO initial FSP and exports CAD view and metadata."""
    if lumapi is None:
        raise RuntimeError("lumapi is required to generate the FSP file.")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    to_base_dir = os.path.join(output_dir, "TO_base")
    os.makedirs(to_base_dir, exist_ok=True)

    # Load base_TO.lsf template
    lsf_path = os.path.join(output_dir, "base_TO.lsf")
    if not os.path.isfile(lsf_path):
        lsf_path = "base_TO.lsf"
    
    with open(lsf_path, "r", encoding="utf-8") as f:
        lsf_code = f.read()

    # Geometry & Physical parameters matching Nat. Commun. 16, 2372 (2025) Fig. S10
    opt_size_x = 6.0e-6
    opt_size_y = 6.0e-6
    num_in_wg = 4
    num_out_wg = 1
    input_wg_width = 0.5e-6
    in_wg_pitch = 1.5e-6
    output_wg_width = 1.8e-6
    center_wavelength = 1.55e-6
    wavelength_range = 40e-9
    mode_num = 4
    in_mode_list = [1, 1, 1, 1]
    out_mode_list = [1, 2, 3, 4]
    wg_thickness = 220e-9

    # Replace macros in LSF
    replacements = {
        "opt_size_x=none": f"opt_size_x={opt_size_x}",
        "opt_size_y=none": f"opt_size_y={opt_size_y}",
        "num_in_wg=none": f"num_in_wg={num_in_wg}",
        "num_out_wg=none": f"num_out_wg={num_out_wg}",
        "lam_c=none": f"lam_c={center_wavelength}",
        "out_wg_width=none": f"out_wg_width={output_wg_width}",
        "in_wg_width=none": f"in_wg_width={input_wg_width}",
        "in_wg_pitch=none": f"in_wg_pitch={in_wg_pitch}",
        "wavelength_span=none": f"wavelength_span={wavelength_range}",
        "in_mode_list=none": f"in_mode_list={in_mode_list}",
        "out_mode_list=none": f"out_mode_list={out_mode_list}",
        "mode_num=none": f"mode_num={mode_num}",
    }

    configured_script = lsf_code
    for k, v in replacements.items():
        configured_script = configured_script.replace(k, str(v))

    # Add the initial all-Si design region (Fig. S10(a))
    initial_region_script = f"""

############################################
### Initial Structure: All-Si Design Region
############################################
addrect;
set('name', 'initial_design_region');
set('x', 0);
set('x span', {opt_size_x});
set('y', 0);
set('y span', {opt_size_y});
set('z', 0);
set('z span', {wg_thickness});
set('material', 'n_wg');
"""

    full_script = configured_script + initial_region_script
    gen_lsf = os.path.join(output_dir, "generate_4mode_TO.lsf")
    with open(gen_lsf, "w", encoding="utf-8") as f:
        f.write(full_script)

    print("[1/5] Launching Lumerical FDTD session...")
    fdtd = lumapi.FDTD(hide=True)

    print(f"[2/5] Running {gen_lsf} in FDTD...")
    clean_dir = output_dir.replace("\\", "/")
    fdtd.eval(f"cd('{clean_dir}'); generate_4mode_TO;")

    # Save comprehensive master initial FSP
    master_fsp = os.path.join(output_dir, "four_mode_TO_initial.fsp")
    print(f"[3/5] Saving master initial FSP: {master_fsp}")
    fdtd.save(master_fsp)

    # Export layout top-view image
    layout_img = os.path.join(output_dir, "four_mode_TO_layout.png")
    print(f"[4/5] Exporting layout CAD screenshot: {layout_img}")
    try:
        fdtd.eval('setview("view xy");')
        fdtd.exportview(layout_img)
    except Exception as e:
        print(f"[WARN] exportview failed: {e}")

    # Generate individual mode forward simulation files in TO_base/
    print("[5/5] Generating individual mode simulation files in TO_base/...")
    for i in range(mode_num):
        fdtd.putv("current_source", i + 1)
        mode_script = (
            'for (j=1:mode_num){'
            '  setnamed("Source_TE_mode"+num2str(j)+"_1550", "enabled", (j==current_source));'
            '}'
            'setglobalmonitor("frequency points", 11);'
        )
        fdtd.eval(mode_script)
        mode_fsp = os.path.join(to_base_dir, f"TE_mode_{i}_TO.fsp")
        fdtd.save(mode_fsp)

    fdtd.close()

    # Write metadata JSON for inspection
    meta = {
        "device": "Four-mode MDM Multiplexer (TE0 -> TE0, TE1, TE2, TE3)",
        "paper_reference": "Nat. Commun. 16, 2372 (2025) Supplementary Note 3C / Fig. S10",
        "design_region": {
            "x_span_um": opt_size_x * 1e6,
            "y_span_um": opt_size_y * 1e6,
            "thickness_nm": wg_thickness * 1e9,
            "grid_pixels": "50 x 50 (120 nm x 120 nm)",
            "initial_state": "All-Si initial pattern (Fig. S10(a))"
        },
        "inputs": [
            {"channel": 1, "width_nm": 500, "y_center_um": 2.25, "source_width_um": 2.5, "source_bounds_um": [1.0, 3.5], "injected_mode": "TE0"},
            {"channel": 2, "width_nm": 500, "y_center_um": 0.75, "source_width_um": 2.5, "source_bounds_um": [-0.5, 2.0], "injected_mode": "TE0"},
            {"channel": 3, "width_nm": 500, "y_center_um": -0.75, "source_width_um": 2.5, "source_bounds_um": [-2.0, 0.5], "injected_mode": "TE0"},
            {"channel": 4, "width_nm": 500, "y_center_um": -2.25, "source_width_um": 2.5, "source_bounds_um": [-3.5, -1.0], "injected_mode": "TE0"},
        ],
        "input_spacing_edge_to_edge_um": 1.0,
        "input_pitch_center_to_center_um": 1.5,
        "output": {
            "width_um": 1.8,
            "y_center_um": 2.1,
            "y_top_edge_um": 3.0,
            "y_bottom_edge_um": 1.2,
            "monitor_width_um": 3.8,
            "monitor_bounds_um": [0.2, 4.0],
            "alignment": "Top-aligned with design area upper boundary",
            "target_modes": [
                {"input_channel": 1, "target_mode": "TE0"},
                {"input_channel": 2, "target_mode": "TE1"},
                {"input_channel": 3, "target_mode": "TE2"},
                {"input_channel": 4, "target_mode": "TE3"}
            ]
        },
        "simulation": {
            "wavelength_center_nm": center_wavelength * 1e9,
            "wavelength_span_nm": wavelength_range * 1e9,
            "core_material": "Si (Silicon) - Palik",
            "cladding_material": "SiO2 (Glass) - Palik",
            "mesh_size_nm": 20,
            "fdtd_dimension": "3D",
            "z_symmetry": "Symmetric (z min bc)",
            "fdtd_span_um": [7.0, 9.0, 2.0]
        },
        "generated_files": {
            "master_fsp": master_fsp,
            "layout_image": layout_img,
            "mode_fsp_files": [
                os.path.join(to_base_dir, f"TE_mode_{i}_TO.fsp") for i in range(mode_num)
            ]
        }
    }

    meta_file = os.path.join(output_dir, "four_mode_specs.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[SUCCESS] 4-mode TO files generated in {output_dir}")
    return master_fsp

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    build_4mode_to_fsp(out)
