"""Verify optical performance of Scheme B 0-DRC baseline structure (initial.gds).
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shutil

# Setup lumapi
try:
    import lumapi
except ImportError:
    api_path = r"E:\Program Files\ANSYS Inc\v251\Lumerical\api\python"
    if os.path.isdir(api_path) and api_path not in sys.path:
        sys.path.insert(0, api_path)
    import lumapi

REPO_DIR = Path(r"G:\codex_project\inverse_design_sun\EGADO-4mode")
GDS_PATH = REPO_DIR / "scheme_b_dbs_inputs" / "initial.gds"
OUT_DIR = REPO_DIR / "verification_results_scheme_b_initial"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(r"C:\Users\USER\.gemini\antigravity\brain\39ad5e21-1313-4872-992d-8dc8c32f9b61")

OPT_SIZE_X = 6.0e-6
OPT_SIZE_Y = 6.0e-6
NUM_IN_WG = 4
NUM_OUT_WG = 1
IN_WG_WIDTH = 0.5e-6
IN_WG_PITCH = 1.5e-6
OUT_WG_WIDTH = 1.8e-6
LAM_C = 1.55e-6
WAVELENGTH_SPAN = 40e-9
MODE_NUM = 4
IN_MODE_LIST = [1, 1, 1, 1]
OUT_MODE_LIST = [1, 2, 3, 4]
WG_THICKNESS = 220e-9

def build_verification_script():
    lsf_path = REPO_DIR / "base_EGA2D.lsf"
    with open(lsf_path, "r", encoding="utf-8") as f:
        lsf_code = f.read()

    replacements = {
        "opt_size_x=none": f"opt_size_x={OPT_SIZE_X}",
        "opt_size_y=none": f"opt_size_y={OPT_SIZE_Y}",
        "num_in_wg=none": f"num_in_wg={NUM_IN_WG}",
        "num_out_wg=none": f"num_out_wg={NUM_OUT_WG}",
        "lam_c=none": f"lam_c={LAM_C}",
        "out_wg_width=none": f"out_wg_width={OUT_WG_WIDTH}",
        "in_wg_width=none": f"in_wg_width={IN_WG_WIDTH}",
        "in_wg_pitch=none": f"in_wg_pitch={IN_WG_PITCH}",
        "wavelength_span=none": f"wavelength_span={WAVELENGTH_SPAN}",
        "in_mode_list=none": f"in_mode_list={IN_MODE_LIST}",
        "out_mode_list=none": f"out_mode_list={OUT_MODE_LIST}",
        "mode_num=none": f"mode_num={MODE_NUM}",
    }

    script = lsf_code
    for k, v in replacements.items():
        script = script.replace(k, str(v))

    gds_clean_str = GDS_PATH.as_posix()
    import_gds_cmd = f"""
#######################################################
### Import Scheme B 0-DRC Baseline GDS              ###
#######################################################
setmaterial("n_wg", "Mesh Order", 1);
gdsimport("{gds_clean_str}", "TOP", "13:1", "n_wg", -{WG_THICKNESS/2.0}, {WG_THICKNESS/2.0});

# Override FDTD solver settings
setnamed("FDTD", "simulation time", 5000e-15);
setnamed("FDTD", "auto shutoff min", 1e-5);
setglobalmonitor("frequency points", 11);
"""
    return script + import_gds_cmd

def run_single_source(source_idx, configured_script):
    print(f"\n[{source_idx}/4] Launching FDTD for Input Channel {source_idx} (CH{source_idx} -> Target TE{source_idx-1})...")
    t0 = time.time()
    fdtd = lumapi.FDTD(hide=True)
    try:
        fdtd.eval(configured_script)

        for i in range(1, MODE_NUM + 1):
            fdtd.setnamed(f"Source_TE_mode{i}_1550", "enabled", (i == source_idx))

        project_fsp = OUT_DIR / f"four_mode_scheme_b_init_CH{source_idx}.fsp"
        fdtd.save(str(project_fsp))

        print(f"  FSP saved: {project_fsp.name}. Running FDTD simulation...")
        fdtd.run()

        channel_trans = {}
        for m in range(1, MODE_NUM + 1):
            res = fdtd.getresult(f"fom{m}_mode_exp", "expansion for Output")
            t_fwd = np.real(res["T_forward"]).flatten()
            f_pts = np.real(res["f"]).flatten()
            channel_trans[f"TE{m-1}"] = t_fwd

        dt = time.time() - t0
        target_mode = f"TE{source_idx-1}"
        target_t = channel_trans[target_mode]
        mean_t = np.mean(target_t)
        il_db = -10.0 * np.log10(max(mean_t, 1e-6))
        print(f"  Simulation completed in {dt:.1f} s ({dt/60:.2f} min).")
        print(f"  Target Mode {target_mode}: Mean Transmittance = {mean_t*100:.2f}%, IL = {il_db:.2f} dB")
        for m in range(1, MODE_NUM + 1):
            if m != source_idx:
                xt_m = channel_trans[f"TE{m-1}"]
                xt_db = 10.0 * np.log10(max(np.mean(xt_m), 1e-6))
                print(f"    Crosstalk into TE{m-1}: {np.mean(xt_m)*100:.3f}% ({xt_db:.1f} dB)")

        return channel_trans, f_pts
    finally:
        fdtd.close()

def main():
    print("=" * 72)
    print("3D-FDTD Optical Verification: Scheme B 0-DRC Baseline (initial.gds)")
    print(f"GDS target: {GDS_PATH}")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 72)

    script = build_verification_script()
    all_results = {}
    f_axis = None

    t_all_start = time.time()
    for s_idx in range(1, MODE_NUM + 1):
        ch_res, f_pts = run_single_source(s_idx, script)
        all_results[f"CH{s_idx}"] = ch_res
        if f_axis is None:
            f_axis = f_pts

    t_total = time.time() - t_all_start
    print("\n" + "=" * 72)
    print(f"All 4 channels evaluated successfully in {t_total:.1f} s ({t_total/60:.2f} min)!")

    # Summary table
    c_const = 299792458.0
    wavelengths_nm = (c_const / f_axis) * 1e9

    summary_targets = []
    print("\n=== Comprehensive Optical Summary (Scheme B Initial @ 0-DRC) ===")
    for s_idx in range(1, MODE_NUM + 1):
        target_mode = f"TE{s_idx-1}"
        t_arr = all_results[f"CH{s_idx}"][target_mode]
        mean_t = np.mean(t_arr)
        il_db = -10.0 * np.log10(max(mean_t, 1e-6))
        summary_targets.append(mean_t)
        print(f"  CH{s_idx} -> {target_mode}: Transmittance = {mean_t*100:.2f}%, Insertion Loss = {il_db:.2f} dB")

    overall_mean = np.mean(summary_targets)
    overall_il = -10.0 * np.log10(max(overall_mean, 1e-6))
    print(f"  Overall 4-Mode Average Transmittance: {overall_mean*100:.2f}% (IL = {overall_il:.2f} dB)")

    # Save spectra npz
    save_path = OUT_DIR / "scheme_b_initial_spectra.npz"
    np.savez_compressed(
        save_path,
        wavelengths_nm=wavelengths_nm,
        CH1=all_results["CH1"],
        CH2=all_results["CH2"],
        CH3=all_results["CH3"],
        CH4=all_results["CH4"],
        summary_targets=summary_targets,
        overall_mean=overall_mean,
        overall_il=overall_il
    )
    print(f"Full spectral data saved to: {save_path}")

    # Plot spectral curves
    plt.figure(figsize=(10, 8), dpi=300)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for s_idx in range(1, MODE_NUM + 1):
        plt.subplot(2, 2, s_idx)
        target_mode = f"TE{s_idx-1}"
        for m in range(MODE_NUM):
            m_name = f"TE{m}"
            style = '-' if m_name == target_mode else '--'
            alpha = 1.0 if m_name == target_mode else 0.5
            lw = 2.2 if m_name == target_mode else 1.2
            plt.plot(wavelengths_nm, all_results[f"CH{s_idx}"][m_name],
                     style, color=colors[m], label=m_name, alpha=alpha, linewidth=lw)

        plt.title(f"Input CH{s_idx} (Target: {target_mode})", fontsize=11, fontweight='bold')
        plt.xlabel("Wavelength (nm)", fontsize=10)
        plt.ylabel("Transmittance", fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.ylim(0, 1.0)
        plt.legend(loc='upper right', fontsize=8)

    plt.suptitle(f"Scheme B 0-DRC Baseline: 4-Channel Optical Transmission Spectra\nOverall Mean Transmittance = {overall_mean*100:.2f}% ({overall_il:.2f} dB)", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path = OUT_DIR / "scheme_b_initial_spectra.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Spectral plot saved to: {plot_path}")

    if ARTIFACT_DIR.is_dir():
        shutil.copy2(plot_path, ARTIFACT_DIR / "scheme_b_initial_spectra.png")
        print(f"Artifact synced to: {ARTIFACT_DIR / 'scheme_b_initial_spectra.png'}")

if __name__ == "__main__":
    main()
