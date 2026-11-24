"""Verify optical performance of Scheme B cleaned skeleton @ Iter 266 (baseline_4mode_cleaned_iter266.gds).

Runs 4 forward 3D-FDTD simulations (one for each input port CH1~CH4) using local Lumerical.
Measures:
- Transmission into target modes: CH1->TE0, CH2->TE1, CH3->TE2, CH4->TE3
- Modal crosstalk into non-target modes
- Insertion loss (IL in dB)
- Full transmission spectrum from 1530 to 1570 nm (11 wavelength points)
Generates high-resolution 4-panel spectral plots.
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
GDS_PATH = REPO_DIR / "baseline_4mode_cleaned_iter266.gds"
OUT_DIR = REPO_DIR / "verification_results_iter266"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(r"C:\Users\USER\.gemini\antigravity\brain\39ad5e21-1313-4872-992d-8dc8c32f9b61")

# 4-Mode Device Specs
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
### Import Scheme B Cleaned Continuous Skeleton GDS ###
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
        # Build base model and import GDS
        fdtd.eval(configured_script)

        # Enable only the current input source
        for i in range(1, MODE_NUM + 1):
            fdtd.setnamed(f"Source_TE_mode{i}_1550", "enabled", (i == source_idx))

        project_fsp = OUT_DIR / f"four_mode_cleaned_CH{source_idx}.fsp"
        fdtd.save(str(project_fsp))

        print(f"  FSP saved: {project_fsp.name}. Running FDTD simulation...")
        fdtd.run()

        # Extract transmission spectra for all 4 output modes (TE0 ~ TE3)
        channel_trans = {}
        for m in range(1, MODE_NUM + 1):
            res = fdtd.getresult(f"fom{m}_mode_exp", "expansion for Output")
            T_fwd = np.real(np.asarray(res["T_forward"]).squeeze())
            freq = np.asarray(res["f"], dtype=float).squeeze()
            channel_trans[f"TE{m-1}"] = T_fwd

        dt = time.time() - t0
        wl_nm = 299792458.0 / freq * 1e9
        target_mode = f"TE{source_idx-1}"
        target_T = channel_trans[target_mode]
        mean_target_T = float(np.mean(target_T))
        mean_IL_dB = -10.0 * np.log10(mean_target_T) if mean_target_T > 0 else 99.0

        print(f"  Simulation completed in {dt:.1f} s ({dt/60.0:.2f} min).")
        print(f"  Target Mode {target_mode}: Mean Transmittance = {mean_target_T*100.0:.2f}%, IL = {mean_IL_dB:.2f} dB")
        for m in range(MODE_NUM):
            m_name = f"TE{m}"
            if m_name != target_mode:
                xtalk_T = float(np.mean(channel_trans[m_name]))
                xtalk_dB = 10.0 * np.log10(xtalk_T / mean_target_T) if mean_target_T > 0 and xtalk_T > 0 else -99.0
                print(f"    Crosstalk into {m_name}: {xtalk_T*100.0:.3f}% ({xtalk_dB:.1f} dB)")

        return channel_trans, freq, wl_nm, dt
    finally:
        fdtd.close()

def plot_spectra(results, freq, wl_nm, targets_mean, overall_mean):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    axes = axes.flatten()

    colors = {'TE0': '#2ca02c', 'TE1': '#1f77b4', 'TE2': '#9467bd', 'TE3': '#d62728'}
    styles = {'TE0': '-', 'TE1': '--', 'TE2': '-.', 'TE3': ':'}

    for ch_idx in range(1, 5):
        ax = axes[ch_idx - 1]
        ch_data = results[ch_idx]
        target_mode = f"TE{ch_idx - 1}"

        for m_idx in range(4):
            m_name = f"TE{m_idx}"
            T = ch_data[m_name] * 100.0  # %
            mean_T = np.mean(T)
            is_target = (m_name == target_mode)
            lw = 2.4 if is_target else 1.4
            alpha = 1.0 if is_target else 0.75
            label = f"{m_name} (Target: {mean_T:.1f}%)" if is_target else f"{m_name} (Xtalk: {mean_T:.1f}%)"
            ax.plot(wl_nm, T, color=colors[m_name], linestyle=styles[m_name],
                    linewidth=lw, alpha=alpha, label=label)

        ax.set_title(f"CH{ch_idx} Excitation (TE0) -> Target: {target_mode}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Wavelength (nm)", fontsize=10, fontweight='bold')
        ax.set_ylabel("Transmittance (%)", fontsize=10, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=9, loc='upper right')

    plt.suptitle(f"Scheme B Cleaned Skeleton Optical Verification @ Iter 266 (Local 3D-FDTD)\n4-Mode Average Transmittance: {overall_mean*100.0:.2f}% (IL: {-10*np.log10(overall_mean):.2f} dB)",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()

    out_img = REPO_DIR / "scheme_b_iter266_spectra.png"
    artifact_img = ARTIFACT_DIR / "scheme_b_iter266_spectra.png"
    plt.savefig(out_img, dpi=300)
    plt.savefig(artifact_img, dpi=300)
    plt.close()
    print("Spectral plots generated successfully:")
    print(f"  {out_img}")
    print(f"  {artifact_img}")

def main():
    print("=" * 72)
    print("Optical Verification: Scheme B Cleaned Smooth Skeleton @ Iter 266 (4-mode MUX)")
    print(f"GDS target: {GDS_PATH}")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 72)

    script = build_verification_script()

    t_start = time.time()
    results = {}

    for s_idx in range(1, 5):
        ch_trans, freq, wl_nm, dt = run_single_source(s_idx, script)
        results[s_idx] = ch_trans

    total_time = time.time() - t_start
    print("\n" + "=" * 72)
    print(f"All 4 channels evaluated successfully in {total_time:.1f} s ({total_time/60.0:.2f} min)!")

    # Summary table
    print("\n=== Comprehensive Optical Summary (Scheme B Cleaned Skeleton @ Iter 266) ===")
    targets_mean = []
    for s in range(1, 5):
        t_mode = f"TE{s-1}"
        t_mean = float(np.mean(results[s][t_mode]))
        targets_mean.append(t_mean)
        il_db = -10.0 * np.log10(t_mean) if t_mean > 0 else 99.0
        print(f"  CH{s} -> {t_mode}: Transmittance = {t_mean*100.0:.2f}%, Insertion Loss = {il_db:.2f} dB")

    overall_mean = float(np.mean(targets_mean))
    print(f"  Overall 4-Mode Average Transmittance: {overall_mean*100.0:.2f}% (IL = {-10.0*np.log10(overall_mean):.2f} dB)")

    # Save full matrix
    np.savez_compressed(
        OUT_DIR / "verification_spectra.npz",
        results=results, freq=freq, wl_nm=wl_nm,
        targets_mean=targets_mean, overall_mean=overall_mean
    )
    print(f"Full spectral data saved to: {OUT_DIR / 'verification_spectra.npz'}")

    plot_spectra(results, freq, wl_nm, targets_mean, overall_mean)

if __name__ == '__main__':
    main()
