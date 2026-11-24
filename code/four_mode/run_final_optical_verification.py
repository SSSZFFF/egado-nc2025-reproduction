"""Comprehensive 4-Channel Optical & Modal Purity Verification on Remote Tesla P100.
Evaluates final 120nm DRC-compliant layout across 1530-1570 nm band.
Calculates:
- Insertion Loss (IL) for each channel
- Modal transmission (T_forward) for TE0, TE1, TE2, TE3
- Inter-modal crosstalk & Extinction Ratio (ER)
- Mean overall transmission J
"""

import os
import sys
import time
from pathlib import Path
import json
import numpy as np

# Setup lumapi path
for p in [r"D:\Program Files\ANSYS Inc\v251\Lumerical\api\python",
          r"C:\Program Files\ANSYS Inc\v251\Lumerical\api\python",
          r"E:\Program Files\ANSYS Inc\v251\Lumerical\api\python"]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
import lumapi

REPO_DIR = Path(__file__).resolve().parent
GDS_PATH = REPO_DIR / "results_dbs_phase2" / "final_core_6x6.gds"
if not GDS_PATH.is_file():
    GDS_PATH = REPO_DIR / "results_dbs_phase2" / "final.gds"

OUT_DIR = REPO_DIR / "results_dbs_phase2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

def build_lsf_script(gds_path: Path):
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

    for k, v in replacements.items():
        lsf_code = lsf_code.replace(k, str(v))

    import_cmd = f"""
setmaterial("n_wg", "Mesh Order", 1);
gdsimport("{gds_path.as_posix()}", "TOP", "13:1", "n_wg", -{WG_THICKNESS/2.0}, {WG_THICKNESS/2.0});
setnamed("FDTD", "simulation time", 5000e-15);
setnamed("FDTD", "auto shutoff min", 1e-5);
setglobalmonitor("frequency points", 21);
"""
    return lsf_code + import_cmd

def run_channel(s_idx: int, script_code: str):
    print(f"\n>>> Running FDTD for Channel {s_idx} (Input {s_idx} -> Target TE{s_idx-1})...", flush=True)
    t0 = time.time()
    fdtd = lumapi.FDTD(hide=True)
    try:
        fdtd.eval(script_code)
        for i in range(1, MODE_NUM + 1):
            fdtd.setnamed(f"Source_TE_mode{i}_1550", "enabled", (i == s_idx))

        fsp_path = OUT_DIR / f"verify_final_CH{s_idx}.fsp"
        fdtd.save(str(fsp_path))
        fdtd.run()

        data = {}
        for m in range(1, MODE_NUM + 1):
            res = fdtd.getresult(f"fom{m}_mode_exp", "expansion for Output")
            data[f"TE{m-1}"] = np.real(np.asarray(res["T_forward"]).squeeze())
            freq = np.asarray(res["f"], dtype=float).squeeze()

        dt = time.time() - t0
        target_mode = f"TE{s_idx-1}"
        target_T = data[target_mode]
        mean_T = float(np.mean(target_T))
        mean_IL = -10.0 * np.log10(mean_T) if mean_T > 0 else 99.0

        print(f"  Channel {s_idx} completed in {dt:.1f}s", flush=True)
        print(f"  Target Mode {target_mode}: Mean T = {mean_T*100:.2f}%, IL = {mean_IL:.2f} dB", flush=True)
        for m in range(MODE_NUM):
            m_name = f"TE{m}"
            if m_name != target_mode:
                xtalk_T = float(np.mean(data[m_name]))
                xtalk_dB = 10.0 * np.log10(xtalk_T / mean_T) if mean_T > 0 and xtalk_T > 0 else -99.0
                print(f"    Crosstalk {m_name}: {xtalk_T*100:.3f}% ({xtalk_dB:.1f} dB)", flush=True)

        return s_idx, data, freq, dt
    finally:
        fdtd.close()

def main():
    print("=" * 70, flush=True)
    print("Comprehensive 4-Channel Verification for Final 120nm DBS Design", flush=True)
    print(f"Target GDS: {GDS_PATH}", flush=True)
    print(f"Wavelength Band: 1530 nm - 1570 nm (21 frequency points)", flush=True)
    print("=" * 70, flush=True)

    script_code = build_lsf_script(GDS_PATH)
    t_start = time.time()

    all_data = {}
    freqs = None
    mean_trans = []

    for s_idx in range(1, 5):
        _, ch_data, freqs, dt = run_channel(s_idx, script_code)
        all_data[f"CH{s_idx}"] = ch_data
        target_mode = f"TE{s_idx-1}"
        mean_trans.append(float(np.mean(ch_data[target_mode])))

    c = 299792458.0
    wavelengths_nm = (c / freqs) * 1e9

    overall_J = float(np.mean(mean_trans))
    overall_IL = float(np.mean([-10.0 * np.log10(t) for t in mean_trans]))

    print("\n" + "=" * 70, flush=True)
    print(f"FINAL VERIFICATION SUMMARY (Over 1530 - 1570 nm Band):", flush=True)
    print(f"  CH1 (Input 1 -> TE0): {mean_trans[0]*100:.2f}% (IL = {-10*np.log10(mean_trans[0]):.2f} dB)", flush=True)
    print(f"  CH2 (Input 2 -> TE1): {mean_trans[1]*100:.2f}% (IL = {-10*np.log10(mean_trans[1]):.2f} dB)", flush=True)
    print(f"  CH3 (Input 3 -> TE2): {mean_trans[2]*100:.2f}% (IL = {-10*np.log10(mean_trans[2]):.2f} dB)", flush=True)
    print(f"  CH4 (Input 4 -> TE3): {mean_trans[3]*100:.2f}% (IL = {-10*np.log10(mean_trans[3]):.2f} dB)", flush=True)
    print(f"  Overall 4-Channel Mean J: {overall_J*100:.2f}%", flush=True)
    print(f"  Overall Mean Insertion Loss: {overall_IL:.2f} dB", flush=True)
    print(f"  Total Verification Time: {time.time() - t_start:.1f} s", flush=True)
    print("=" * 70, flush=True)

    # Save to npz
    save_path = OUT_DIR / "final_verification_spectra.npz"
    np.savez_compressed(
        save_path,
        wavelengths_nm=wavelengths_nm,
        mean_trans=mean_trans,
        overall_J=overall_J,
        overall_IL=overall_IL,
        CH1=all_data["CH1"],
        CH2=all_data["CH2"],
        CH3=all_data["CH3"],
        CH4=all_data["CH4"]
    )
    print(f"Saved complete spectra to: {save_path}", flush=True)

if __name__ == "__main__":
    main()
