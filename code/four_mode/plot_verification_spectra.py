import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO_DIR = Path(r"G:\codex_project\inverse_design_sun\EGADO-4mode")
OUT_DIR = REPO_DIR / "verification_results"
ARTIFACT_DIR = Path(r"C:\Users\USER\.gemini\antigravity\brain\39ad5e21-1313-4872-992d-8dc8c32f9b61")
DATA_FILE = OUT_DIR / "verification_spectra.npz"

def plot_spectra():
    data = np.load(DATA_FILE, allow_pickle=True)
    results = data['results'].item()
    wl_nm = data['wl_nm']
    targets_mean = data['targets_mean']
    overall_mean = float(data['overall_mean'])

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
        ax.set_ylim(0, 70)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(fontsize=9, loc='upper right')

    plt.suptitle(f"Scheme B Cleaned Skeleton Optical Verification (Local 3D-FDTD)\n4-Mode Average Transmittance: {overall_mean*100.0:.2f}% (IL: {-10*np.log10(overall_mean):.2f} dB)",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()

    out_img = REPO_DIR / "scheme_b_cleaned_spectra.png"
    artifact_img = ARTIFACT_DIR / "scheme_b_cleaned_spectra.png"
    plt.savefig(out_img, dpi=300)
    plt.savefig(artifact_img, dpi=300)
    plt.close()
    print("Spectral plots generated successfully:")
    print(f"  {out_img}")
    print(f"  {artifact_img}")

if __name__ == '__main__':
    plot_spectra()
