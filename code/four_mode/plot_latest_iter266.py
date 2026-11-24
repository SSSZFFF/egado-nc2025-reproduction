import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARTIFACT_DIR = r"C:\Users\USER\.gemini\antigravity\brain\39ad5e21-1313-4872-992d-8dc8c32f9b61"
LOCAL_DIR = r"G:\codex_project\inverse_design_sun\EGADO-4mode"

def plot_all():
    # 1. Combine Convergence Data across all 3 runs
    rep1 = os.path.join(LOCAL_DIR, "convergence_report.txt")
    rep2 = os.path.join(LOCAL_DIR, "convergence_report_run2.txt")
    rep3 = os.path.join(LOCAL_DIR, "convergence_report_run3.txt")
    
    def load_data(rep):
        data = []
        with open(rep, 'r') as f:
            for line in f:
                parts = [float(p.strip()) for p in line.split(',') if p.strip()]
                if parts:
                    data.append(parts)
        return np.array(data)

    data1 = load_data(rep1)
    data2 = load_data(rep2)
    data3 = load_data(rep3)
    
    # Remap data2 and data3 iterations
    offset1 = data1[-1, 0] # 101
    data2_remap = data2[1:].copy()
    data2_remap[:, 0] = data2_remap[:, 0] + offset1
    
    offset2 = data2_remap[-1, 0] # 113
    data3_remap = data3[1:].copy()
    data3_remap[:, 0] = data3_remap[:, 0] + offset2
    
    combined_data = np.vstack([data1, data2_remap, data3_remap])
    
    iters = combined_data[:, 0]
    total_loss = combined_data[:, 1]
    te0_loss = combined_data[:, 2]
    te1_loss = combined_data[:, 3]
    te2_loss = combined_data[:, 4]
    te3_loss = combined_data[:, 5]
    beta = combined_data[:, 6]
    discreteness = combined_data[:, 7]

    te0_t = (1.0 - te0_loss) * 100.0
    te1_t = (1.0 - te1_loss) * 100.0
    te2_t = (1.0 - te2_loss) * 100.0
    te3_t = (1.0 - te3_loss) * 100.0
    avg_t = (te0_t + te1_t + te2_t + te3_t) / 4.0

    # Plot Convergence (3 subplots)
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, dpi=300)
    
    # Run partition lines
    for ax in axes:
        ax.axvline(x=offset1, color='gray', linestyle=':', alpha=0.7)
        ax.axvline(x=offset2, color='gray', linestyle=':', alpha=0.7)

    # Subplot 1: Transmittance of 4 Modes
    axes[0].plot(iters, te0_t, color='#2ca02c', label='TE0 (Target 1550nm)', linewidth=1.5)
    axes[0].plot(iters, te1_t, color='#1f77b4', label='TE1 (Target 1550nm)', linewidth=1.5)
    axes[0].plot(iters, te2_t, color='#9467bd', label='TE2 (Target 1550nm)', linewidth=1.5)
    axes[0].plot(iters, te3_t, color='#d62728', label='TE3 (Target 1550nm)', linewidth=1.5)
    axes[0].plot(iters, avg_t, color='black', linestyle='--', label=f'Average ({avg_t[-1]:.2f}%)', linewidth=2.0)
    axes[0].set_ylabel('Transmittance (%)', fontsize=10, fontweight='bold')
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc='lower left', ncol=3, fontsize=9)
    axes[0].set_title(f'Four-mode MUX Topology Optimization Progress (Cumulative Iter 0 ~ {int(iters[-1])})\nTesla P100 Remote Optimization | Checkpoint 153 (β={beta[-1]:.2f}, Discreteness={discreteness[-1]*100:.1f}%)', fontsize=12, fontweight='bold')

    # Annotate runs
    axes[0].text(offset1/2, 15, 'Run 1\n(Iter 0~101)', ha='center', fontsize=9, color='dimgray')
    axes[0].text((offset1+offset2)/2, 15, 'Run 2\n(102~113)', ha='center', fontsize=9, color='dimgray')
    axes[0].text((offset2+iters[-1])/2, 15, f'Run 3\n(114~{int(iters[-1])})', ha='center', fontsize=9, color='navy', fontweight='bold')

    # Subplot 2: Total Objective Loss
    axes[1].semilogy(iters, total_loss, color='#e66101', label='Combined Objective Loss', linewidth=1.8)
    axes[1].set_ylabel('FOM Loss (log scale)', fontsize=10, fontweight='bold')
    axes[1].grid(True, which='both', linestyle=':', alpha=0.6)
    axes[1].legend(loc='upper right', fontsize=9)

    # Subplot 3: Beta and Discreteness
    ax3_twin = axes[2].twinx()
    l1 = axes[2].plot(iters, beta, color='#5e3c99', label=f'Threshold Sharpness $\\beta$ ({beta[-1]:.2f})', linewidth=2.0)
    l2 = ax3_twin.plot(iters, discreteness * 100.0, color='#018571', linestyle='-.', label=f'Discreteness ({discreteness[-1]*100.0:.1f}%)', linewidth=2.0)
    axes[2].set_xlabel('Cumulative Iteration Count', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('$\\beta$ parameter', color='#5e3c99', fontsize=10, fontweight='bold')
    ax3_twin.set_ylabel('Discreteness (%)', color='#018571', fontsize=10, fontweight='bold')
    axes[2].set_ylim(0, max(beta) * 1.15)
    ax3_twin.set_ylim(50, 100)
    axes[2].grid(True, linestyle=':', alpha=0.6)
    
    # Combined legend for ax2 and ax2_twin
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    axes[2].legend(lines, labels, loc='lower right', fontsize=9)

    plt.tight_layout()
    conv_img_local = os.path.join(LOCAL_DIR, "to_convergence_iter266.png")
    plt.savefig(conv_img_local, dpi=300)
    plt.close()

    # 2. Plot Permittivity Pattern for Checkpoint 153
    ckpt_path = os.path.join(LOCAL_DIR, "parameters_run3_153.npz")
    ckpt = np.load(ckpt_path)
    params = ckpt['params']
    
    # 50x50 design grid
    x = np.linspace(-3.0, 3.0, params.shape[0])
    y = np.linspace(-3.0, 3.0, params.shape[1])

    # Inputs and output positions
    inputs_y = [2.25, 0.75, -0.75, -2.25]

    fig, ax = plt.subplots(figsize=(9, 7.8), dpi=300)
    im = ax.imshow(params.T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]],
                   cmap='viridis', vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Permittivity Parameter $\\rho$ (0=SiO2, 1=Si)', fontsize=10, fontweight='bold')

    for i, iy in enumerate(inputs_y):
        ax.plot([-3.0, -3.5], [iy, iy], color='crimson', linewidth=4.0)
        ax.text(-3.6, iy, f'IN{i+1}\n(TE0)', color='crimson', fontsize=9, fontweight='bold',
                ha='right', va='center')

    ax.plot([3.0, 3.5], [2.1, 2.1], color='cyan', linewidth=10.0, alpha=0.7)
    ax.text(3.6, 2.1, 'OUT\n(TE0-3)', color='blue', fontsize=9, fontweight='bold',
            ha='left', va='center')

    ax.set_xlim(-4.2, 4.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel('x (μm)', fontsize=11, fontweight='bold')
    ax.set_ylabel('y (μm)', fontsize=11, fontweight='bold')
    ax.set_title(f'Four-mode MUX: Dielectric Distribution @ Iter {int(iters[-1])}\n(Run 3 Ckpt 153, β={beta[-1]:.2f}, Discreteness={discreteness[-1]*100:.1f}%)',
                 fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)

    pat_img_local = os.path.join(LOCAL_DIR, "to_pattern_iter266.png")
    plt.savefig(pat_img_local, dpi=300)
    plt.close()

    # Also plot the filtered permittivity eps
    if 'eps' in ckpt:
        eps = ckpt['eps']
        eps_norm = (eps - ckpt['eps_min']) / (ckpt['eps_max'] - ckpt['eps_min'])
        fig, ax = plt.subplots(figsize=(9, 7.8), dpi=300)
        im = ax.imshow(eps_norm.T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]],
                       cmap='viridis', vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Filtered Relative Permittivity $\\tilde{\\epsilon}$ (0=SiO2, 1=Si)', fontsize=10, fontweight='bold')
        for i, iy in enumerate(inputs_y):
            ax.plot([-3.0, -3.5], [iy, iy], color='crimson', linewidth=4.0)
            ax.text(-3.6, iy, f'IN{i+1}\n(TE0)', color='crimson', fontsize=9, fontweight='bold',
                    ha='right', va='center')
        ax.plot([3.0, 3.5], [2.1, 2.1], color='cyan', linewidth=10.0, alpha=0.7)
        ax.text(3.6, 2.1, 'OUT\n(TE0-3)', color='blue', fontsize=9, fontweight='bold',
                ha='left', va='center')
        ax.set_xlim(-4.2, 4.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_xlabel('x (μm)', fontsize=11, fontweight='bold')
        ax.set_ylabel('y (μm)', fontsize=11, fontweight='bold')
        ax.set_title(f'Four-mode MUX: Filtered Permittivity @ Iter {int(iters[-1])}\n(Run 3 Ckpt 153, β={beta[-1]:.2f}, Discreteness={discreteness[-1]*100:.1f}%)',
                     fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        eps_img_local = os.path.join(LOCAL_DIR, "to_filtered_eps_iter266.png")
        plt.savefig(eps_img_local, dpi=300)
        plt.close()
        shutil.copy(eps_img_local, os.path.join(ARTIFACT_DIR, "to_filtered_eps_iter266.png"))

    # Copy to ARTIFACT_DIR
    shutil.copy(conv_img_local, os.path.join(ARTIFACT_DIR, "to_convergence_iter266.png"))
    shutil.copy(pat_img_local, os.path.join(ARTIFACT_DIR, "to_pattern_iter266.png"))
    print("Done! Artifacts created.")

if __name__ == "__main__":
    plot_all()
