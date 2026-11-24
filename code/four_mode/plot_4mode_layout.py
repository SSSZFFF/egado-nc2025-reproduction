import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_4mode_mux_layout(output_path="G:/codex_project/inverse_design_sun/EGADO-4mode/four_mode_TO_layout.png"):
    fig, ax = plt.subplots(figsize=(12, 11), dpi=300)

    # Design region parameters
    opt_w = 6.0  # um
    opt_h = 6.0  # um
    in_w = 0.5   # um
    in_pitch = 1.5 # um
    in_y_centers = [2.25, 0.75, -0.75, -2.25] # um
    out_w = 1.8  # um
    out_y_center = 2.1 # um (top edge at 3.0 um, bottom at 1.2 um)
    
    # Updated source and monitor spans (+2.0 um)
    src_span_y = in_w + 2.0   # 2.5 um
    fom_span_y = out_w + 2.0  # 3.8 um (bounds: [0.2, 4.0] um)
    
    # FDTD simulation domain (expanded to 9.0 um to fully encompass FOM and sources)
    fdtd_w = 7.0 # um
    fdtd_h = 9.0 # um

    # FDTD simulation domain
    fdtd_rect = patches.Rectangle(
        (-fdtd_w/2, -fdtd_h/2), fdtd_w, fdtd_h,
        linewidth=2.0, edgecolor='#ff7f0e', facecolor='#f8f9fa', linestyle='--',
        label='FDTD Domain (7.0×9.0 μm², PML Margin ≥ 0.5 μm)'
    )
    ax.add_patch(fdtd_rect)

    # SiO2 cladding background indicator
    cladding_rect = patches.Rectangle(
        (-4.2, -4.8), 8.4, 9.6,
        linewidth=0.8, edgecolor='#cccccc', facecolor='#eef5fb', alpha=0.45, label='SiO2 Cladding Background'
    )
    ax.add_patch(cladding_rect)

    # Initial Design Region (All-Si block, Fig. S10(a))
    core_rect = patches.Rectangle(
        (-opt_w/2, -opt_h/2), opt_w, opt_h,
        linewidth=2.5, edgecolor='#1f77b4', facecolor='#b3cde3', alpha=0.85,
        label='Design Region (6.0×6.0 μm²)\nInitial Pattern: All-Si (50×50 pixels @ 120 nm)'
    )
    ax.add_patch(core_rect)

    # 4 Input Waveguides
    for i, yc in enumerate(in_y_centers, 1):
        wg = patches.Rectangle(
            (-3.8, yc - in_w/2), 0.8, in_w,
            linewidth=1.5, edgecolor='#005a32', facecolor='#41ab5d', alpha=0.9
        )
        ax.add_patch(wg)
        
        # Source injection plane (span = 2.5 um)
        src_x = -3.5 + 0.2
        y_bot = yc - src_span_y/2
        y_top = yc + src_span_y/2
        ax.plot([src_x, src_x], [y_bot, y_top], color='#d95f02', linewidth=3.5, solid_capstyle='round', alpha=0.9)
        ax.annotate(f"Source {i} (TE0)\nWidth: {src_span_y:.1f} μm (Wg+2μm)\ny ∈ [{y_bot:+.2f}, {y_top:+.2f}] μm", 
                    xy=(src_x, yc), xytext=(-4.9, yc),
                    arrowprops=dict(arrowstyle="->", color='#d95f02', lw=1.2),
                    fontsize=8.5, fontweight='bold', va='center', ha='right',
                    bbox=dict(boxstyle="round,pad=0.25", fc="#fff3e0", ec="#d95f02", lw=1))

    # Single Multimode Output Waveguide
    out_wg = patches.Rectangle(
        (opt_w/2, out_y_center - out_w/2), 0.8, out_w,
        linewidth=1.5, edgecolor='#49006a', facecolor='#807dba', alpha=0.9,
        label='Multimode Output (Width = 1.8 μm, Top-Aligned)'
    )
    ax.add_patch(out_wg)

    # Output monitor / FOM planes (span = 3.8 um)
    fom_x = 3.5 - 0.2
    fom_bot = out_y_center - fom_span_y/2  # 0.2 um
    fom_top = out_y_center + fom_span_y/2  # 4.0 um
    ax.plot([fom_x, fom_x], [fom_bot, fom_top], 
            color='#810f7c', linewidth=4.0, solid_capstyle='round', alpha=0.9)
    ax.annotate(f"FOM 1~4 Monitors\n(TE0, TE1, TE2, TE3)\nWidth: {fom_span_y:.1f} μm (Wg+2μm)\ny ∈ [{fom_bot:+.1f}, {fom_top:+.1f}] μm", 
                xy=(fom_x, out_y_center), xytext=(4.3, out_y_center),
                arrowprops=dict(arrowstyle="->", color='#810f7c', lw=1.2),
                fontsize=9.0, fontweight='bold', va='center', ha='left',
                bbox=dict(boxstyle="round,pad=0.25", fc="#f3e5f5", ec="#810f7c", lw=1))

    # Geometric Dimension Annotations
    for idx in range(len(in_y_centers)-1):
        y_top_edge = in_y_centers[idx] - in_w/2
        y_bot_edge = in_y_centers[idx+1] + in_w/2
        mid_y = (y_top_edge + y_bot_edge) / 2
        ax.annotate("Gap: 1.0 μm", xy=(-3.2, mid_y), fontsize=8.5, color='#00441b',
                    ha='center', va='center', style='italic',
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))

    # Input waveguide width
    ax.annotate("Width: 500 nm", xy=(-3.4, in_y_centers[0]), color='white',
                fontsize=8.5, fontweight='bold', ha='center', va='center')

    # Pitch annotation
    ax.annotate("Pitch: 1.5 μm", xy=(-2.6, 1.5), fontsize=9, color='#00441b',
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="#e5f5e0", ec="#31a354", lw=0.8))

    # Design area boundary annotations
    ax.annotate("", xy=(-3.0, -3.3), xytext=(3.0, -3.3),
                arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.5))
    ax.text(0, -3.5, "Design Region Width: 6.0 μm", ha="center", va="top", fontsize=10, fontweight="bold", color="#1f77b4")

    ax.annotate("", xy=(3.3, -3.0), xytext=(3.3, 3.0),
                arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.5))
    ax.text(3.55, 0, "Design Region Height: 6.0 μm", ha="left", va="center", fontsize=10, fontweight="bold", color="#1f77b4", rotation=-90)

    # Top alignment highlight
    ax.plot([-3.0, 3.8], [3.0, 3.0], color='#d73027', linestyle=':', lw=1.5)
    ax.text(0.5, 3.12, "Upper Boundary Alignment (y = +3.0 μm)", color='#d73027', fontsize=9, fontweight='bold')

    ax.set_xlim(-5.8, 5.8)
    ax.set_ylim(-5.2, 5.2)
    ax.set_xlabel("X (μm)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Y (μm)", fontsize=11, fontweight='bold')
    ax.set_title("Four-mode MDM Multiplexer Initial Layout (Nat. Commun. 16, 2372 (2025) Fig. S10)\nSource & Monitor Widths = Waveguide Width + 2.0 μm",
                 fontsize=12, fontweight='bold', pad=15)

    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, loc='lower left', fontsize=9, framealpha=0.92)

    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_aspect('equal')
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[SUCCESS] Updated layout diagram saved to {output_path}")

if __name__ == "__main__":
    plot_4mode_mux_layout()
