import os
import sys

from func_MDM_inverse_design import *

"""
Four-mode MDM Multiplexer (TE0, TE1, TE2, TE3)
Reproducing Nat. Commun. 16, 2372 (2025) Supplementary Note 3C / Figs. S10 & S11:
- Design area: 6.0 x 6.0 um^2, 120 nm pixel grid -> 50 x 50 = 2500 pixels
- 4 single-mode input waveguides:
    width = 500 nm, edge-to-edge gap = 1.0 um -> centre-to-centre pitch = 1.5 um
    symmetrically distributed at y = [+2.25, +0.75, -0.75, -2.25] um
    inputs: all fundamental mode TE0 (in_mode_list = [1, 1, 1, 1])
- 1 multimode output waveguide:
    width = 1.8 um, top-aligned with design area upper edge:
    center y = +2.1 um (y span = 1.8 um, upper bound y = +3.0 um, lower bound y = +1.2 um)
    outputs: TE0, TE1, TE2, TE3 (out_mode_list = [1, 2, 3, 4])
- Core thickness: 220 nm Si, SiO2 cladding, center wavelength 1550 nm (span 40 nm)
- Initial structure: All-Si pattern (Fig. S10(a))
"""

if __name__ == "__main__":
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

    PHASE = "TO"
    # PHASE = "EG_ATD"
    # PHASE = "DBS"

    epoch = 0

    TO_working_dir = os.environ.get("EGADO_TO_DIR_4MODE", "")

    if PHASE == "TO":
        # All-Si initial pattern as in Fig. S10(a): uniform ones (mean=0.0, range=1.0 -> 0*rand + 1 = 1.0)
        TO_initial_pattern = [[0.0, 1.0]]
        inverse_design.TO(
            initial_cond=TO_initial_pattern[0],
            initial_type="uniform",
            epoch=epoch,
            resume=False,
            filter_R=240e-9,
        )

    elif PHASE == "EG_ATD":
        if not TO_working_dir:
            raise SystemExit("Set EGADO_TO_DIR_4MODE to the TO output folder")
        inverse_design.EG_ATDO(
            TO_working_dir=TO_working_dir,
            epoch=epoch,
            resume=False,
            resume_from_cur_file=False,
            scan_order="col_by_col",
        )

    elif PHASE == "DBS":
        inverse_design.DBS(epoch=epoch, scan_order="row_by_row", resume=False)

    else:
        assert False, "Supported phases: TO, EG_ATD, DBS"
