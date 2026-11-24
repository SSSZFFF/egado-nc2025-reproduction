import os

from func_MDM_inverse_design import *

"""
       :param pitch: pixel size
       :param shape: square or circle
       :param rect_size: square size
       :param radius: circle radius
       :param z_min: of waveguide core
       :param z_max: of waveguide core
       :param design_region_x: size of optimized region in x direction
       :param design_region_y: size of optimized region in y direction
       :param input_wg_width: width of input waveguides
       :param output_wg_width: width of output waveguides
       :param num_in_wg: number of input waveguides
       :param num_out_wg: number of output waveguides
       :param n_bg: background index (cladding and BOX)
       :param n_wg: waveguide index
       :param wavelength: center wavelength of simulations
       :param wavelength_range: wavelength range of simulations
       :param buff: used in EG_ATD, determines the kernel size of edge-detection, set to be 0, 2, 4 ...
                    (this is the paper's "dilation coefficient D")
       :param target_fom: targeted FoM value, only modifies the display without any impact on the optimization
       :param offset_x: offset of optimized region in the x direction
       :param offset_y: offset of optimized region in the y direction
       :param self_define_materials: Use the material index defined by User when set True
       :param in_mode_list: mode list of input sources
       :param out_mode_list: mode list of output monitors
       :param in_wg_pitch: centre-to-centre spacing of the input waveguides (None ->
                           the evenly-spread formula that base_*.lsf has always used)
"""

# ======================================================================
#  Two-mode MUX  --  reproduces Supplementary Note 3B / Fig. S8 of
#  Nat. Commun. 16, 2372 (2025):
#      two single-mode inputs, 500 nm wide, 1.3 um centre-to-centre
#      square design area 3.6 x 3.6 um^2, 120 nm pixels  ->  30 x 30 = 900
#      multimode output waveguide 900 nm wide
#  The paper digitises with dilation coefficient D = 0, 2, 4; D = 2 is the one
#  shown in Fig. S8(b) and maps onto this code's `buff` parameter.
# ======================================================================

if __name__ == "__main__":
    inverse_design = MDM_InverseDesign(pitch=120e-9,
                                       shape="rect",
                                       rect_size=120e-9,
                                       radius=45e-9,
                                       z_min=-110e-9,
                                       z_max=110e-9,
                                       design_region_x=3.6e-6,
                                       design_region_y=3.6e-6,
                                       num_in_wg=2,
                                       num_out_wg=1,
                                       input_wg_width=0.5e-6,
                                       in_wg_pitch=1.3e-6,
                                       in_mode_list=[1, 1],
                                       output_wg_width=0.9e-6,
                                       out_mode_list=[1, 2],
                                       n_wg="Si (Silicon) - Palik", n_bg="SiO2 (Glass) - Palik",
                                       wavelength=1.55e-6,
                                       wavelength_range=40e-9,
                                       offset_x=0,
                                       offset_y=0,
                                       buff=2,
                                       self_define_materials=True,
                                       processor='CPU',
                                       hide_cad=True,
                                       )

    # PHASE = "TO"
    # PHASE = "ATD"
    PHASE = "EG_ATD"
    # PHASE = "DBS"

    epoch = 0

    # ------------------------------------------------------------------
    # Working directories of the TO phase.
    #
    # These used to be hard-coded paths from the author's machine.  They can
    # now be supplied through environment variables, which keeps the file
    # portable:
    #     set EGADO_TO_DIR_2MODE=...\TO\x3600_y3600_f0240_m0000_0
    #     set EGADO_TO_DIR_ATD=...\TO\WDM_x6000_y6000_f0100_m0000_0
    # The folder name must still encode the filter radius and minimum feature
    # size, because config_for_a2d() parses them out of it:
    #     ..._f<filter_R in nm>..._m<min_feature_size in nm>
    # ------------------------------------------------------------------
    TO_working_dir = os.environ.get("EGADO_TO_DIR_2MODE", "")
    TO_working_dir_atd = os.environ.get("EGADO_TO_DIR_ATD", "")

    if PHASE == "TO":
        # MEAN & RANGE
        TO_initial_pattern = [[0.99, 0.01]]
        # NOTE: the TO phase does not implement resuming yet (see TO(): it
        # asserts if resume=True), so a fresh run must use resume=False.
        inverse_design.TO(initial_cond=TO_initial_pattern[0], initial_type="perlin",
                          epoch=epoch, resume=False, filter_R=240e-9)

    elif PHASE == "ATD":
        if not TO_working_dir_atd:
            raise SystemExit("Set EGADO_TO_DIR_ATD to the TO output folder, e.g. "
                             "...\\TO\\WDM_x6000_y6000_f0100_m0000_0")
        inverse_design.ATDO(TO_working_dir=TO_working_dir_atd, epoch=epoch,
                            resume=False, scan_order="col_by_col")

    elif PHASE == "EG_ATD":
        if not TO_working_dir:
            raise SystemExit("Set EGADO_TO_DIR_2MODE to the TO output folder, e.g. "
                             "...\\TO\\x3600_y3600_f0240_m0000_0")
        # resume_from_cur_file=True reuses the .fsp files that are already in
        # EGA2D/ and continues the flip-pixel sweep from A2D_states.npz.
        # Use resume=False, resume_from_cur_file=False for a fresh run.
        inverse_design.EG_ATDO(TO_working_dir=TO_working_dir, epoch=epoch,
                               resume=False, resume_from_cur_file=False,
                               scan_order="col_by_col")

    elif PHASE == "DBS":
        inverse_design.DBS(epoch=epoch, scan_order="row_by_row", resume=False)

    else:
        assert False, "We only support TO, ATD, EG_ATD & DBS for silicon photonic inverse design!"
