""""""""""""""""""""""""""""""""""""""""""""""""""""
Inverse Design Project implementing multiple methods
Copyright (C)   - FiOWIN Lab @ Fudan University
Change Log      - 2022 v1.0: DBS
                - 2023 v2.0: TO
                - 2023 v3.0: ATD
                - 2023 v4.0: EG_ATD
                
Last Edition    - 2023.12.8
                            by Aolong Sun & Xuyu Deng
"""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
import sys


# ---------------------------------------------------------------------------
# Locate the Lumerical Python API (lumapi.py + lumopt/) instead of relying on a
# hard-coded path from the author's machine.  Order of precedence:
#   1. the LUMERICAL_API environment variable
#   2. an automatic scan of the usual install roots
# ---------------------------------------------------------------------------
def find_lumerical_api():
    """Return the directory containing lumapi.py, or None if not found."""
    env = os.environ.get("LUMERICAL_API")
    if env and os.path.isfile(os.path.join(env, "lumapi.py")):
        return env

    roots = ["E:\\Program Files\\ANSYS Inc", "C:\\Program Files\\ANSYS Inc",
             "D:\\Program Files\\ANSYS Inc", "F:\\Program Files\\ANSYS Inc",
             "E:\\Program Files", "C:\\Program Files", "D:\\Program Files",
             "E:\\", "D:\\", "C:\\", "G:\\"]
    candidates = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        try:
            entries = sorted(os.listdir(root))
        except OSError:
            continue
        for name in entries:
            # current Ansys layout: <root>\v251\Lumerical\api\python
            candidates.append(os.path.join(root, name, "Lumerical", "api", "python"))
            # legacy layout:        <root>\Lumerical\v202\api\python
            candidates.append(os.path.join(root, "Lumerical", name, "api", "python"))
    for c in candidates:
        if os.path.isfile(os.path.join(c, "lumapi.py")):
            return c
    return None


LUMERICAL_API = find_lumerical_api()
if LUMERICAL_API:
    sys.path.insert(0, LUMERICAL_API)
    print("[EGADO] Lumerical API found at: {}".format(LUMERICAL_API))
else:
    print("[EGADO] WARNING: Lumerical API not found. Set the LUMERICAL_API "
          "environment variable to the folder that contains lumapi.py.")

# ---------------------------------------------------------------------------
# NumPy 2 compatibility shim for lumopt.
#
# lumopt (as shipped with Lumerical 2025 R1) calls np.trapz in
# figures_of_merit/modematch.py and utilities/edge.py.  np.trapz was removed in
# NumPy 2.0 and renamed to np.trapezoid, so the optimization dies with
# "module 'numpy' has no attribute 'trapz'" *after* the first FDTD sweep has
# already been computed.  This is the only removed NumPy API lumopt uses.
# Patching it here keeps the whole project on a modern NumPy; the alternative
# is to run everything with the Lumerical-bundled Python 3.9 (NumPy 1.19).
# ---------------------------------------------------------------------------
import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid

# ---------------------------------------------------------------------------
# SciPy compatibility shim for lumopt.
#
# scipy.misc.derivative was removed in SciPy 1.10, but lumopt's
# Optimizer.create_jac_approx() still calls it.  Only the finite-difference
# fallback path uses it, so it is re-implemented here (a faithful copy of the
# removed SciPy routine) to keep a long optimization run from dying on it.
# ---------------------------------------------------------------------------
import scipy
import scipy.misc

if not hasattr(scipy.misc, "derivative"):
    def central_diff_weights(Np, ndiv=1):
        if Np < ndiv + 1:
            raise ValueError("Number of points must be at least the derivative order + 1.")
        if Np % 2 == 0:
            raise ValueError("The number of points must be odd.")
        from scipy import linalg
        ho = Np >> 1
        x = np.arange(-ho, ho + 1.0)
        x = x[:, np.newaxis]
        X = x ** 0
        for k in range(1, Np):
            X = np.hstack([X, x ** k])
        return np.prod(np.arange(1, ndiv + 1), axis=0) * linalg.inv(X)[ndiv]

    def derivative(func, x0, dx=1.0, n=1, args=(), order=3):
        if order % 2 == 0:
            raise ValueError("Order must be odd.")
        if n >= order:
            raise ValueError("n must be less than order")
        weights = central_diff_weights(order, n)
        val = 0.0
        ho = order >> 1
        for k in range(order):
            val = val + weights[k] * func(x0 + (k - ho) * dx, *args)
        return val / dx ** n

    scipy.misc.central_diff_weights = central_diff_weights
    scipy.misc.derivative = derivative

from splayout import *
import lumapi
import copy
from lumopt.geometries.topology import TopologyOptimization3DLayered
from lumopt.utilities.load_lumerical_scripts import load_from_lsf
from lumopt.figures_of_merit.modematch import ModeMatch
from lumopt.optimization import Optimization
from lumopt.optimizers.generic_optimizers import ScipyOptimizers
from lumopt.utilities.wavelengths import Wavelengths
from func_file_operation import *
from func_get_conversion_matrix import *


# ---------------------------------------------------------------------------
# matplotlib compatibility shim for lumopt's live plot.
#
# lumopt/utilities/plotter.py defines SnapShots(FileMovieWriter) whose
# grab_frame()/finish() use the private helper `FileMovieWriter._frame_sink()`.
# That helper was removed in matplotlib >= 3.8, so the optimization crashed on
# the very first callback with
#     AttributeError: 'SnapShots' object has no attribute '_frame_sink'
# *after* a complete FDTD iteration had already been computed.
#
# The two methods are re-implemented against the current matplotlib API: each
# frame simply overwrites <working_dir>/optimization.png, which keeps the
# convergence plot available without producing a movie.
# ---------------------------------------------------------------------------
from lumopt.utilities import plotter as _lumopt_plotter


def _snapshots_grab_frame(self, **fig_kwargs):
    outfile = getattr(self, "outfile", None)
    if not outfile:
        return
    try:
        self.fig.savefig(outfile, format=self.frame_format, dpi=self.dpi, **fig_kwargs)
    except Exception:
        pass


def _snapshots_finish(self):
    pass


_lumopt_plotter.SnapShots.grab_frame = _snapshots_grab_frame
_lumopt_plotter.SnapShots.finish = _snapshots_finish


# ---------------------------------------------------------------------------
# The optimizer's own penalty is identically zero in this project: the
# minimum-feature-size constraint is applied through the *geometry*
# (TopologyOptimization3DLayered.calc_penalty_term, added to the FOM by
# Optimization itself).  These two functions mirror lumopt's defaults but give
# the gradient analytically, so lumopt never falls back to its
# finite-difference loop.  The argument MUST be called `params` - lumopt
# validates the signature with inspect.signature().
# ---------------------------------------------------------------------------
def _zero_penalty(params):
    return np.zeros(1)


def _zero_penalty_jac(params):
    return np.zeros_like(params)


class MDM_InverseDesign:
    def __init__(self, pitch, shape, rect_size, radius, z_min, z_max, design_region_x, design_region_y, input_wg_width,
                 output_wg_width, num_in_wg, num_out_wg, n_bg, n_wg, wavelength, wavelength_range=100e-9, buff=0,
                 target_fom=1, offset_x=0.0, offset_y=0.0, self_define_materials=True, in_mode_list=None, hide_cad=True,
                 out_mode_list=None, processor='CPU', in_wg_pitch=None):
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
        :param target_fom: targeted FoM value, only modifies the display without any impact on the optimization
        :param offset_x: offset of optimized region in the x direction
        :param offset_y: offset of optimized region in the y direction
        :param self_define_materials: Use the material index defined by User when set True
        :param in_mode_list: mode list of input sources
        :param out_mode_list: mode list of output monitors
        :param in_wg_pitch: centre-to-centre spacing of the input waveguides.  Leave
                         None (default) to keep the evenly-spread formula that
                         `base_*.lsf` has always used; pass a length (e.g. 1.3e-6)
                         to place the inputs symmetrically about y=0 with exactly
                         that pitch - this is what the paper's 2-mode MUX uses.
        """

        self.target_fom = target_fom
        self.fdtd = None
        self.pitch = pitch
        self.shape = shape
        self.rect_size_x = rect_size
        self.rect_size_y = rect_size
        self.radius = radius
        self.z_min = z_min
        self.z_max = z_max
        self.design_region_x = design_region_x
        self.design_region_y = design_region_y
        self.num_in_wg = num_in_wg
        self.num_out_wg = num_out_wg
        self.output_wg_width = output_wg_width
        self.input_wg_width = input_wg_width
        # None -> keep the evenly-spread formula baked into base_*.lsf (legacy behaviour);
        # a number -> exact centre-to-centre pitch, inputs symmetric about y=0.
        self.in_wg_pitch = in_wg_pitch
        self.wavelength = wavelength
        self.wavelength_range = wavelength_range
        self.buff = buff
        self.in_mode_list = in_mode_list
        self.out_mode_list = out_mode_list
        self.mode_num = len(self.in_mode_list)
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.logger = setlogging()
        self.self_define_materials = self_define_materials
        self.cur_path = os.path.dirname(os.path.abspath(__file__))
        self.processor = processor
        self.hide_cad = hide_cad

        self.pixels_per_row = int(self.design_region_x / self.pitch)
        self.pixels_per_col = int(self.design_region_y / self.pitch)
        self.pixels_number = self.pixels_per_row * self.pixels_per_col

        if int(self.design_region_x / self.rect_size_x) != self.pixels_per_row or int(
                self.design_region_y / self.rect_size_y) != self.pixels_per_col:
            assert False, "Pixel per row or pixel per column is not an integer!"

        if len(self.in_mode_list) != len(self.out_mode_list):
            assert False, "Input mode numbers should equal to output mode numbers!"

        # determine the index values either by material names or by numbers
        if isinstance(n_bg, str) and isinstance(n_wg, str):
            c = 299792458
            self.center_wavelength = self.wavelength
            # A short-lived FDTD session is opened only to query the material
            # database; it must always be closed again, otherwise every
            # MDM_InverseDesign instance leaks a licence/session.
            fdtd = lumapi.FDTD(hide=True)
            try:
                self.n_bg = np.real(fdtd.getindex(n_bg, c / self.center_wavelength))[0, 0]
                self.n_wg = np.real(fdtd.getindex(n_wg, c / self.center_wavelength))[0, 0]
            finally:
                try:
                    fdtd.close()
                except Exception:
                    pass
        elif isinstance(n_bg, float) and isinstance(n_wg, float):
            self.n_bg = n_bg
            self.n_wg = n_wg
        else:
            assert False, ("n_bg and n_wg must both be material-name strings or "
                           "both be floats (refractive indices)!")

    #########################################
    ######### Optimization Methods ##########
    #########################################
    ################## TOPOLOGY OPTIMIZATION USING ADJOINT METHOD ######################
    def TO(self, lsf_filename='base_TO.lsf', initial_cond=None, initial_type="uniform", filter_R=100e-9,
           min_feature_size=0, startingBeta=1, start_gradient=0.25, resume=False, epoch=0, max_iter=400,
           num_threads=1, initial_file=None):
        """
        :param lsf_filename: base script file of .lsf
        :param initial_cond: initial index distribution
        :param initial_type: initial type
        :param filter_R: radius of filter
        :param min_feature_size: minimum feature size
        :param startingBeta: initial beta (determines the discreteness)
        :param start_gradient: enforces a rescaling of the gradient to change the optimization parameters by at least this much;
                                          the default value of zero disables automatic scaling.
        :param resume: set to True when you want to resume this optimization from last interruption
        :param epoch: epoch number
        :param max_iter: maximum number of L-BFGS-B iterations.  One iteration
                         costs (number of modes) x (forward + adjoint) FDTD runs,
                         so lower this for a quick smoke test.
        :param num_threads: how many FDTD solves lumopt runs concurrently
                         (default 1).  On a single-GPU machine concurrency gives
                         no speed-up but multiplies peak memory, so 1 is both
                         the safest and, in practice, the fastest setting.
        :param initial_file: path to a `parameters_<n>.npz` checkpoint.  Used
                         together with initial_type="file" to resume a TO run
                         that was interrupted (params AND the beta continuation
                         schedule are restored from the checkpoint).
        :return: return current directory name
        """

        def runSim(params, eps_bg, eps_wg, x_pos, y_pos, z_pos, filter_R,
                   min_feature_size, working_dir, beta, scale_initial_gradient_to, lsf_filename,
                   num_threads=1):

            ######## DEFINE A 3D TOPOLOGY OPTIMIZATION REGION ########
            geometry = TopologyOptimization3DLayered(
                params=params, eps_min=eps_bg, eps_max=eps_wg, x=x_pos, y=y_pos, z=z_pos,
                filter_R=filter_R, min_feature_size=min_feature_size, beta=beta)

            ######## DEFINE FIGURE OF MERIT FOR EACH OUTPUT WAVEGUIDE ########
            foms = [ModeMatch(monitor_name=f'fom{i + 1}', mode_number=self.out_mode_list[i], direction='Forward',
                              target_T_fwd=lambda wl: 1 * np.ones(wl.size), norm_p=2, target_fom=self.target_fom) for i
                    in
                    range(self.mode_num)]

            ######## DEFINE OPTIMIZATION ALGORITHM ########
            # `penalty_fun`/`penalty_jac` are the optimizer's own penalty, which
            # is *separate* from `geometry.calc_penalty_term()` - the latter is
            # added to the FOM directly by Optimization (optimization.py:214).
            # lumopt's default is a constant-zero penalty whose gradient it
            # approximates by finite differences over every single parameter
            # (501*301 = 150,831 of them) on *every* iteration, which is both
            # pointless and crashes on SciPy >= 1.10 (scipy.misc.derivative).
            # Passing the analytic gradient of the (identically zero) penalty
            # is exactly equivalent and avoids that loop.
            optimizer = ScipyOptimizers(max_iter=max_iter, method='L-BFGS-B', scaling_factor=1, pgtol=1e-6,
                                        ftol=1e-4, scale_initial_gradient_to=scale_initial_gradient_to,
                                        penalty_fun=_zero_penalty, penalty_jac=_zero_penalty_jac)

            ######## DEFINE SETUP SCRIPT AND INDIVIDUAL OPTIMIZERS ########
            base_script = self.config_for_lsf(dir_name=self.cur_path, lsf_filename=lsf_filename)

            wavelength = Wavelengths(start=self.wavelength - 0.5 * self.wavelength_range,
                                     stop=self.wavelength + 0.5 * self.wavelength_range, points=11)

            # source name is indexed by mode numbers used in this optimization rather than the actual mode names
            opts = [Optimization(base_script=base_script, wavelengths=wavelength, fom=foms[i],
                                 label=f"TE{self.in_mode_list[i] - 1}", source_name=f"Source_TE_mode{i + 1}_1550",
                                 geometry=geometry, optimizer=optimizer, use_deps=False,
                                 hide_fdtd_cad=self.hide_cad, plot_history=False, store_all_simulations=False,
                                 save_global_index=True, fields_on_cad_only=False) for i in range(self.mode_num)]

            ######## PUT EVERYTHING TOGETHER AND RUN ########
            opt = opts[0]
            for i in range(1, self.mode_num):
                opt += opts[i]
            opt.continuation_max_iter = 40

            ######## RUN THE OPTIMIZER ########
            # num_threads controls how many FDTD solves lumopt runs at once
            # (forward and adjoint pools both use it).  On a single-GPU machine
            # the solves serialise on the GPU anyway, so concurrency buys no
            # speed - but it multiplies the peak memory: each concurrent solve
            # is a separate fdtd-engine process committing ~1.4 GB, on top of
            # the in-process field arrays, which can exhaust the Windows commit
            # limit and abort the run with
            #     _ArrayMemoryError: Unable to allocate 75.9 MiB ...
            opt.run(working_dir=working_dir, num_threads=num_threads)
            # lumopt.chdir()'s into its working directory; restore ours so the
            # rest of the pipeline keeps resolving relative paths correctly.
            os.chdir(self.cur_path)
            runSim.working_dir_used = getattr(opt, "workingDir", None)

        os.makedirs("TO", exist_ok=True)

        size_x = int(self.design_region_x * 1e9)
        size_y = int(self.design_region_y * 1e9)
        size_z = int((self.z_max - self.z_min) * 1e9)

        eps_wg = self.n_wg ** 2
        eps_bg = self.n_bg ** 2

        x_points = int(size_x / 20) + 1
        y_points = int(size_y / 20) + 1
        z_points = int(size_z / 110) + 1

        x_pos = np.linspace(-size_x / 2 * 1e-9, size_x / 2 * 1e-9, x_points)
        y_pos = np.linspace(-size_y / 2 * 1e-9, size_y / 2 * 1e-9, y_points)
        z_pos = np.linspace(-size_z / 2 * 1e-9, size_z / 2 * 1e-9, z_points)

        if not resume:
            working_dir = os.path.join(self.cur_path, "TO",
                                       'x{:04d}_y{:04d}_f{:04d}_m{:04d}'
                                       .format(size_x, size_y, int(filter_R * 1e9), int(min_feature_size * 1e9)))

            # ------------------------------------------------------------------
            # Resume support.
            #
            # lumopt's own TO resume was never implemented in this project (see
            # the `else` branch below), but every L-BFGS-B iteration writes a
            # `parameters_<iter>.npz` checkpoint containing params + beta + eps.
            # Because Optimization.initialize() falls back to
            # geometry.get_current_params() when start_params is None
            # (optimization.py:145), simply seeding the geometry with those
            # params makes the optimizer pick up exactly where it stopped.
            # `initial_type="file"` does that, and it also continues the beta
            # continuation schedule from the checkpoint's beta.
            # ------------------------------------------------------------------
            if initial_type == "file":
                if not initial_file:
                    assert False, "initial_type='file' requires initial_file=<path to parameters_<n>.npz>"
                if not os.path.isfile(initial_file):
                    assert False, "checkpoint not found: {}".format(initial_file)
                ckpt = np.load(initial_file)
                initial_cond = ckpt["params"]
                startingBeta = float(ckpt["beta"])
                if initial_cond.shape != (x_points, y_points):
                    assert False, ("checkpoint shape {} does not match the design grid {} - "
                                   "check design_region_x/y, pitch and filter_R"
                                   .format(initial_cond.shape, (x_points, y_points)))
                disc = 1.0 - float(np.mean(4.0 * initial_cond * (1.0 - initial_cond)))
                print("[EGADO] resuming from {}".format(initial_file))
                print("[EGADO]   beta = {:g}, discreteness = {:.4f}, "
                      "gray pixels = {:.1f}%".format(
                          startingBeta, disc,
                          100.0 * float(np.mean((initial_cond > 0.05) & (initial_cond < 0.95)))))
            else:
                assert is_list_with_two_numbers(initial_cond), \
                    "Initial_cond must be a list of two numbers (mean, range)!"
                if initial_type == "uniform":
                    initial_cond = np.random.rand(x_points, y_points) * initial_cond[0] + initial_cond[1]
                elif initial_type == "blurred":
                    initial_cond = generate_blurred_noise(x_points, y_points, initial_cond[0], initial_cond[1],
                                                          os.path.dirname(working_dir), str(epoch))
                elif initial_type == "perlin":
                    initial_cond = generate_perlin_noise(x_points, y_points, initial_cond[0], initial_cond[1],
                                                         os.path.dirname(working_dir), str(epoch))
                else:
                    assert False, "Initial type must be uniform, blurred, perlin or file!"

            runSim(initial_cond, eps_bg, eps_wg, x_pos, y_pos, z_pos, filter_R=filter_R,
                   min_feature_size=min_feature_size, working_dir=working_dir, beta=startingBeta,
                   scale_initial_gradient_to=start_gradient, lsf_filename=lsf_filename,
                   num_threads=num_threads)
            # lumopt's prepare_working_dir() appends its own numeric suffix and
            # bumps it if the folder already exists, so the directory that was
            # actually used is NOT necessarily `working_dir + "_" + epoch`.
            # Return the real path, otherwise the hand-off to EG-ATD points at
            # a folder that does not exist.
            actual = runSim.working_dir_used
            if actual:
                working_dir = actual
            print("[EGADO] TO working directory: {}".format(working_dir))
        else:
            assert False, "The resume program has not been developed yet, sorry!"

        return working_dir

    ################## ANALOG TO DIGITAL CONVERSION USING ATDO ######################
    def ATDO(self, TO_working_dir, epoch=0, scan_order="row_by_row", resume=False):
        """
        :param TO_working_dir: directory name of TO files
        :param epoch: epoch number (corresponding to the TO files)
        :param scan_order: pixel scan order
        :param resume: set to True when you want to resume the optimization from last interruption
        :return: this function returns nothing but a well-optimized device
        """

        fdtd_filename = self.config_for_a2d(TO_working_dir=TO_working_dir, lsf_filename="base_A2D.lsf")
        file_path = os.path.dirname(fdtd_filename[0])

        logger = self.logger
        fdtd = [FDTDSimulation(hide=self.hide_cad, fdtd_path=LUMERICAL_API,
                               load_file=filename) for filename in fdtd_filename]
        if scan_order == "row_by_row":
            order = np.arange(0, self.pixels_number)
        elif scan_order == "col_by_col":
            order = np.arange(0, self.pixels_number).reshape(-1, self.pixels_per_row).flatten(order="F")
        elif scan_order == "random":
            order = np.random.permutation(self.pixels_number)
        else:
            assert False, "scan_order must be row_by_row, col_by_col, or random"
        A2D_pixel_array = []
        self.fdtd = fdtd

        if resume:
            print("---------------------------------------------------------")
            print("Now we are resuming the interrupted project ...\n")
            npz_path = os.path.join(file_path, "A2D_states.npz")
            iterations = np.load(npz_path)["iterations"]
            order = np.load(npz_path)["order"]
            A2D_pixel_array = np.load(npz_path)["A2D_pixel_array"]
            A2D_pixel_array = A2D_pixel_array.tolist()
            print("---------------------------------------------------------")
            print("Now we are reloading the pixels ...\n")
            self.load_pixels(array=A2D_pixel_array, order=order, to_n=iterations, for_resume_a2d=True)
        else:
            iterations = 0
            for fdtd in self.fdtd:
                fdtd.fdtd.run()
            for fdtd in self.fdtd:
                fdtd.fdtd.switchtolayout()

        # START PIXEL FLIPPING OPERATION
        for n in range(iterations, self.pixels_number):
            for fdtd in self.fdtd:
                self.flipPixel(fdtd, order[n], to_state=1)

            etch = self.cost_function(filename=fdtd_filename)
            etch_cost = self.target_fom * len(etch) - sum(etch)

            for fdtd in self.fdtd:
                self.flipPixel(fdtd, order[n], to_state=0)

            deposition = self.cost_function(filename=fdtd_filename)
            deposition_cost = self.target_fom * len(deposition) - sum(deposition)

            etch_info = "; ".join(["TE%d:%.3f" % (self.out_mode_list[i], etch[i]) for i in range(len(etch))])
            deposit_info = "; ".join(
                ["TE%d:%.3f" % (self.out_mode_list[i], deposition[i]) for i in range(len(deposition))])
            progress_info = "%.3f%%" % ((n + 1) / self.pixels_number * 100)

            if etch_cost > deposition_cost:
                # state 0 = deposited (silicon)
                log_info = (f"[Etch]{etch_info}; [Deposit]{deposit_info}. "
                            f"Pixel {order[n]} is deposited; [Progress]{progress_info}")
                logger.info(log_info)
                A2D_pixel_array.append(0)
            else:
                # Flip the pixel back to state 1 ("etched", i.e. a hole)
                for fdtd in self.fdtd:
                    self.flipPixel(fdtd, order[n], to_state=1)
                log_info = (f"[Etch]{etch_info}; [Deposit]{deposit_info}. "
                            f"Pixel {order[n]} is etched; [Progress]{progress_info}")
                logger.info(log_info)
                A2D_pixel_array.append(1)

            np.savez(os.path.join(file_path, "A2D_states.npz"), iterations=n + 1, A2D_pixel_array=A2D_pixel_array,
                     order=order)

        for fdtd in self.fdtd:
            fdtd.fdtd.close()

        # rearrange the pixel array in this order: 1, 2, 3, 4 ...
        order_original = np.argsort(order)
        for i in range(self.pixels_number):
            temp = copy.deepcopy(A2D_pixel_array)
            A2D_pixel_array[i] = temp[order_original[i]]

        pixel_array = np.array(A2D_pixel_array).reshape(self.pixels_per_row, self.pixels_per_col)
        pixel_array = np.ravel(pixel_array, order='F')
        np.savez(os.path.join(file_path, 'initial_DBS.npz'), initial_DBS=pixel_array)
        np.savez(os.path.join(file_path, 'scan_order.npz'), order=order)

        if os.path.exists(os.path.join(file_path, "A2D_states.npz")):
            os.remove(os.path.join(file_path, "A2D_states.npz"))

        move_files_to(file_path, os.path.join(file_path, "data"))
        new_dirname = "data_epoch_" + str(epoch)
        os.rename(os.path.join(file_path, "data"), os.path.join(file_path, new_dirname))

    ################## ANALOG TO DIGITAL CONVERSION USING EG-ATDO ######################
    def EG_ATDO(self, TO_working_dir, epoch=0, scan_order="row_by_row", resume=False, resume_from_cur_file=False):
        """
        :param TO_working_dir: directory name of TO files
        :param epoch: epoch number (corresponding to the TO files)
        :param scan_order: pixel scan order
        :param resume: set to True when you want to resume the optimization from last interruption
        :return: this function returns nothing but a well-optimized device
        :param resume_from_cur_file: resume without reloading the pixels
        """

        fdtd_filename, conversion_matrix = self.config_for_eg_a2d(TO_working_dir=TO_working_dir,
                                                                  lsf_filename="base_EGA2D.lsf",
                                                                  resume=resume_from_cur_file)
        file_path = os.path.dirname(fdtd_filename[0])


        logger = self.logger
        fdtd = [FDTDSimulation(hide=self.hide_cad, fdtd_path=LUMERICAL_API,
                               load_file=filename) for filename in fdtd_filename]

        edge = []
        etch_convert = []
        deposit_convert = []

        # 1.Obtain direct_etch_conversion matrix and direct_deposition_conversion matrix through conversion matrix
        # 2.Obtain edge pixel matrix which needs to be further determined through flip-pixel operation
        # Warning: matrix values are sometimes a little bit away from 0, 0.5, and 1
        for row_idx, row in enumerate(conversion_matrix):
            for col_idx, col in enumerate(row):
                if (col > 0.49) and (col < 0.51):
                    edge.append(row_idx * len(row) + col_idx)
                if col > 0.99:
                    etch_convert.append(row_idx * len(row) + col_idx)
                if col < 0.01:
                    deposit_convert.append(row_idx * len(row) + col_idx)

        # Pixels whose grey value is neither ~0, ~0.5 nor ~1 fall into none of
        # the three lists above: they are never decided by the flip-pixel sweep
        # and keep their grey value in initial_DBS.npz (which is why
        # appendix_read_npz.py exists).  This only happens if the TO result is
        # not strongly binarised, so report it instead of failing silently.
        n_classified = len(edge) + len(etch_convert) + len(deposit_convert)
        n_undecided = self.pixels_number - n_classified
        print("Pixel classification: {} deposited + {} etched + {} edge = {} / {}".format(
            len(deposit_convert), len(etch_convert), len(edge), n_classified, self.pixels_number))
        if n_undecided > 0:
            print("WARNING: {} pixel(s) ({:.2f}%) have an intermediate grey value "
                  "(0.01 < v < 0.49 or 0.51 < v < 0.99) and will NOT be decided by "
                  "FDTD; they stay grey in initial_DBS.npz. Run the TO phase to full "
                  "binarisation (beta continuation) or binarise the TO result first."
                  .format(n_undecided, 100.0 * n_undecided / self.pixels_number))

        if scan_order == "row_by_row":
            order = edge
        elif scan_order == "col_by_col":
            position = np.arange(self.pixels_number).reshape((self.pixels_per_col, self.pixels_per_row))
            order = []
            for col in range(self.pixels_per_row):
                for row in range(self.pixels_per_col):
                    if position[row][col] in edge:
                        order.append(position[row][col])
            order = np.array(order)
        else:
            assert False, "Scan_order must be row_by_row, col_by_col! Random_order is under developing!"
        A2D_pixel_array = conversion_matrix.reshape(-1)
        self.fdtd = fdtd

        if resume:
            # In EG-ATD, the saved pixel array is in proper sequence (i.e., row_by_row)
            print("---------------------------------------------------------")
            print("Now we are resuming the interrupted project ...\n")
            npz_path = os.path.join(file_path, "A2D_states.npz")
            iterations = np.load(npz_path)["iterations"]
            order = np.load(npz_path)["order"]
            A2D_pixel_array = np.load(npz_path)["A2D_pixel_array"]
            A2D_pixel_array = A2D_pixel_array.tolist()
            if resume_from_cur_file is False:
                print("---------------------------------------------------------")
                print("Now we are reloading the pixels ...\n")
                self.load_pixels_eg(array=A2D_pixel_array, for_resume_a2d=True)
                print("---------------------------------------------------------")
                print("Pixel loading is done. Now the flip-pixel sweep begins ...\n")
            else:
                print("---------------------------------------------------------")
                print("The optimization is resumed from current files, which don't require reloading the pixels!")
        else:
            iterations = 0
            print("---------------------------------------------------------")
            print("Now we are adding directly-converted pixels ...\n")
            for fdtd in self.fdtd:
                for i in range(len(etch_convert)):
                    self.flipPixel(fdtd, etch_convert[i], to_state=1)
                for i in range(len(deposit_convert)):
                    self.flipPixel(fdtd, deposit_convert[i], to_state=0)
            print("---------------------------------------------------------")
            print("Direct conversion is done. Now the flip-pixel sweep begins ...\n")

        # The .fsp files are closed and reopened periodically: a long-running
        # Lumerical session becomes unstable after a few hundred iterations.
        # NOTE: the original code incremented `reopen_flag` *after* resetting it,
        # so the branch fired on every single iteration instead of every second
        # one.  The interval is now explicit and configurable.
        REOPEN_INTERVAL = 50
        for n in range(iterations, len(order)):
            if n > iterations and (n - iterations) % REOPEN_INTERVAL == 0:
                print("---------------------------------------------------------")
                print("To prevent annoying stop of optimization, we close those fdtd files and reopen them now...")
                for i in range(len(fdtd_filename)):
                    self.fdtd[i].fdtd.eval('save("%s");' % fdtd_filename[i])
                    # if we don't save fdtd files, the previous etched-back pixel will not be saved,
                    # and all pixels will be deposited
                for fdtd in self.fdtd:
                    fdtd.fdtd.close()

                fdtd = [FDTDSimulation(hide=self.hide_cad, fdtd_path=LUMERICAL_API,
                                       load_file=filename) for filename in fdtd_filename]
                self.fdtd = fdtd

            print(f"Starting ITER.{n} of total {len(order)} iterations...")
            for fdtd in self.fdtd:
                print("Now we etch this pixel...")
                self.flipPixel(fdtd, order[n], to_state=1)

            etch = self.cost_function(filename=fdtd_filename)
            etch_cost = self.target_fom * len(etch) - sum(etch)

            for fdtd in self.fdtd:
                print("Now we deposit this pixel...")
                self.flipPixel(fdtd, order[n], to_state=0)

            deposition = self.cost_function(filename=fdtd_filename)
            deposition_cost = self.target_fom * len(deposition) - sum(deposition)

            etch_info = "; ".join(["TE%d:%.3f" % (self.out_mode_list[i], etch[i]) for i in range(len(etch))])
            deposit_info = "; ".join(
                ["TE%d:%.3f" % (self.out_mode_list[i], deposition[i]) for i in range(len(deposition))])
            progress_info = "%.3f%%" % ((n + 1) / len(order) * 100)

            if etch_cost > deposition_cost:
                print("Deposit is better, the pixel remains deposited!")
                log_info = f"[Etch]{etch_info}; [Deposit]{deposit_info}. Pixel {order[n]} is deposited; [Progress]{progress_info}"
                logger.info(log_info)
                A2D_pixel_array[order[n]] = 0
            else:
                # Flip the pixel state back to 1 ("deposition")
                print("Etch is better, the pixel is etched back!")
                for fdtd in self.fdtd:
                    self.flipPixel(fdtd, order[n], to_state=1)
                log_info = f"[Etch]{etch_info}; [Deposit]{deposit_info}. Pixel {order[n]} is etched; [Progress]{progress_info}"
                logger.info(log_info)
                A2D_pixel_array[order[n]] = 1

            np.savez(os.path.join(file_path, "A2D_states.npz"), iterations=n + 1, A2D_pixel_array=A2D_pixel_array,
                     order=order)

        for fdtd in self.fdtd:
            fdtd.fdtd.close()

        pixel_array = np.array(A2D_pixel_array).reshape(self.pixels_per_row, self.pixels_per_col)
        pixel_array = np.ravel(pixel_array, order='F')
        np.savez(os.path.join(file_path, 'initial_DBS.npz'), initial_DBS=pixel_array)
        np.savez(os.path.join(file_path, 'scan_order.npz'), order=order)

        if os.path.exists(os.path.join(file_path, "A2D_states.npz")):
            os.remove(os.path.join(file_path, "A2D_states.npz"))

        move_files_to(file_path, os.path.join(file_path, "data"))
        new_dirname = "data_epoch_" + str(epoch)
        os.rename(os.path.join(file_path, "data"), os.path.join(file_path, new_dirname))

    ################## DBS OPTIMIZATION USING OPEN-ACCESS LIBRARY ######################
    def DBS(self, resume, epoch, scan_order="row_by_row"):
        """
        :param resume: set to True when you want to resume the interrupted optimization
        :param epoch: epoch number
        :param scan_order: pixel scan order
        :return: return nothing but a well-optimized device
        """

        # ATDO/EG_ATDO rename their output folder to `data_epoch_<epoch>`;
        # the original code looked for `data_<epoch>` and therefore always
        # failed with FileNotFoundError.  Accept both spellings.
        a2d_root = os.path.join(self.cur_path, "A2D")
        candidates = [os.path.join(a2d_root, "data_epoch_{}".format(epoch)),
                      os.path.join(a2d_root, "data_{}".format(epoch)),
                      os.path.join(a2d_root, "data")]
        data_dir = next((d for d in candidates if os.path.isfile(
            os.path.join(d, "initial_DBS.npz"))), None)
        if data_dir is None:
            assert False, ("initial_DBS.npz not found. Looked in: {}".format(candidates))
        initial_condition = np.load(os.path.join(data_dir, "initial_DBS.npz"))['initial_DBS']
        fdtd_filename = self.config_for_dbs(lsf_filename="base_DBS.lsf")
        # definitions for gdsii file creation
        design_region_width = self.design_region_y * 1e6
        design_region_length = self.design_region_x * 1e6
        file_path = os.path.dirname(fdtd_filename[0])
        data_path = os.path.join(file_path, "data")

        self.logger = setlogging(txt=os.path.join(file_path, "log.txt"))

        if self.self_define_materials:
            etch = "n_bg"
        else:
            etch = "SiO2 (Glass) - Palik"
        self.fdtd = []
        pixels = []

        # initialize the simulation frame
        for filename in fdtd_filename:
            if "initial" not in filename:
                self.fdtd.append(FDTDSimulation(hide=self.hide_cad, fdtd_path=LUMERICAL_API,
                                                load_file=filename.replace("_model", "")))

        ## pixels region definition
        if self.shape == "rect":
            for fdtd in self.fdtd:
                pixels.append(RectanglePixelsRegion(Point(-design_region_length / 2 + self.offset_x * 1e6,
                                                          -design_region_width / 2 + self.offset_y * 1e6),
                                                    Point(design_region_length / 2 + self.offset_x * 1e6,
                                                          design_region_width / 2 + self.offset_y * 1e6),
                                                    pixel_x_length=self.rect_size_x * 1e6,
                                                    pixel_y_length=self.rect_size_y * 1e6, fdtd_engine=fdtd,
                                                    material=etch, z_start=self.z_min * 1e6, z_end=self.z_max * 1e6))
        elif self.shape == "circle":
            for fdtd in self.fdtd:
                pixels.append(CirclePixelsRegion(Point(-design_region_length / 2 + self.offset_x * 1e6,
                                                       -design_region_width / 2 + self.offset_y * 1e6),
                                                 Point(design_region_length / 2 + self.offset_x * 1e6,
                                                       design_region_width / 2 + self.offset_y * 1e6),
                                                 pixel_radius=self.radius * 1e6, fdtd_engine=fdtd, material=etch,
                                                 z_start=self.z_min * 1e6, z_end=self.z_max * 1e6))

        def cost_function(based_matrix):
            for i in range(len(self.fdtd)):
                pixels[i].update(based_matrix.reshape(self.pixels_per_row, self.pixels_per_col))
                self.fdtd[i].fdtd.eval('save("%s");' % fdtd_filename[i].replace("_model", ""))
                self.fdtd[0].fdtd.eval('addjob("%s");' % fdtd_filename[i].replace("_model", ""))

            self.fdtd[0].fdtd.eval('runjobs;')

            for i in range(len(self.fdtd)):
                self.fdtd[i].fdtd.eval('load("%s");' % fdtd_filename[i].replace("_model", ""))

            mode_transmission = []
            for i, fdtd in enumerate(self.fdtd):
                mode_transmission.append(np.mean(fdtd.get_mode_transmission(f"fom{i + 1}_mode_exp")[0, 1, :]))

            fom_info = "; ".join(
                ["TE%d:%.3f" % (self.out_mode_list[i], mode_transmission[i]) for i in range(len(mode_transmission))])

            self.logger.info(fom_info)
            FOM = len(mode_transmission) - sum(mode_transmission)
            return FOM

        def final_file_arrange():
            move_files_to(file_path, data_path)
            new_dirname = "data_epoch_" + str(epoch)
            os.rename(data_path, os.path.join(file_path, new_dirname))

        def call_back():
            pass

        # DEFINE DBS OPTIMIZATION INSTANCE
        initial_array = initial_condition
        DBS = DirectBinarySearchAlgorithm(loS=self.pixels_number, pixels_per_row=self.pixels_per_row,
                                          pixels_per_col=self.pixels_per_col,
                                          cost_function=cost_function, max_iteration=400, threshold=0.5e-2,
                                          scan_order=scan_order,
                                          callback_function=call_back, initial_solution=initial_array,
                                          logger=self.logger,
                                          resume=resume, data_savepath=data_path,
                                          )

        # RUN DBS OPTIMIZATION
        DBS.run()
        # SAVE OPTIMIZATION RESULTS
        npztomatlab(os.path.join(data_path, "final_DBS.mat"), name="final_DBS", array=DBS.best_solution)

        for fdtd in self.fdtd:
            fdtd.fdtd.close()
        final_file_arrange()

    #########################################
    ########## Functional Methods ###########
    #########################################
    class Simulation:
        def __init__(self, fdtdname):
            self.eta = 0.5
            self.dx = 20e-9
            self.dy = 20e-9
            self.dz = 20e-9

            self.eps_max = None
            self.eps_min = None

            self.params = None
            self.sim = None
            self.simname = fdtdname
            self.beta = None

            self.eps = None

            self.x = None
            self.y = None
            self.z = None

        def getnpzsim(self, npzfilename: str):
            if npzfilename[-3:] == 'npz':
                data = np.load(npzfilename)
                self.params = data['params']
                self.eps_max = data['eps_max']
                self.eps_min = data['eps_min']
                self.x = data['x']
                self.y = data['y']
                self.z = data['z']
                self.beta = data['beta']
                self.eps = data['eps']
                self.sim = lumapi.FDTD(filename=self.simname)
                self.sim.switchtolayout()

        def get_eps_from_params(self):
            rho = np.reshape(self.params, (len(self.x), len(self.y)))

            ## Use script function to convert the raw parameters to a permittivity distribution and get the result
            self.sim.putv("topo_rho", rho)
            self.sim.eval(('params = struct;'
                           'params.eps_levels=[{0},{1}];'
                           'params.filter_radius = {2};'
                           'params.beta = {3};'
                           'params.eta = {4};'
                           'params.dx = {5};'
                           'params.dy = {6};'
                           'params.dz = 0.0;'
                           'eps_geo = topoparamstoindex(params,topo_rho);').format(self.eps_min, self.eps_max,
                                                                                   self.filter_R,
                                                                                   self.beta, self.eta, self.dx,
                                                                                   self.dy))
            eps = self.sim.getv("eps_geo")

            return eps

        def set_spatial_interp(self, monitor_name, setting):
            script = 'select("{}");set("spatial interpolation","{}");'.format(monitor_name, setting)
            self.sim.eval(script)

        def add_geo(self, npzfilename: str, filter_R=100e-9, min_feature_size=0.0):
            self.filter_R = filter_R
            self.min_feature_size = min_feature_size
            self.getnpzsim(npzfilename=npzfilename)
            eps = self.get_eps_from_params()

            script = ('addimport;'
                      'set("detail",1);')
            self.sim.eval(script)

            full_eps = np.broadcast_to(eps[:, :, None], (
                len(self.x), len(self.y), len(self.z)))  # < TODO: Move to Lumerical script to reduce transfers

            self.sim.putv('x_geo', self.x)
            self.sim.putv('y_geo', self.y)
            self.sim.putv('z_geo', self.z)
            self.sim.putv('eps_geo', full_eps)

            ## We delete and re-add the import to avoid a warning
            script = ('select("import");'
                      'delete;'
                      'addimport;'
                      'importnk2(sqrt(eps_geo),x_geo,y_geo,z_geo);')
            self.sim.eval(script)
            self.sim.save()
            self.sim.close()

    def flipPixel(self, fdtd, n, to_state):
        if self.shape == 'rect':
            fdtd.fdtd.eval("switchtolayout;")
            i = n // self.pixels_per_row
            j = n % self.pixels_per_row
            if to_state == 1:
                fdtd.fdtd.putv('n', n)
                fdtd.fdtd.eval('''select("pixel"+num2str(n));delete;''')
                fdtd.fdtd.addrect(
                    name="pixel" + str(n),
                    material="n_bg" if self.self_define_materials else "SiO2 (Glass) - Palik",
                    x=-self.design_region_x / 2 + j * self.pitch + self.pitch / 2 + self.offset_x,
                    y=self.design_region_y / 2 - self.pitch / 2 - i * self.pitch + self.offset_y,
                    x_span=self.rect_size_x,
                    y_span=self.rect_size_y,
                    z_max=self.z_max,
                    z_min=self.z_min,
                    override_mesh_order_from_material_database=True,
                    mesh_order=1)
            elif to_state == 0:
                fdtd.fdtd.putv('n', n)
                fdtd.fdtd.eval('''select("pixel"+num2str(n));delete;''')
                fdtd.fdtd.addrect(
                    name="pixel" + str(n),
                    material="n_wg" if self.self_define_materials else "Si (Silicon) - Palik",
                    x=-self.design_region_x / 2 + j * self.pitch + self.pitch / 2 + self.offset_x,
                    y=self.design_region_y / 2 - self.pitch / 2 - i * self.pitch + self.offset_y,
                    x_span=self.rect_size_x,
                    y_span=self.rect_size_y,
                    z_max=self.z_max,
                    z_min=self.z_min,
                    override_mesh_order_from_material_database=True,
                    mesh_order=1)
            else:
                raise ValueError('State Invalid value')
        elif self.shape == "circle":
            fdtd.fdtd.eval("switchtolayout;")
            i = n // self.pixels_per_row
            j = n % self.pixels_per_row
            if to_state == 1:
                fdtd.fdtd.putv('n', n)
                fdtd.fdtd.eval('''select("pixel"+num2str(n));delete;''')
                fdtd.fdtd.addrect(
                    name="pixel" + str(n),
                    material="n_wg" if self.self_define_materials else "Si (Silicon) - Palik",
                    x=-self.design_region_x / 2 + j * self.pitch + self.pitch / 2 + self.offset_x,
                    y=self.design_region_y / 2 - self.pitch / 2 - i * self.pitch + self.offset_y,
                    x_span=self.rect_size_x,
                    y_span=self.rect_size_y,
                    z_max=self.z_max,
                    z_min=self.z_min,
                    override_mesh_order_from_material_database=True,
                    mesh_order=1)
                fdtd.fdtd.addcircle(
                    name="pixel" + str(n),
                    material="n_bg" if self.self_define_materials else "SiO2 (Glass) - Palik",
                    x=-self.design_region_x / 2 + j * self.pitch + self.pitch / 2 + self.offset_x,
                    y=self.design_region_y / 2 - self.pitch / 2 - i * self.pitch + self.offset_y,
                    radius=self.radius,
                    z_max=self.z_max,
                    z_min=self.z_min,
                    override_mesh_order_from_material_database=True,
                    mesh_order=1)
            elif to_state == 0:
                fdtd.fdtd.putv('n', n)
                fdtd.fdtd.eval('''select("pixel"+num2str(n));delete;''')
                fdtd.fdtd.addrect(
                    name="pixel" + str(n),
                    material="n_wg" if self.self_define_materials else "Si (Silicon) - Palik",
                    x=-self.design_region_x / 2 + j * self.pitch + self.pitch / 2 + self.offset_x,
                    y=self.design_region_y / 2 - self.pitch / 2 - i * self.pitch + self.offset_y,
                    x_span=self.rect_size_x,
                    y_span=self.rect_size_y,
                    z_max=self.z_max,
                    z_min=self.z_min,
                    override_mesh_order_from_material_database=True,
                    mesh_order=1)
            else:
                raise ValueError('State Invalid value')

    def load_pixels(self, array, order=None, to_n=None, fdtd=None, for_resume_a2d=False):
        if for_resume_a2d:
            if to_n is None:
                assert False, "for resume A2D, to_n must be specified"
            for fdtd in self.fdtd:
                for i in range(to_n):
                    self.flipPixel(fdtd, order[i], to_state=array[i])
        else:
            for i in range(self.pixels_number):
                if array[i] in (0, 1):
                    # only add pixels whose values equal 0 or 1
                    self.flipPixel(fdtd, i, to_state=array[i])

    def load_pixels_eg(self, array, fdtd=None, for_resume_a2d=False):
        load_order = np.arange(0, self.pixels_number)
        if for_resume_a2d:
            for fdtd in self.fdtd:
                for i in range(self.pixels_number):
                    if array[i] in (0, 1):
                        # only add pixels whose values equal 0 or 1
                        self.flipPixel(fdtd, load_order[i], to_state=array[i])
        else:
            # work for the function-input FDTD rather the self.fdtd
            for i in range(self.pixels_number):
                if array[i] in (0, 1):
                    # only add pixels whose values equal 0 or 1
                    self.flipPixel(fdtd, load_order[i], to_state=array[i])

    def cost_function(self, filename):
        print("Now we run FDTD to get FOM...")
        for i in range(len(filename)):
            self.fdtd[i].fdtd.eval('save("%s");' % filename[i])
            print(f"FDTD file {i} saved...")
            self.fdtd[0].fdtd.eval('addjob("%s");' % filename[i])
            print(f"FDTD job {i} added to the first FDTD GUI...")
        print("Running FDTD...")
        self.fdtd[0].fdtd.eval('runjobs;')

        for i in range(len(filename)):
            # self.fdtd[i].fdtd.eval('save("%s");' % filename[i])
            # print("FDTD completed and files saved...")
            self.fdtd[i].fdtd.eval('load("%s");' % filename[i])
            print(f"FDTD {i} loaded...")

        mode_transmission = []
        for i, fdtd in enumerate(self.fdtd):
            mode_transmission.append(np.mean(fdtd.get_mode_transmission(f"fom{i + 1}_mode_exp")[0, 1, :]))
            print("Monitor data obtained...")

        return mode_transmission

    def get_digital_array_from_text(self, file):
        result = [0] * self.pixels_number  # 初始化结果列表为全零
        i = -1
        with open(file, 'r') as f:
            for line in f:
                i += 1
                last_item = line.strip().split()[-1]
                if last_item == 'deposited':
                    result[i] = 0
                elif last_item == 'etched':
                    result[i] = 1
                else:
                    raise ValueError('Invalid value')
            if i != (self.pixels_number - 1):
                raise ValueError('Invalid length')

        # transform to splayout DBS algorithm matrix order
        pixel_array = np.array(result).reshape(self.pixels_per_row, self.pixels_per_col)
        pixel_array = np.ravel(pixel_array, order='F')
        np.savez('initial_DBS.npz', initial_DBS=pixel_array)

        return pixel_array

    #########################################
    ######### Configuration Methods #########
    #########################################
    def config_for_lsf(self, dir_name, lsf_filename):
        base_script = load_from_lsf(os.path.join(dir_name, lsf_filename))
        base_script = base_script.replace("opt_size_x=none", "opt_size_x={}".format(self.design_region_x))
        base_script = base_script.replace("opt_size_y=none", "opt_size_y={}".format(self.design_region_y))
        base_script = base_script.replace("num_in_wg=none", "num_in_wg={}".format(self.num_in_wg))
        base_script = base_script.replace("num_out_wg=none", "num_out_wg={}".format(self.num_out_wg))
        base_script = base_script.replace("lam_c=none", "lam_c={}".format(self.center_wavelength))
        base_script = base_script.replace("out_wg_width=none", "out_wg_width={}".format(self.output_wg_width))
        base_script = base_script.replace("in_wg_width=none", "in_wg_width={}".format(self.input_wg_width))
        # in_wg_pitch=0 makes base_*.lsf fall back to its own original evenly-spread
        # formula; a positive value selects the explicit-pitch branch instead.
        base_script = base_script.replace(
            "in_wg_pitch=none",
            "in_wg_pitch={}".format(0 if self.in_wg_pitch is None else self.in_wg_pitch))
        base_script = base_script.replace("wavelength_span=none",
                                          "wavelength_span={}".format(self.wavelength_range))
        base_script = base_script.replace("in_mode_list=none", "in_mode_list={}".format(self.in_mode_list))
        base_script = base_script.replace("out_mode_list=none", "out_mode_list={}".format(self.out_mode_list))
        base_script = base_script.replace("mode_num=none", "mode_num={}".format(self.mode_num))

        # ------------------------------------------------------------------
        # Solver resource selection.
        #
        # The original code appended `setresource("FDTD","CPU",true);` (or the
        # GPU equivalent) to the script.  That three-argument form no longer
        # exists in Lumerical 2025 R1: setresource() is now
        # setresource(<solver>, <config index>, <property>, <value>) and the
        # local resource configuration exposes only name/active/hostname/
        # processes/threads - there is no GPU property at all (the engine is
        # launched with -gpu according to the application-level Resource
        # Manager).  Because the appended call raised an error, the *entire*
        # base script failed to evaluate and the optimization never started.
        #
        # The call is therefore no longer appended.  If your installation
        # really needs an explicit resource statement, put it in
        # `self.resource_script`; it is applied on a best-effort basis.
        # ------------------------------------------------------------------
        if self.processor == 'GPU':
            script = ('select("FDTD");'
                      'set("z min bc","PML");'
                      'set("express mode",true);'
                      )
        else:
            script = ''
        base_script = base_script + script
        return base_script

    def apply_resource_setting(self, fdtd):
        """Best-effort activation of the local FDTD resource configuration.

        Safe to call on any Lumerical version: failures are reported, never
        raised, because a missing resource statement must not abort a run.
        """
        if not getattr(self, "resource_script", None):
            return
        try:
            fdtd.eval(self.resource_script)
            print("[EGADO] applied resource_script: {}".format(self.resource_script))
        except Exception as exc:
            print("[EGADO] could not apply resource_script ({}): {}".format(
                type(exc).__name__, self.resource_script))

    def config_for_a2d(self, TO_working_dir, lsf_filename=None):
        print(TO_working_dir)
        filter_R = int(TO_working_dir.split("_f")[1][:4]) * 1e-9
        min_feature_size = int(TO_working_dir.split("_m")[1][:4]) * 1e-9
        target_folder = os.path.join(self.cur_path, "A2D")
        copy_specified_filename_to(target_folder=target_folder, filename="_A2D")
        param_path = export_TO_param_for_a2d(working_dir=TO_working_dir, type="MAX")
        file_list = []
        if lsf_filename:
            self.config_for_a2d_fdtd_file(target_folder, lsf_filename)

        for file in os.listdir(target_folder):
            if file.endswith(".fsp"):
                fdtd = self.Simulation(fdtdname=os.path.join(target_folder, file))
                fdtd.add_geo(npzfilename=param_path, filter_R=filter_R, min_feature_size=min_feature_size)
                file_list.append(os.path.join(target_folder, file))

        return file_list

    @staticmethod
    def _force_self_define_materials(base_script, value):
        """Set `self_define_materials` in an .lsf script to a literal true/false.

        The original implementation sliced the current value with [:4] and
        replaced that substring, which corrupts the script when the value is
        `false` (`"fals"` -> `"false"` leaves a stray `e;`).
        """
        import re
        return re.sub(r"self_define_materials\s*=\s*(true|false)",
                      "self_define_materials={}".format("true" if value else "false"),
                      base_script)

    def config_for_a2d_fdtd_file(self, target_folder, lsf_filename):
        base_script = self.config_for_lsf(dir_name=target_folder, lsf_filename=lsf_filename)
        base_script = self._force_self_define_materials(base_script, self.self_define_materials)

        for i in range(self.mode_num):
            temp_fdtd = lumapi.FDTD(hide=self.hide_cad)
            temp_fdtd.putv("current_source", i + 1)
            script = (
                'for (i=1:mode_num){setnamed("Source_TE_mode"+num2str(i)+"_1550", "enabled", (i==current_source));}'
                'setglobalmonitor("frequency points", 11);')
            temp_fdtd.eval(base_script + script)
            fdtd_file = f"TE_mode_{i}_A2D.fsp"
            temp_fdtd.save(os.path.join(target_folder, fdtd_file))
            temp_fdtd.close()

    def config_for_eg_a2d(self, TO_working_dir, lsf_filename=None, resume=False):
        print(TO_working_dir)
        filter_R = int(TO_working_dir.split("_f")[1][:4]) * 1e-9
        min_feature_size = int(TO_working_dir.split("_m")[1][:4]) * 1e-9
        target_folder = os.path.join(self.cur_path, "EGA2D")
        copy_specified_filename_to(target_folder=target_folder, filename="_EGA2D")
        param_path = export_TO_param_for_a2d(working_dir=TO_working_dir, type="MAX")
        conversion_matrix = get_conversion_matrix(analog_file=param_path, target_folder=target_folder,
                                                  pixel_size=self.rect_size_x, buff=self.buff)

        file_list = []
        if resume is False:
            if lsf_filename:
                self.config_for_eg_a2d_fdtd_file(target_folder, lsf_filename)
            for i in range(self.mode_num):
                for file in os.listdir(target_folder):
                    if file.endswith(f"{i}_EGA2D.fsp"):
                        fdtd = self.Simulation(fdtdname=os.path.join(target_folder, file))
                        fdtd.add_geo(npzfilename=param_path, filter_R=filter_R, min_feature_size=min_feature_size)
                        file_list.append(os.path.join(target_folder, file))
        else:
            for i in range(self.mode_num):
                for file in os.listdir(target_folder):
                    if file.endswith(f"{i}_EGA2D.fsp"):
                        file_list.append(os.path.join(target_folder, file))
        print(file_list)
        return file_list, conversion_matrix

    def config_for_eg_a2d_fdtd_file(self, target_folder, lsf_filename):
        base_script = self.config_for_lsf(dir_name=target_folder, lsf_filename=lsf_filename)
        base_script = self._force_self_define_materials(base_script, self.self_define_materials)

        for i in range(self.mode_num):
            temp_fdtd = lumapi.FDTD(hide=self.hide_cad)
            temp_fdtd.putv("current_source", i + 1)
            script = (
                'for (i=1:mode_num){setnamed("Source_TE_mode"+num2str(i)+"_1550", "enabled", (i==current_source));}'
                'setglobalmonitor("frequency points", 11);')
            temp_fdtd.eval(base_script + script)
            fdtd_file = f"TE_mode_{i}_EGA2D.fsp"
            temp_fdtd.save(os.path.join(target_folder, fdtd_file))
            temp_fdtd.close()

    def config_for_dbs(self, lsf_filename=None):
        target_folder = os.path.join(self.cur_path, "DBS")
        copy_specified_filename_to(target_folder=target_folder, filename="_DBS")
        file_list = []
        if lsf_filename:
            self.config_for_dbs_fdtd_file(target_folder, lsf_filename)

        for file in os.listdir(target_folder):
            if file.endswith(".fsp"):
                file_list.append(os.path.join(target_folder, file))
        return file_list

    def config_for_dbs_fdtd_file(self, target_folder, lsf_filename):
        base_script = self.config_for_lsf(target_folder, lsf_filename)
        base_script = self._force_self_define_materials(base_script, self.self_define_materials)
        for i in range(self.mode_num):
            temp_fdtd = lumapi.FDTD(hide=self.hide_cad)
            temp_fdtd.putv("current_source", i + 1)
            script = (
                'for (i=1:mode_num){setnamed("Source_TE_mode"+num2str(i)+"_1550", "enabled", (i==current_source));}'
                'setglobalmonitor("frequency points", 11);')
            temp_fdtd.eval(base_script + script)
            fdtd_file = f"TE_mode_{i}_DBS.fsp"
            temp_fdtd.save(os.path.join(target_folder, fdtd_file))
            temp_fdtd.close()

# End of this file
