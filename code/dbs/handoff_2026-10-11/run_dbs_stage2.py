"""Stage 2: Two-Cell Joint Action DBS Engine for Scheme B.

Adheres strictly to 《B结构修复复算_结果校核与下一步建议.md》 & 《两方案DBS_跨电脑操作指南.md》:
1. Starts from Stage 1 converged baseline (J = 0.88645, 0 DRC violations).
2. Explores pairwise adjacent joint actions (4-connected horizontal & vertical pairs in 108 grid):
   - Same-state pair inversion: (0,0) -> (1,1) [add 240nm Si bar] or (1,1) -> (0,0) [etch 240nm void].
   - Diff-state pair swap: (0,1) -> (1,0) or (1,0) -> (0,1) [joint 120nm shift].
3. Real-time dual DRC gate before FDTD:
   - Level 1: 120nm supplemental width/space check (0.05s).
   - Level 2: Official 248-category PDK DRC check (1.4s).
   Only candidates with 0 violations proceed to dual-source forward FDTD.
4. Concurrent dual-source FDTD with persistent GPU mode and IPC retry.
5. Checkpoints per step, resume capability, and full history logging.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

# Set environment
os.environ["OMP_NUM_THREADS"] = "20"
LUMERICAL_API = r"D:\Program Files\ANSYS Inc\v251\Lumerical\api\python"
if LUMERICAL_API not in sys.path and os.path.isdir(LUMERICAL_API):
    sys.path.insert(0, LUMERICAL_API)

import lumapi
import klayout.db as db

# Ensure tools can be imported
PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT / "tools"))
import geometry_io
import run_drc


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def ensure_gpu_ini():
    """Permanently locks Lumerical settings into GPU mode."""
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return
    ini_path = Path(appdata) / "Lumerical" / "FDTD Solutions.ini"
    if not ini_path.is_file():
        return
    try:
        txt = ini_path.read_text(encoding="utf-8", errors="replace")
        changed = False
        if "autoCheckGPUSimulation=true" in txt:
            txt = txt.replace("autoCheckGPUSimulation=true", "autoCheckGPUSimulation=false")
            changed = True
        if r"FDTD\GPU-mode=false" in txt:
            txt = txt.replace(r"FDTD\GPU-mode=false", r"FDTD\GPU-mode=true")
            changed = True
        if changed:
            ini_path.write_text(txt, encoding="utf-8")
    except Exception:
        pass


def _eval_single_source_worker(
    gds_path_str: str,
    cell_name: str,
    src_idx: int,
    scratch_dir_str: str,
    template_text: str,
    max_attempts: int = 3,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates a single input source in an isolated process with automatic IPC retry."""
    scratch_dir = Path(scratch_dir_str)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    fsp_path = scratch_dir / f"eval_source{src_idx}.fsp"
    last_error = None

    ensure_gpu_ini()

    for attempt in range(1, max_attempts + 1):
        f = None
        try:
            time.sleep(0.3)
            ensure_gpu_ini()
            f = lumapi.FDTD(hide=True)

            f.putv("full_gds_filename", gds_path_str)
            f.putv("full_gds_cellname", cell_name)
            f.putv("source_index", src_idx)
            f.eval(template_text)
            f.save(fsp_path.as_posix())
            f.run()

            res_te0 = f.getresult("fom1_mode_exp", "expansion for Output")
            res_te1 = f.getresult("fom2_mode_exp", "expansion for Output")

            t0_arr = np.real(np.asarray(res_te0["T_forward"]).squeeze())
            t1_arr = np.real(np.asarray(res_te1["T_forward"]).squeeze())
            f_arr = np.asarray(res_te0["f"]).squeeze()

            assert t0_arr.shape == (11,), f"Bad T_forward shape: {t0_arr.shape}"
            assert t1_arr.shape == (11,), f"Bad T_forward shape: {t1_arr.shape}"
            assert f_arr.shape == (11,), f"Bad f shape: {f_arr.shape}"

            f.close()
            return src_idx, t0_arr, t1_arr, f_arr

        except Exception as e:
            last_error = e
            log(f"Warning: Source {src_idx} attempt {attempt}/{max_attempts} failed ({e}). Retrying in 2s...")
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
            time.sleep(2.0)

    raise RuntimeError(f"Source {src_idx} failed after {max_attempts} attempts: {last_error}")


class OpticalEvaluator:
    """Evaluates optical transmission of full devices via Lumerical FDTD with concurrent dual-source execution."""

    def __init__(self, root: Path, template_path: Path):
        self.root = root
        self.template_text = template_path.read_text(encoding="utf-8")
        self.cache: dict[bytes, tuple[float, np.ndarray, np.ndarray, np.ndarray]] = {}
        self.forward_solves = 0

    def evaluate_state(
        self,
        gds_path: Path,
        cell_name: str,
        state_key: bytes,
        scratch_dir: Path,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """Runs 2 sources concurrently for given GDS and returns (J, T_full, freq, wl_nm, elapsed_s)."""
        if state_key in self.cache:
            J, T_full, freq, wl = self.cache[state_key]
            return J, T_full, freq, wl, 0.0

        t0 = time.time()
        scratch_dir.mkdir(parents=True, exist_ok=True)
        T_full = np.zeros((2, 2, 11), dtype=float)

        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            fut1 = executor.submit(
                _eval_single_source_worker,
                gds_path.as_posix(),
                cell_name,
                1,
                str(scratch_dir),
                self.template_text,
            )
            fut2 = executor.submit(
                _eval_single_source_worker,
                gds_path.as_posix(),
                cell_name,
                2,
                str(scratch_dir),
                self.template_text,
            )
            res1 = fut1.result()
            res2 = fut2.result()

        results = {res1[0]: res1, res2[0]: res2}
        src1_res = results[1]
        src2_res = results[2]

        freq_ref = src1_res[3]
        assert np.allclose(freq_ref, src2_res[3], rtol=1e-12), "Frequency mismatch between sources"

        T_full[0, 0, :] = src1_res[1]
        T_full[0, 1, :] = src1_res[2]
        T_full[1, 0, :] = src2_res[1]
        T_full[1, 1, :] = src2_res[2]
        self.forward_solves += 2

        elapsed = time.time() - t0
        wl_nm = (299792458.0 / freq_ref) * 1e9

        t_target_1 = T_full[0, 0, :].mean()
        t_target_2 = T_full[1, 1, :].mean()
        J = float(0.5 * (t_target_1 + t_target_2))

        self.cache[state_key] = (J, T_full, freq_ref, wl_nm)
        return J, T_full, freq_ref, wl_nm, elapsed


class Stage2JointRunner:
    """Manages Stage 2 Two-Cell Joint Action DBS optimization."""

    def __init__(
        self,
        root: Path,
        scheme: str = "B",
        initial_state_npz: Path | None = None,
        output_dir: Path | None = None,
        max_sweeps: int = 3,
        tolerance: float = 1e-4,
    ):
        self.root = root
        self.scheme = scheme.upper()
        self.max_sweeps = max_sweeps
        self.tolerance = tolerance

        self.output_dir = output_dir or root / "results/B_curve_stage2_joint"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = OpticalEvaluator(root, root / "tools/common_device_template.lsf")
        self.pdk_index_path = root / "config/pdk-index.local.json"

        # Load inputs & mask
        self.folder, self.mask, default_state, _ = geometry_io.inputs(root, self.scheme)

        # Initial state from Stage 1 or default
        if initial_state_npz and initial_state_npz.is_file():
            log(f"Loading initial state from Stage 1: {initial_state_npz}")
            with np.load(initial_state_npz, allow_pickle=False) as arr:
                self.initial_state = arr["silicon_state"].copy()
        else:
            self.initial_state = default_state.copy()

        # Build candidate adjacent pairs (4-connected horizontal & vertical)
        rows, cols = np.where(self.mask)
        cand_set = set(zip(rows, cols))

        pairs_h = [((r, c), (r, c + 1)) for r, c in sorted(cand_set) if (r, c + 1) in cand_set]
        pairs_v = [((r, c), (r + 1, c)) for r, c in sorted(cand_set) if (r + 1, c) in cand_set]
        self.pairs = sorted(pairs_h + pairs_v)
        self.num_pairs = len(self.pairs)

        log(f"Stage 2 Joint DBS Initialized: Scheme {self.scheme}")
        log(f"Total candidate cells: {int(self.mask.sum())}")
        log(f"Adjacent candidate pairs (4-connected): {self.num_pairs} (H: {len(pairs_h)}, V: {len(pairs_v)})")
        log(f"Max sweeps: {self.max_sweeps}, acceptance tolerance: {self.tolerance}")

        # Directories
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.scratch_dir = self.output_dir / "scratch"
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

        self.history_csv = self.output_dir / "history.csv"
        self.history_file = None
        self.csv_writer = None

    def init_history(self, resume: bool):
        fieldnames = [
            "sweep", "step", "pair_idx",
            "r1", "c1", "r2", "c2", "action_type",
            "s1_before", "s2_before", "s1_trial", "s2_trial",
            "accepted", "J_before", "J_trial", "J_current",
            "drc_status", "step_time_s", "total_forward_solves", "wall_time_s"
        ]
        if not resume or not self.history_csv.is_file():
            self.history_file = self.history_csv.open("w", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(self.history_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            self.history_file.flush()
        else:
            self.history_file = self.history_csv.open("a", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(self.history_file, fieldnames=fieldnames)

    def check_drc(self, gds_path: Path) -> tuple[bool, str]:
        """Runs fast 120nm supplemental check, then full 248-category PDK DRC."""
        supp = geometry_io.supplemental(gds_path, 120)
        if supp["width"] > 0 or supp["space"] > 0:
            return False, f"supp_fail(w={supp['width']},s={supp['space']})"

        drc_out = self.scratch_dir / "trial_drc"
        if drc_out.exists():
            shutil.rmtree(drc_out, ignore_errors=True)
        try:
            status = run_drc.main([
                "--gds", str(gds_path),
                "--profile", "common",
                "--index", str(self.pdk_index_path),
                "--output-dir", str(drc_out)
            ])
            if status != 0:
                report_file = drc_out / "result.json"
                v_count = -1
                if report_file.is_file():
                    with report_file.open(encoding="utf-8") as f:
                        v_count = json.load(f).get("violation_count", -1)
                return False, f"pdk_fail(v={v_count})"
        except Exception as e:
            return False, f"drc_error({e})"

        return True, "passed"

    def run(self):
        start_wall_time = time.time()
        latest_ckpt = self.ckpt_dir / "latest_checkpoint.json"
        resume = latest_ckpt.is_file()

        current_state = self.initial_state.copy()
        current_J = -1.0
        current_spectra = None
        start_sweep = 0
        start_step = 0
        total_flips = 0

        self.init_history(resume)

        # Baseline evaluation
        baseline_gds = self.output_dir / "baseline.gds"
        cell_name = f"{self.scheme}_FULL_DEVICE"
        init_spectra_file = self.output_dir / "initial_spectra.npz"

        base_npz = self.scratch_dir / "baseline_state.npz"
        np.savez(base_npz, silicon_state=current_state, candidate_mask=self.mask)
        geometry_io.export_geometry(self.root, self.scheme, base_npz, baseline_gds)

        if resume and init_spectra_file.is_file():
            with np.load(init_spectra_file, allow_pickle=False) as arr:
                init_J = float(arr["J"])
                init_T = arr["T"]
                freq = arr["f"]
                wl = arr["wavelength_nm"]
            current_J = init_J
            current_spectra = init_T
            log(f"Loaded existing baseline initial J = {init_J:.5f} (TE0: {init_T[0,0,:].mean():.4f}, TE1: {init_T[1,1,:].mean():.4f})")
        else:
            log("Evaluating Stage 2 baseline structure (from Stage 1 final)...")
            init_J, init_T, freq, wl, dt = self.evaluator.evaluate_state(
                baseline_gds, cell_name, current_state.tobytes(), self.scratch_dir / "baseline"
            )
            current_J = init_J
            current_spectra = init_T
            log(f"Baseline J = {init_J:.5f} (TE0: {init_T[0,0,:].mean():.4f}, TE1: {init_T[1,1,:].mean():.4f}) in {dt:.1f}s")
            np.savez(
                init_spectra_file,
                J=init_J, T=init_T, f=freq, wavelength_nm=wl,
                state=current_state, mask=self.mask
            )

        # Resume logic
        if resume:
            with latest_ckpt.open(encoding="utf-8") as f:
                ckpt_data = json.load(f)
            start_sweep = ckpt_data["sweep"]
            start_step = ckpt_data["next_step"]
            current_J = ckpt_data["current_J"]
            total_flips = ckpt_data.get("total_flips", 0)
            saved_state_path = self.ckpt_dir / f"state_sweep{start_sweep + 1}_step{start_step}.npz"
            if saved_state_path.is_file():
                with np.load(saved_state_path, allow_pickle=False) as arr:
                    current_state = arr["silicon_state"]
            self.evaluator.forward_solves = ckpt_data.get("forward_solves", 0)
            log(f"Resumed from sweep {start_sweep + 1}, step {start_step + 1}, J = {current_J:.5f}, total_flips = {total_flips}")

        # Optimization Loop
        for sweep in range(start_sweep, self.max_sweeps):
            sweep_start_time = time.time()
            log(f"=== Starting Sweep {sweep + 1}/{self.max_sweeps} (Current J = {current_J:.5f}) ===")
            accepted_in_sweep = 0

            step_range = range(start_step, self.num_pairs) if sweep == start_sweep else range(self.num_pairs)
            start_step = 0  # reset for subsequent sweeps

            for step_idx in step_range:
                step_t0 = time.time()
                (r1, c1), (r2, c2) = self.pairs[step_idx]

                s1_curr = bool(current_state[r1, c1])
                s2_curr = bool(current_state[r2, c2])

                # Determine joint action
                if s1_curr == s2_curr:
                    action_type = "invert"
                    s1_trial = not s1_curr
                    s2_trial = not s2_curr
                else:
                    action_type = "swap"
                    s1_trial = not s1_curr
                    s2_trial = not s2_curr

                trial_state = current_state.copy()
                trial_state[r1, c1] = s1_trial
                trial_state[r2, c2] = s2_trial

                trial_gds = self.scratch_dir / f"trial_s{sweep}_p{step_idx}.gds"
                trial_npz = self.scratch_dir / "trial_state.npz"
                np.savez(trial_npz, silicon_state=trial_state, candidate_mask=self.mask)
                geometry_io.export_geometry(self.root, self.scheme, trial_npz, trial_gds)

                # Strict DRC gate
                drc_ok, drc_status = self.check_drc(trial_gds)

                accepted = False
                trial_J = -1.0

                if not drc_ok:
                    action_str = f"DRC REJECT ({drc_status})"
                else:
                    t_eval_dir = self.scratch_dir / f"eval_s{sweep}_p{step_idx}"
                    trial_J, trial_T, _, _, eval_dt = self.evaluator.evaluate_state(
                        trial_gds, cell_name, trial_state.tobytes(), t_eval_dir
                    )

                    if trial_J > current_J + self.tolerance:
                        accepted = True
                        accepted_in_sweep += 1
                        total_flips += 1
                        current_state = trial_state
                        current_J = trial_J
                        current_spectra = trial_T
                        action_str = f"ACCEPT (J: {current_J:.5f} -> {trial_J:.5f}, +{trial_J - current_J:.5f})"
                    else:
                        action_str = f"REJECT (J: {trial_J:.5f} <= {current_J + self.tolerance:.5f})"

                step_time = time.time() - step_t0
                total_wall = time.time() - start_wall_time

                log(
                    f"Sweep {sweep+1} [{step_idx+1:3d}/{self.num_pairs}] "
                    f"Pair ({r1},{c1})&({r2},{c2}) [{action_type}]: {int(s1_curr)}{int(s2_curr)}->{int(s1_trial)}{int(s2_trial)} "
                    f"=> {action_str} [{step_time:.1f}s, Solves: {self.evaluator.forward_solves}]"
                )

                self.csv_writer.writerow({
                    "sweep": sweep + 1,
                    "step": step_idx + 1,
                    "pair_idx": step_idx,
                    "r1": int(r1),
                    "c1": int(c1),
                    "r2": int(r2),
                    "c2": int(c2),
                    "action_type": action_type,
                    "s1_before": int(s1_curr),
                    "s2_before": int(s2_curr),
                    "s1_trial": int(s1_trial),
                    "s2_trial": int(s2_trial),
                    "accepted": int(accepted),
                    "J_before": round(current_J, 6),
                    "J_trial": round(trial_J, 6) if trial_J >= 0 else "",
                    "J_current": round(current_J, 6),
                    "drc_status": drc_status,
                    "step_time_s": round(step_time, 2),
                    "total_forward_solves": self.evaluator.forward_solves,
                    "wall_time_s": round(total_wall, 1),
                })
                self.history_file.flush()

                # Checkpoint
                next_step = step_idx + 1
                curr_sweep_ckpt = sweep
                if next_step >= self.num_pairs:
                    curr_sweep_ckpt = sweep + 1
                    next_step = 0

                state_ckpt_file = self.ckpt_dir / f"state_sweep{sweep+1}_step{step_idx+1}.npz"
                np.savez(state_ckpt_file, silicon_state=current_state, candidate_mask=self.mask)

                ckpt_meta = {
                    "sweep": curr_sweep_ckpt,
                    "next_step": next_step,
                    "current_J": current_J,
                    "total_flips": total_flips,
                    "forward_solves": self.evaluator.forward_solves,
                    "wall_time_s": total_wall,
                }
                with latest_ckpt.open("w", encoding="utf-8") as f:
                    json.dump(ckpt_meta, f, indent=2)

            log(f"Sweep {sweep + 1} finished in {time.time() - sweep_start_time:.1f}s. Accepted flips: {accepted_in_sweep}")
            if accepted_in_sweep == 0:
                log(f"Early stop condition met: 0 accepted flips in Sweep {sweep + 1}.")
                break

        # Final export
        log("=== Stage 2 Joint Optimization Finished. Exporting final structure... ===")
        final_gds = self.output_dir / "final.gds"
        final_npz = self.output_dir / "final_state.npz"
        np.savez(final_npz, silicon_state=current_state, candidate_mask=self.mask)
        geometry_io.export_geometry(self.root, self.scheme, final_npz, final_gds)
        geometry_io.check_geometry(self.root, self.scheme, final_npz, final_gds)

        np.savez(
            self.output_dir / "final_spectra.npz",
            J=current_J, T=current_spectra, f=freq, wavelength_nm=wl,
            state=current_state, mask=self.mask
        )

        final_drc_dir = self.output_dir / "final_DRC"
        if final_drc_dir.exists():
            shutil.rmtree(final_drc_dir, ignore_errors=True)
        log("Running final 248-category PDK DRC...")
        try:
            run_drc.main([
                "--gds", str(final_gds),
                "--profile", "common",
                "--index", str(self.pdk_index_path),
                "--output-dir", str(final_drc_dir)
            ])
        except Exception as e:
            log(f"Final DRC notice: {e}")

        final_supp = geometry_io.supplemental(final_gds, 120)
        with (self.output_dir / "final_120nm_checks.json").open("w", encoding="utf-8") as f:
            json.dump(final_supp, f, indent=2)

        total_time = time.time() - start_wall_time
        log(f"STAGE 2 ALL DONE! Final J: {current_J:.5f} (Initial: {init_J:.5f}, Gain: +{current_J - init_J:.5f})")
        log(f"Total Joint Flips: {total_flips}, Total Forward Solves: {self.evaluator.forward_solves}, Wall Time: {total_time:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 2 Two-Cell Joint Action DBS optimization.")
    parser.add_argument("--scheme", choices=["B"], default="B", help="Scheme B")
    parser.add_argument("--initial-state-npz", type=str, default="", help="Path to initial state npz from Stage 1")
    parser.add_argument("--output-dir", type=str, default="", help="Custom output directory")
    parser.add_argument("--max-sweeps", type=int, default=3, help="Maximum number of sweeps")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Acceptance threshold dJ")
    args = parser.parse_args()

    root = PACKAGE_ROOT
    init_npz = Path(args.initial_state_npz) if args.initial_state_npz else None
    out_dir = Path(args.output_dir) if args.output_dir else root / "results/B_curve_stage2_joint"

    runner = Stage2JointRunner(
        root=root,
        scheme=args.scheme,
        initial_state_npz=init_npz,
        output_dir=out_dir,
        max_sweeps=args.max_sweeps,
        tolerance=args.tolerance,
    )
    runner.run()
