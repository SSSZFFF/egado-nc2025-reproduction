"""Production Online DRC-Gated DBS Fine-Tuning Engine for 4-Mode Multiplexer.

Features:
1. Frozen Candidate Mask: Operates strictly on the 161 candidate cells extracted in Scheme B.
2. Preserves Smooth Vector Geometry: 93.6% of the device area is locked in fixed_background.gds.
3. Online 120nm DRC Gate: Instant C++ filter (<0.05s) rejects any trial state violating min width/space (120nm).
4. Robust ProcessPool Dual-Source Execution: With auto-retry, fresh FSP initialization, and clean memory handling.
5. Seamless Resume: Determines next step from history.csv and loads latest checkpoint state.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import numpy as np
import klayout.db as db

# Setup lumapi path
for p in [r"E:\Program Files\ANSYS Inc\v251\Lumerical\api\python",
          r"D:\Program Files\ANSYS Inc\v251\Lumerical\api\python",
          r"C:\Program Files\ANSYS Inc\v251\Lumerical\api\python"]:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
import lumapi

REPO_DIR = Path(__file__).resolve().parent
INPUT_DIR = REPO_DIR / "scheme_b_dbs_inputs"

ROWS = COLS = 50
CELL_NM = 120
ORIGIN_NM = -3000
DBU_UM = 0.001
UNITS_PER_NM = 1.0 / (DBU_UM * 1000.0)

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

ROI_BOX = db.Box(int(round(-3000 * UNITS_PER_NM)), int(round(-3000 * UNITS_PER_NM)),
                 int(round(3000 * UNITS_PER_NM)), int(round(3000 * UNITS_PER_NM)))
ROI_REGION = db.Region(ROI_BOX)

def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def u(nm):
    return int(round(nm * UNITS_PER_NM))

def cell_box(row, col):
    x0 = ORIGIN_NM + col * CELL_NM
    y0 = ORIGIN_NM + row * CELL_NM
    return db.Box(u(x0), u(y0), u(x0 + CELL_NM), u(y0 + CELL_NM))

def build_lsf_template(gds_placeholder: str = "__GDS_PATH__"):
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

    import_cmd = f"""
setmaterial("n_wg", "Mesh Order", 1);
gdsimport("{gds_placeholder}", "TOP", "13:1", "n_wg", -{WG_THICKNESS/2.0}, {WG_THICKNESS/2.0});
setnamed("FDTD", "simulation time", 5000e-15);
setnamed("FDTD", "auto shutoff min", 1e-5);
setglobalmonitor("frequency points", 11);
"""
    return script + import_cmd

def _eval_single_source_worker(gds_path_str: str, source_idx: int, scratch_dir_str: str, template_code: str, max_retries: int = 3):
    """Evaluates a single optical source channel in an isolated worker process with auto-retry."""
    scratch_dir = Path(scratch_dir_str)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    fsp_path = scratch_dir / f"eval_source{source_idx}.fsp"

    last_error = None
    for attempt in range(1, max_retries + 1):
        # Always remove old fsp files to prevent corrupted state
        if fsp_path.is_file():
            try:
                fsp_path.unlink()
            except Exception:
                pass

        fdtd = None
        try:
            fdtd = lumapi.FDTD(hide=True)
            run_script = template_code.replace("__GDS_PATH__", Path(gds_path_str).as_posix())
            fdtd.eval(run_script)

            for i in range(1, MODE_NUM + 1):
                fdtd.setnamed(f"Source_TE_mode{i}_1550", "enabled", (i == source_idx))

            fdtd.save(str(fsp_path))
            fdtd.run()

            res = fdtd.getresult(f"fom{source_idx}_mode_exp", "expansion for Output")
            t_fwd = np.real(res["T_forward"]).flatten()
            return source_idx, float(np.mean(t_fwd))

        except Exception as e:
            last_error = e
            time.sleep(2.0)
        finally:
            if fdtd is not None:
                try:
                    fdtd.close()
                except Exception:
                    pass

    raise RuntimeError(f"Source {source_idx} failed after {max_retries} attempts: {last_error}")

class SchemeBDBSOptimizer:
    def __init__(self, out_dir: Path, max_sweeps: int = 3, tol: float = 1e-4):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_sweeps = max_sweeps
        self.tol = tol

        self.ckpt_dir = self.out_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.scratch_dir = self.out_dir / "scratch"
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

        # Load inputs
        archive = np.load(INPUT_DIR / "initial_state.npz")
        self.mask = archive["candidate_mask"]
        self.initial_state = archive["silicon_state"]
        self.order_rc = archive["order_rc"]
        self.num_candidates = int(self.mask.sum())

        # Load fixed background geometry
        l_bg = db.Layout()
        l_bg.read(str(INPUT_DIR / "fixed_background.gds"))
        top_bg = l_bg.top_cell()
        self.fixed_bg = db.Region(top_bg.begin_shapes_rec(l_bg.layer(13, 1)))
        self.fixed_bg.merge()

        # Build LSF template
        self.lsf_template = build_lsf_template("__GDS_PATH__")
        self.history_csv = self.out_dir / "history.csv"
        self.history_file = None
        self.csv_writer = None

        log(f"Initialized SchemeBDBSOptimizer for 4-Mode Device")
        log(f"Candidate cells: {self.num_candidates}, Max sweeps: {self.max_sweeps}, Tol: {self.tol}")

    def check_drc(self, candidate_si_boxes: db.Region) -> tuple[bool, int, int]:
        """Fast KLayout 120nm width and space DRC check."""
        device = self.fixed_bg | candidate_si_boxes
        device.merge()
        design_core = device & ROI_REGION
        w = design_core.width_check(u(120), ignore_angle=85).count()
        s = design_core.space_check(u(120), ignore_angle=85).count()
        return (w == 0 and s == 0), w, s

    def export_gds(self, candidate_si_boxes: db.Region, gds_path: Path):
        device = self.fixed_bg | candidate_si_boxes
        device.merge()
        layout = db.Layout()
        layout.dbu = DBU_UM
        top = layout.create_cell("TOP")
        top.shapes(layout.layer(13, 1)).insert(device)
        layout.write(str(gds_path))

    def evaluate_all_channels(self, gds_path: Path) -> tuple[float, list[float], float]:
        """Evaluates 4 channels concurrently using ProcessPoolExecutor."""
        t0 = time.time()
        results = [0.0] * MODE_NUM

        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            futs = [
                executor.submit(
                    _eval_single_source_worker,
                    gds_path.as_posix(),
                    s_idx,
                    str(self.scratch_dir / f"eval_src{s_idx}"),
                    self.lsf_template
                )
                for s_idx in range(1, MODE_NUM + 1)
            ]
            for fut in concurrent.futures.as_completed(futs):
                s_idx, t_val = fut.result()
                results[s_idx - 1] = t_val

        J = float(np.mean(results))
        elapsed = time.time() - t0
        return J, results, elapsed

    def init_history(self, resume: bool):
        fieldnames = [
            "sweep", "step", "row", "col", "candidate_idx",
            "flipped_to", "accepted", "J_before", "J_trial", "J_current",
            "drc_status", "step_time_s", "wall_time_s"
        ]
        if not resume or not self.history_csv.is_file():
            self.history_file = self.history_csv.open("w", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(self.history_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            self.history_file.flush()
        else:
            self.history_file = self.history_csv.open("a", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(self.history_file, fieldnames=fieldnames)

    def run(self):
        start_wall_time = time.time()
        latest_ckpt = self.ckpt_dir / "latest_checkpoint.json"
        resume = latest_ckpt.is_file() and self.history_csv.is_file()

        current_state = self.initial_state.copy()
        current_J = -1.0
        start_sweep = 0
        start_step = 0
        total_flips = 0

        if resume:
            with latest_ckpt.open(encoding="utf-8") as f:
                ckpt_data = json.load(f)
            current_J = ckpt_data["current_J"]
            total_flips = ckpt_data.get("total_flips", 0)

            # Determine next step strictly from history.csv
            with open(self.history_csv, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                last_row = rows[-1]
                start_sweep = int(last_row["sweep"]) - 1
                start_step = int(last_row["step"])  # 1-based step becomes 0-based index for next
                if start_step >= self.num_candidates:
                    start_sweep += 1
                    start_step = 0

            # Find the latest state checkpoint file by numerical (sweep, step)
            def _parse_ckpt(p: Path):
                m = re.search(r"state_sweep(\d+)_step(\d+)", p.name)
                return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

            ckpt_files = sorted(self.ckpt_dir.glob("state_sweep*.npz"), key=_parse_ckpt)
            if ckpt_files:
                latest_state_file = ckpt_files[-1]
                with np.load(latest_state_file, allow_pickle=False) as arr:
                    current_state = arr["silicon_state"]
                log(f"Resumed from {latest_state_file.name}: Sweep {start_sweep+1}, Step {start_step+1}/{self.num_candidates}, J = {current_J*100:.2f}%, Total Flips = {total_flips}")
            else:
                log(f"Resumed from history.csv: Sweep {start_sweep+1}, Step {start_step+1}/{self.num_candidates}, J = {current_J*100:.2f}%")
        else:
            # Baseline evaluation
            baseline_gds = self.out_dir / "baseline.gds"
            cand_si = db.Region()
            for r, c in np.argwhere(self.mask & current_state):
                cand_si.insert(cell_box(r, c))
            cand_si.merge()
            self.export_gds(cand_si, baseline_gds)

            log("Evaluating baseline initial structure across 4 channels...")
            init_J, init_res, init_dt = self.evaluate_all_channels(baseline_gds)
            current_J = init_J
            log(f"Baseline Initial J = {init_J*100:.2f}% (CH1: {init_res[0]*100:.2f}%, "
                f"CH2: {init_res[1]*100:.2f}%, CH3: {init_res[2]*100:.2f}%, "
                f"CH4: {init_res[3]*100:.2f}%) in {init_dt:.1f}s")

        self.init_history(resume)

        # Optimization sweeps
        for sweep in range(start_sweep, self.max_sweeps):
            sweep_t0 = time.time()
            log(f"\n=======================================================")
            log(f"=== Starting Sweep {sweep + 1}/{self.max_sweeps} (Current J = {current_J*100:.2f}%) ===")
            log(f"=======================================================")
            accepted_in_sweep = 0

            step_range = range(start_step, self.num_candidates) if sweep == start_sweep else range(self.num_candidates)
            start_step = 0  # reset for subsequent sweeps

            for step_idx in step_range:
                step_t0 = time.time()
                r, c = self.order_rc[step_idx]
                curr_val = bool(current_state[r, c])
                trial_val = not curr_val

                trial_state = current_state.copy()
                trial_state[r, c] = trial_val

                # Form candidate silicon region for trial state
                trial_cand_si = db.Region()
                for tr, tc in np.argwhere(self.mask & trial_state):
                    trial_cand_si.insert(cell_box(tr, tc))
                trial_cand_si.merge()

                # Fast Online DRC Gate
                drc_pass, w_err, s_err = self.check_drc(trial_cand_si)

                accepted = False
                trial_J = -1.0
                drc_status = "passed"

                if not drc_pass:
                    drc_status = f"drc_reject(w={w_err},s={s_err})"
                    action_str = f"DRC REJECT ({drc_status})"
                else:
                    trial_gds = self.scratch_dir / f"trial_s{sweep}_c{step_idx}.gds"
                    self.export_gds(trial_cand_si, trial_gds)
                    try:
                        trial_J, trial_res, eval_dt = self.evaluate_all_channels(trial_gds)

                        if trial_J > current_J + self.tol:
                            accepted = True
                            accepted_in_sweep += 1
                            total_flips += 1
                            action_str = f"ACCEPT (J: {current_J*100:.2f}% -> {trial_J*100:.2f}%, +{(trial_J-current_J)*100:.3f}%)"
                            current_state = trial_state
                            current_J = trial_J
                        else:
                            action_str = f"REJECT (J: {trial_J*100:.2f}% <= {current_J*100:.2f}%)"
                    except Exception as e:
                        action_str = f"EVAL ERROR ({e}), REJECTED"
                        trial_J = -1.0
                        accepted = False

                step_time = time.time() - step_t0
                wall_time = time.time() - start_wall_time
                log(f"Sweep {sweep+1} [{step_idx+1:3d}/{self.num_candidates}] "
                    f"Cell ({r:2d},{c:2d}) -> {int(trial_val)}: {action_str} [{step_time:.1f}s]")

                self.csv_writer.writerow({
                    "sweep": sweep + 1, "step": step_idx + 1,
                    "row": int(r), "col": int(c), "candidate_idx": step_idx,
                    "flipped_to": int(trial_val), "accepted": int(accepted),
                    "J_before": round(current_J, 6),
                    "J_trial": round(trial_J, 6) if trial_J >= 0 else "",
                    "J_current": round(current_J, 6),
                    "drc_status": drc_status,
                    "step_time_s": round(step_time, 2),
                    "wall_time_s": round(wall_time, 1)
                })
                self.history_file.flush()

                # Save checkpoint on every step
                if accepted:
                    ckpt_file = self.ckpt_dir / f"state_sweep{sweep+1}_step{step_idx+1}.npz"
                    np.savez_compressed(ckpt_file, silicon_state=current_state, candidate_mask=self.mask)

                with latest_ckpt.open("w", encoding="utf-8") as f:
                    json.dump({
                        "sweep": sweep, "step": step_idx + 1,
                        "current_J": current_J, "total_flips": total_flips,
                        "wall_time_s": wall_time
                    }, f, indent=2)

            log(f"Sweep {sweep + 1} finished in {time.time()-sweep_t0:.1f}s. Accepted flips: {accepted_in_sweep}")
            if accepted_in_sweep == 0:
                log(f"Early stop condition met: 0 flips accepted in Sweep {sweep + 1}.")
                break

        log("=== Optimization Finished. Exporting final GDS structure... ===")
        final_cand_si = db.Region()
        for r, c in np.argwhere(self.mask & current_state):
            final_cand_si.insert(cell_box(r, c))
        final_cand_si.merge()

        final_gds = self.out_dir / "final.gds"
        self.export_gds(final_cand_si, final_gds)
        np.savez_compressed(self.out_dir / "final_state.npz", silicon_state=current_state, candidate_mask=self.mask)

        drc_ok, fw, fs = self.check_drc(final_cand_si)
        log(f"Final 120nm DRC Status: {'PASSED (0 violations)' if drc_ok else f'FAILED (w={fw},s={fs})'}")
        log(f"ALL DONE! Final J: {current_J*100:.2f}% (Total Flips: {total_flips})")

        if self.history_file:
            self.history_file.close()

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "results_dbs_phase2")
        parser.add_argument("--max-sweeps", type=int, default=3)
        parser.add_argument("--tolerance", type=float, default=1e-4)
        args = parser.parse_args()

        runner = SchemeBDBSOptimizer(args.output_dir, args.max_sweeps, args.tolerance)
        runner.run()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
