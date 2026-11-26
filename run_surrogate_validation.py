"""
Drop-in Validation Launcher for SSSZFFF/egado-nc2025-reproduction
=================================================================
Phase 1: 2-mode (30x30) validation — plugs directly into:
    code/dbs/handoff_2026-10-11/run_dbs.py  (OpticalEvaluator + geometry_io DRC)
Phase 2: 4-mode (50x50) validation — plugs directly into:
    code/four_mode/run_dbs_4mode.py         (SchemeBDBSOptimizer.check_drc + evaluate_all_channels)

Place both `physics_surrogate_digitizer.py` and this script in your repo root (or `code/`),
then run:
    # 1) 2-mode ablation (same 108-cell mask as B_strict, apples-to-apples vs 424-solve baseline)
    python run_surrogate_validation.py --mode 2mode --repo . --ablation-mask --max-fdtd 150

    # 2) 2-mode full merged Stage 2+3 (no Canny mask, searches all DRC-legal pixels in 30x30)
    python run_surrogate_validation.py --mode 2mode --repo . --max-fdtd 150

    # 3) 4-mode validation (50x50, reuses SchemeBDBSOptimizer KLayout DRC + multi-process FDTD)
    python run_surrogate_validation.py --mode 4mode --repo . --ablation-mask --max-fdtd 160
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from physics_surrogate_digitizer import (
    PasswiseMagnitudeDigitizer,
    PhysicsContext,
    SurrogateGuidedDigitizer,
    downsample_to_blocks,
)
from extract_discrete_adjoint import extract_discrete_adjoint


def _ensure_lumerical_api() -> None:
    """Prefer the newest installed Lumerical API before importing run_dbs modules."""
    candidates = [
        r"D:\Program Files\ANSYS Inc\v261\Lumerical\api\python",
        r"D:\Program Files\ANSYS Inc\v251\Lumerical\api\python",
        r"E:\Program Files\ANSYS Inc\v251\Lumerical\api\python",
        r"C:\Program Files\ANSYS Inc\v261\Lumerical\api\python",
        r"C:\Program Files\ANSYS Inc\v251\Lumerical\api\python",
    ]
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def _find_handoff(repo: Path) -> Path:
    candidates = [
        repo / "code" / "dbs" / "handoff_2026-10-11",
        repo / "DBS_handoff_2026-10-11",
        repo / "DBS_交接包_2026-10-11",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        "DBS handoff directory not found. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def _find_four_mode_dir(repo: Path) -> Path:
    candidates = [
        repo / "code" / "four_mode",
        repo / "EGADO-4mode",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        "4-mode directory not found. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def _load_optional_physics_maps(
    to_npz_path: Path | None,
    rows: int,
    cols: int,
    fallback_state: np.ndarray,
) -> PhysicsContext:
    """
    Loads Stage-1 `eps` (and optional `grad_eps`, `e2_field` once dumped from lumopt)
    from `parameters_<n>.npz` and aggregates them onto the (rows, cols) digital grid.
    Works out-of-the-box even if `grad_eps` has not been dumped yet (fast-start mode).
    """
    if to_npz_path is not None and to_npz_path.exists():
        data = np.load(to_npz_path)
        eps_raw = data["eps"]
        if eps_raw.ndim == 3:
            eps_raw = eps_raw[:, :, eps_raw.shape[2] // 2]
        eps_block = downsample_to_blocks(eps_raw, rows, cols, mode="mean")

        grad_block = None
        if "grad_eps" in data:
            g = data["grad_eps"]
            if g.ndim == 2:
                grad_block = downsample_to_blocks(g, rows, cols, mode="sum")
            elif g.ndim == 3:
                grad_block = np.stack(
                    [downsample_to_blocks(g[b], rows, cols, mode="sum") for b in range(g.shape[0])],
                    axis=0,
                )

        e2_block = None
        if "e2_field" in data:
            e2 = data["e2_field"]
            if e2.ndim == 2:
                e2_block = downsample_to_blocks(e2, rows, cols, mode="mean")
            elif e2.ndim == 3:
                e2_block = np.stack(
                    [downsample_to_blocks(e2[b], rows, cols, mode="mean") for b in range(e2.shape[0])],
                    axis=0,
                )
        return PhysicsContext(eps_analog=eps_block, grad_map=grad_block, e2_map=e2_block)

    # Fallback: use initial binary state smoothed slightly as analog proxy
    return PhysicsContext(eps_analog=fallback_state.astype(np.float32))


def run_2mode(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    handoff_dir = _find_handoff(repo)
    _ensure_lumerical_api()
    sys.path.insert(0, str(handoff_dir))
    sys.path.insert(0, str(handoff_dir / "tools"))

    import geometry_io  # type: ignore
    from run_dbs import DBSRunner, OpticalEvaluator  # type: ignore

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (handoff_dir / "out_surrogate_2mode")
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = out_dir / "scratch"
    scratch_dir.mkdir(exist_ok=True)

    # Reuse DBSRunner to load exact B_strict initial_state.npz & template paths
    runner = DBSRunner(
        root=handoff_dir,
        scheme="B",
        mode="strict",
        output_dir=out_dir / "_bootstrap",
        max_sweeps=1,
        tolerance=args.tol,
    )
    init_state = runner.initial_state.astype(bool)
    rows, cols = init_state.shape
    if not args.ablation_mask:
        # Full-grid mode: initialize from the actual initial.gds raster over
        # all 30x30 cells. runner.initial_state only contains the 108 Canny
        # candidates and would make the other 792 cells incorrectly oxide.
        _, _, original_init = geometry_io.read_gds(handoff_dir / "inputs" / "B_curve_DRC120" / "initial.gds")
        original_regions = original_init
        full_init = np.zeros((rows, cols), dtype=bool)
        for r, c in np.argwhere(np.ones((rows, cols), dtype=bool)):
            box = geometry_io.cell_box(int(r), int(c))
            cell_area = float(box.area())
            core_frac = float((original_init[geometry_io.CORE] & geometry_io.db.Region(box)).area()) / cell_area
            full_init[r, c] = core_frac > 0.5
        init_state = full_init
    mask = runner.mask.astype(bool) if args.ablation_mask else None

    to_npz_default = repo / "code" / "two_mode" / "analysis-2mode" / "2mode_final.npz"
    if not to_npz_default.exists():
        to_npz_default = handoff_dir / "inputs" / "common" / "2mode_final.npz"
    to_npz = Path(args.to_npz) if args.to_npz else to_npz_default
    ctx = _load_optional_physics_maps(to_npz, rows, cols, init_state)

    evaluator: OpticalEvaluator = runner.evaluator
    trial_gds = scratch_dir / "surrogate_trial_2mode.gds"

    # Full-grid export: use the original vector CORE as the base and apply
    # cell add/subtract operations. This preserves the sub-cell curves that a
    # pure 900-cell boolean raster cannot represent (and avoids false DRC).
    base_full_core = None
    if not args.ablation_mask:
        base_full_core = original_regions[geometry_io.CORE].merged()

        def build_full_core(state: np.ndarray):
            s = np.asarray(state, dtype=bool)
            diff = s ^ full_init
            if not np.any(diff):
                return base_full_core.dup()
            core = base_full_core.dup()
            add_mask = diff & s
            sub_mask = diff & (~s)
            if np.any(add_mask):
                core = core | geometry_io.selected_region(add_mask)
            if np.any(sub_mask):
                core = core - geometry_io.selected_region(sub_mask)
            core.merge()
            return core

        def export_gds(state: np.ndarray, path: Path) -> None:
            core = build_full_core(state)
            geometry_io.write_gds(path, "B_SURROGATE_FULL", core, original_regions)

        def check_drc_full_fast(state: np.ndarray) -> bool:
            core = build_full_core(state)
            w = core.width_check(1200, ignore_angle=85).count()
            sp = core.space_check(1200, ignore_angle=85).count()
            return bool(w == 0 and sp == 0)

    def export_state_gds(trial_state: np.ndarray, gds_path: Path) -> None:
        """Export a 30x30 state to the exact top-cell layout DBSRunner evaluates."""
        trial_state = np.asarray(trial_state, dtype=bool)
        if trial_state.shape != (rows, cols):
            raise ValueError(f"Expected state shape {(rows, cols)}, got {trial_state.shape}")
        if args.ablation_mask:
            tmp = scratch_dir / "surrogate_state.npz"
            np.savez(tmp, silicon_state=trial_state, candidate_mask=runner.mask)
            geometry_io.export_geometry(handoff_dir, runner.scheme, tmp, Path(gds_path))
        else:
            export_gds(trial_state, Path(gds_path))

    def check_drc_2mode(trial_state: np.ndarray) -> bool:
        # Fast screening for every candidate in the grid. Full 248-category
        # deck checking is deferred to verify_drc_2mode() for Top-K only.
        if not args.ablation_mask:
            return check_drc_full_fast(trial_state)
        export_state_gds(trial_state, trial_gds)
        res = geometry_io.supplemental(trial_gds, 120)
        return bool(res["width"] == 0 and res["space"] == 0)

    def verify_drc_2mode(trial_state: np.ndarray) -> bool:
        export_state_gds(trial_state, trial_gds)
        if runner.pdk_index_path.is_file():
            ok, _status = runner.check_drc(trial_gds)
            return bool(ok)
        res = geometry_io.supplemental(trial_gds, 120)
        return bool(res["width"] == 0 and res["space"] == 0)

    def eval_2mode(trial_state: np.ndarray) -> float:
        export_state_gds(trial_state, trial_gds)
        key = np.packbits(trial_state.astype(np.uint8)).tobytes()
        cell_name = "B_SURROGATE_FULL" if not args.ablation_mask else f"{runner.scheme}_FULL_DEVICE"
        j_val, *_ = evaluator.evaluate_state(trial_gds, cell_name, key, scratch_dir)
        return float(j_val)

    def on_accept_2mode(state: np.ndarray, j_val: float, flip_idx: int) -> None:
        export_state_gds(state, out_dir / f"accepted_flip{flip_idx}_J{j_val:.4f}.gds")

    # Refresh the exact discrete adjoint on the current accepted layout every
    # `--refresh-every-flips` flips, or after a run of no-improvement rounds.
    refresh_count = 0
    template_path = handoff_dir / "tools" / "common_device_template.lsf"

    def refresh_physics_2mode(state: np.ndarray) -> PhysicsContext:
        nonlocal refresh_count
        refresh_count += 1
        refresh_gds = out_dir / f"refresh_input_{refresh_count}.gds"
        refresh_npz = out_dir / f"adjoint_refresh_{refresh_count}.npz"
        export_state_gds(state, refresh_gds)
        print(f"[Surrogate] refreshing discrete adjoint #{refresh_count} on accepted state...", flush=True)
        extract_discrete_adjoint(
            gds_path=refresh_gds,
            cell_name="auto",
            template_path=template_path,
            out_npz=refresh_npz,
            mode_count=2,
            roi_um=3.6,
            out_wg_x_um=2.1,
            out_wg_yspan_um=1.9,
            rows=rows,
            cols=cols,
            freq_points=11,
            scratch_dir=out_dir / f"refresh_scratch_{refresh_count}",
        )
        return _load_optional_physics_maps(refresh_npz, rows, cols, state)

    top_n_list = [int(x.strip()) for x in args.pass_top_n.split(",") if x.strip()]
    pass_active = None
    if getattr(args, "pass_active_boundary", None):
        pass_active = [bool(int(x.strip())) for x in args.pass_active_boundary.split(",") if x.strip()]
    if args.strategy == "passwise":
        opt = PasswiseMagnitudeDigitizer(
            initial_state=init_state,
            physics_ctx=ctx,
            check_drc_fn=check_drc_2mode,
            verify_drc_fn=verify_drc_2mode,
            eval_fn=eval_2mode,
            out_dir=out_dir,
            candidate_mask=mask,
            on_accept_fn=on_accept_2mode,
            refresh_physics_fn=refresh_physics_2mode if args.enable_refresh else None,
            pass_top_n=top_n_list,
            pair_escape_top_m=4,
            active_boundary=pass_active if pass_active is not None else args.active_boundary,
            tol=args.tol,
            max_fdtd_evals=args.max_fdtd,
        )
    else:
        opt = SurrogateGuidedDigitizer(
            initial_state=init_state,
            physics_ctx=ctx,
            check_drc_fn=check_drc_2mode,
            eval_fn=eval_2mode,
            out_dir=out_dir,
            candidate_mask=mask,
            on_accept_fn=on_accept_2mode,
            verify_drc_fn=verify_drc_2mode,
            pure_adjoint=args.pure_adjoint,
            compact_features=None,
            refresh_physics_fn=refresh_physics_2mode if args.enable_refresh else None,
            refresh_every_flips=args.refresh_every_flips,
            refresh_no_improve_rounds=args.refresh_no_improve_rounds,
            top_k=args.top_k,
            mag_top_n=args.mag_top_n,
            acquisition=args.acquisition,
            ei_threshold=args.ei_threshold,
            ucb_kappa=args.kappa,
            tol=args.tol,
            max_rounds=args.max_rounds,
            max_fdtd_evals=args.max_fdtd,
        )
    summary = opt.run()
    print("[2-mode Surrogate Finished]", summary)


def run_4mode(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    four_dir = _find_four_mode_dir(repo)
    _ensure_lumerical_api()
    sys.path.insert(0, str(four_dir))

    import klayout.db as db  # type: ignore
    from run_dbs_4mode import CELL_NM, ORIGIN_NM, SchemeBDBSOptimizer  # type: ignore

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (four_dir / "out_surrogate_4mode")
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = out_dir / "scratch"
    scratch_dir.mkdir(exist_ok=True)

    # Reuse SchemeBDBSOptimizer for KLayout fixed_bg + check_drc + multi-process evaluate_all_channels
    base_opt = SchemeBDBSOptimizer(out_dir=out_dir, max_sweeps=1, tol=args.tol)
    init_state = base_opt.initial_state.astype(bool)
    mask = base_opt.mask.astype(bool) if args.ablation_mask else None
    rows, cols = init_state.shape

    to_npz = Path(args.to_npz) if args.to_npz else None
    if to_npz is None:
        # Optional convenience: use the newest Stage-1 parameter snapshot if present.
        candidates = sorted(four_dir.glob("parameters*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            to_npz = candidates[0]
    ctx = _load_optional_physics_maps(to_npz, rows, cols, init_state)
    trial_gds = scratch_dir / "surrogate_trial_4mode.gds"

    if not args.ablation_mask:
        # The Scheme-B fixed_background.gds contains fixed non-candidate cells
        # inside the 50x50 design area. For a full-grid search those cells must
        # become variable, so the fixed region for this run is empty (the LSF
        # template supplies the waveguides outside the design region).
        base_opt.fixed_bg = db.Region()

    def _state_to_region(trial_state: np.ndarray) -> db.Region:
        region = db.Region()
        active_mask = base_opt.mask if args.ablation_mask else np.ones_like(trial_state, dtype=bool)
        for r, c in np.argwhere(active_mask & trial_state):
            x0 = ORIGIN_NM + int(c) * CELL_NM
            y0 = ORIGIN_NM + int(r) * CELL_NM
            region.insert(db.Box(x0, y0, x0 + CELL_NM, y0 + CELL_NM))
        return region

    def check_drc_4mode(trial_state: np.ndarray) -> bool:
        ok, _w, _s = base_opt.check_drc(_state_to_region(trial_state))
        return bool(ok)

    def eval_4mode(trial_state: np.ndarray) -> float:
        base_opt.export_gds(_state_to_region(trial_state), trial_gds)
        j_val, _ch_list, _elapsed = base_opt.evaluate_all_channels(trial_gds)
        return float(j_val)

    def on_accept_4mode(state: np.ndarray, j_val: float, flip_idx: int) -> None:
        base_opt.export_gds(_state_to_region(state), out_dir / f"accepted_flip{flip_idx}_J{j_val:.4f}.gds")

    top_n_list = [int(x.strip()) for x in args.pass_top_n.split(",") if x.strip()]
    pass_active = None
    if getattr(args, "pass_active_boundary", None):
        pass_active = [bool(int(x.strip())) for x in args.pass_active_boundary.split(",") if x.strip()]
    if args.strategy == "passwise":
        opt = PasswiseMagnitudeDigitizer(
            initial_state=init_state,
            physics_ctx=ctx,
            check_drc_fn=check_drc_4mode,
            verify_drc_fn=None,
            eval_fn=eval_4mode,
            out_dir=out_dir,
            candidate_mask=mask,
            on_accept_fn=on_accept_4mode,
            refresh_physics_fn=None,
            pass_top_n=top_n_list,
            pair_escape_top_m=4,
            active_boundary=pass_active if pass_active is not None else args.active_boundary,
            tol=args.tol,
            max_fdtd_evals=args.max_fdtd,
        )
    else:
        opt = SurrogateGuidedDigitizer(
            initial_state=init_state,
            physics_ctx=ctx,
            check_drc_fn=check_drc_4mode,
            eval_fn=eval_4mode,
            out_dir=out_dir,
            candidate_mask=mask,
            on_accept_fn=on_accept_4mode,
            pure_adjoint=args.pure_adjoint,
            compact_features=None,
            top_k=args.top_k,
            mag_top_n=args.mag_top_n,
            acquisition=args.acquisition,
            ei_threshold=args.ei_threshold,
            ucb_kappa=args.kappa,
            tol=args.tol,
            max_rounds=args.max_rounds,
            max_fdtd_evals=args.max_fdtd,
        )
    summary = opt.run()
    print("[4-mode Surrogate Finished]", summary)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2-mode & 4-mode Physics-Residual Online Surrogate Validation")
    p.add_argument("--mode", choices=["2mode", "4mode"], required=True)
    p.add_argument("--strategy", choices=["surrogate", "passwise"], default="surrogate")
    p.add_argument("--pass-top-n", type=str, default="36,36,48", help="Comma-separated Top-N per pass")
    p.add_argument("--active-boundary", action="store_true", help="Use dynamic 4-neighbour Si/air boundary mask")
    p.add_argument("--pass-active-boundary", type=str, default=None, help="Comma-separated 0/1 active-boundary flags per pass")
    p.add_argument("--repo", type=str, default=".", help="Path to egado-nc2025-reproduction root")
    p.add_argument("--to-npz", type=str, default=None, help="Optional Stage-1 parameters_<n>.npz path")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory for checkpoints & history")
    p.add_argument("--ablation-mask", action="store_true", help="Restrict search to existing candidate_mask")
    p.add_argument("--top-k", type=int, default=4, help="FDTD evaluations per active-learning round")
    p.add_argument("--mag-top-n", type=int, default=36, help="Keep only top-N candidates by |linear prior| (0 disables)")
    p.add_argument("--acquisition", choices=["ei", "ucb"], default="ei", help="EI or raw UCB acquisition")
    p.add_argument("--pure-adjoint", action="store_true", help="Pure refreshed adjoint ranking (no Ridge)")
    p.add_argument("--enable-refresh", action="store_true", help="Refresh discrete adjoint during the run")
    p.add_argument("--ei-threshold", type=float, default=1e-4, help="EI improvement threshold")
    p.add_argument("--refresh-every-flips", type=int, default=5, help="Refresh discrete adjoint after this many accepted flips")
    p.add_argument("--refresh-no-improve-rounds", type=int, default=4, help="Refresh discrete adjoint after this many no-improvement rounds")
    p.add_argument("--kappa", type=float, default=0.5, help="UCB exploration coefficient")
    p.add_argument("--tol", type=float, default=1e-4, help="Minimum J improvement to accept a flip")
    p.add_argument("--max-rounds", type=int, default=120)
    p.add_argument("--max-fdtd", type=int, default=150, help="State evaluations; 2-mode uses 2 forward solves/state")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "2mode":
        run_2mode(args)
    else:
        run_4mode(args)
