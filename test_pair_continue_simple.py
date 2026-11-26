from pathlib import Path
import sys
import numpy as np

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"F:\codex_project\inverse_design_sun")
HANDOFF = REPO / "DBS_handoff_2026-10-11"
for p in (REPO, HANDOFF, HANDOFF / "tools"):
    sys.path.insert(0, str(p))
for candidate in (
    r"D:\Program Files\ANSYS Inc\v261\Lumerical\api\python",
    r"D:\Program Files\ANSYS Inc\v251\Lumerical\api\python",
):
    if Path(candidate).exists():
        sys.path.insert(0, candidate)
        break

import geometry_io  # type: ignore
from run_dbs import DBSRunner  # type: ignore
from physics_surrogate_digitizer import PhysicsContext, PasswiseMagnitudeDigitizer, downsample_to_blocks  # type: ignore


def main() -> None:
    out_dir = REPO / "runs" / "surrogate_2mode_ablation_v5c_pair_continue"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "scratch"
    scratch.mkdir(exist_ok=True)

    pair = np.load(REPO / "runs" / "v5b_pair_escape" / "pair_20_13_23_21.npz")
    base_state = pair["silicon_state"].astype(bool)
    mask = pair["candidate_mask"].astype(bool) if "candidate_mask" in pair.files else None
    base_j = float(pair["J"])
    print(f"[PairContinue] base_J={base_j:.9f} true={int(base_state.sum())}", flush=True)

    adj = np.load(out_dir / "adjoint_pair.npz")
    ctx = PhysicsContext(
        eps_analog=downsample_to_blocks(adj["eps"], *base_state.shape),
        grad_map=downsample_to_blocks(adj["grad_eps"], *base_state.shape, mode="sum"),
        e2_map=downsample_to_blocks(adj["e2_field"], *base_state.shape),
    )
    print("[PairContinue] adjoint loaded", flush=True)

    runner = DBSRunner(
        root=HANDOFF,
        scheme="B",
        mode="strict",
        output_dir=out_dir / "_bootstrap",
        max_sweeps=1,
        tolerance=1e-4,
    )
    evaluator = runner.evaluator
    trial_state_path = scratch / "trial_state.npz"
    trial_gds = scratch / "trial.gds"

    def export_state(state: np.ndarray, path: Path) -> None:
        np.savez(trial_state_path, silicon_state=state, candidate_mask=mask)
        geometry_io.export_geometry(HANDOFF, "B", trial_state_path, path)

    def check_drc(state: np.ndarray) -> bool:
        export_state(state, trial_gds)
        res = geometry_io.supplemental(trial_gds, 120)
        return bool(res["width"] == 0 and res["space"] == 0)

    def verify_drc(state: np.ndarray) -> bool:
        export_state(state, trial_gds)
        if runner.pdk_index_path.is_file():
            ok, _status = runner.check_drc(trial_gds)
            return bool(ok)
        res = geometry_io.supplemental(trial_gds, 120)
        return bool(res["width"] == 0 and res["space"] == 0)

    def eval_fom(state: np.ndarray) -> float:
        export_state(state, trial_gds)
        key = np.packbits(state.astype(np.uint8)).tobytes()
        j_val, *_ = evaluator.evaluate_state(trial_gds, "B_FULL_DEVICE", key, scratch)
        return float(j_val)

    def on_accept(state: np.ndarray, j_val: float, flip_idx: int) -> None:
        export_state(state, out_dir / f"accepted_flip{flip_idx}_J{j_val:.4f}.gds")

    opt = PasswiseMagnitudeDigitizer(
        initial_state=base_state,
        physics_ctx=ctx,
        check_drc_fn=check_drc,
        eval_fn=eval_fom,
        out_dir=out_dir,
        candidate_mask=mask,
        verify_drc_fn=verify_drc,
        on_accept_fn=on_accept,
        refresh_physics_fn=None,
        pass_top_n=[30],
        pair_escape_top_m=4,
        tol=1e-4,
        max_fdtd_evals=35,
    )
    summary = opt.run()
    print("[PairContinue Finished]", summary, flush=True)


if __name__ == "__main__":
    main()
