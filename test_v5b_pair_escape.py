from itertools import combinations
from pathlib import Path
import sys
import json
import numpy as np

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"F:\codex_project\inverse_design_sun")
HANDOFF = REPO / "DBS_handoff_2026-10-11"
if not HANDOFF.exists():
    HANDOFF = REPO / "code" / "dbs" / "handoff_2026-10-11"
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


def main() -> None:
    out_dir = REPO / "runs" / "v5b_pair_escape"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch = out_dir / "scratch"
    scratch.mkdir(exist_ok=True)

    print("[PairEscape] creating DBSRunner...", flush=True)
    runner = DBSRunner(
        root=HANDOFF,
        scheme="B",
        mode="strict",
        output_dir=out_dir / "_bootstrap",
        max_sweeps=1,
        tolerance=1e-4,
    )
    evaluator = runner.evaluator

    v5b = np.load(REPO / "runs" / "surrogate_2mode_ablation_v5b_hourglass" / "final_state.npz")
    base_state = v5b["silicon_state"].astype(bool)
    base_j = float(v5b["J"])
    mask = v5b["candidate_mask"].astype(bool) if "candidate_mask" in v5b.files else runner.mask.astype(bool)
    print(f"[PairEscape] base_J={base_j:.9f} true={int(base_state.sum())} mask={int(mask.sum())}", flush=True)

    trial_state_path = scratch / "pair_state.npz"
    trial_gds = scratch / "pair_trial.gds"

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

    top4 = [(13, 25), (20, 13), (23, 21), (23, 25)]
    print("[PairEscape] top4 =", top4, flush=True)
    results = []
    for (r1, c1), (r2, c2) in combinations(top4, 2):
        trial = base_state.copy()
        trial[r1, c1] = not trial[r1, c1]
        trial[r2, c2] = not trial[r2, c2]
        label = f"({r1},{c1})+({r2},{c2})"
        if not check_drc(trial):
            print(f"{label}: SKIP Tier-1 DRC", flush=True)
            results.append({"pair": label, "status": "tier1_skip"})
            continue
        if not verify_drc(trial):
            print(f"{label}: SKIP Tier-2 DRC", flush=True)
            results.append({"pair": label, "status": "tier2_skip"})
            continue
        j_val = eval_fom(trial)
        dj = j_val - base_j
        print(f"{label}: J={j_val:.9f} dJ={dj:+.9f}", flush=True)
        results.append({"pair": label, "status": "evaluated", "J": j_val, "dJ": dj})
        np.savez(out_dir / f"pair_{r1}_{c1}_{r2}_{c2}.npz", silicon_state=trial, candidate_mask=mask, J=j_val)

    (out_dir / "pair_escape_results.json").write_text(
        json.dumps({"base_J": base_j, "results": results}, indent=2), encoding="utf-8"
    )
    print("[PairEscape] saved", out_dir / "pair_escape_results.json", flush=True)


if __name__ == "__main__":
    main()
