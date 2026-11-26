from pathlib import Path
import sys
import json
import shutil
import numpy as np

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"F:\codex_project\inverse_design_sun")
HANDOFF = REPO / "DBS_handoff_2026-10-11"
OUT = REPO / "runs" / "surrogate_2mode_ablation_v6b_boundary_ports"
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
from extract_discrete_adjoint import extract_discrete_adjoint  # noqa: F401


def metrics(path):
    d = np.load(path)
    T = np.asarray(d["T"], dtype=float)
    t0 = float(T[0, 0].mean()); t1 = float(T[1, 1].mean())
    c0 = float(T[0, 1].mean()); c1 = float(T[1, 0].mean())
    return {
        "J": float(d["J"]),
        "IL_db": [-10.0 * np.log10(t0), -10.0 * np.log10(t1)],
        "IL_mean_db": float((-10.0 * np.log10(t0) - 10.0 * np.log10(t1)) / 2.0),
        "CT_mean_db": [float(10.0 * np.log10(c0 / t0)), float(10.0 * np.log10(c1 / t1))],
        "CT_worst_db": [float(10.0 * np.log10((T[0, 1] / T[0, 0]).max())), float(10.0 * np.log10((T[1, 0] / T[1, 1]).max()))],
        "wavelength_nm": np.asarray(d["wavelength_nm"], dtype=float).tolist(),
    }


def main() -> None:
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    final = np.load(out / "final_state.npz")
    state = final["silicon_state"].astype(bool)
    mask = final["candidate_mask"].astype(bool) if "candidate_mask" in final.files else np.ones_like(state, dtype=bool)
    print("[Audit] final state true", int(state.sum()), flush=True)

    _, _, original = geometry_io.read_gds(HANDOFF / "inputs" / "B_curve_DRC120" / "initial.gds")
    base = original[geometry_io.CORE].merged()
    full_init = np.zeros((30, 30), dtype=bool)
    for r, c in np.argwhere(np.ones((30, 30), dtype=bool)):
        box = geometry_io.cell_box(int(r), int(c))
        frac = (original[geometry_io.CORE] & geometry_io.db.Region(box)).area() / box.area()
        full_init[r, c] = frac > 0.5

    def build(s):
        s = np.asarray(s, dtype=bool)
        diff = s ^ full_init
        if not np.any(diff):
            return base.dup()
        core = base.dup()
        add = diff & s
        sub = diff & (~s)
        if np.any(add):
            core = core | geometry_io.selected_region(add)
        if np.any(sub):
            core = core - geometry_io.selected_region(sub)
        core.merge()
        return core

    final_gds = out / "final_audit.gds"
    geometry_io.write_gds(final_gds, "B_SURROGATE_FULL", build(state), original)
    print("[Audit] wrote", final_gds, flush=True)

    runner = DBSRunner(
        root=HANDOFF, scheme="B", mode="strict", output_dir=out / "_audit_bootstrap",
        max_sweeps=1, tolerance=1e-4,
    )
    ev = runner.evaluator
    scratch = out / "audit_scratch"
    scratch.mkdir(exist_ok=True)
    key = np.packbits(state.astype(np.uint8)).tobytes()
    j_val, T, freq, wl, elapsed = ev.evaluate_state(final_gds, "B_SURROGATE_FULL", key, scratch)
    np.savez(out / "final_spectra.npz", J=float(j_val), T=np.asarray(T), f=np.asarray(freq), wavelength_nm=np.asarray(wl), state=state, mask=mask)
    print("[Audit] J", float(j_val), "elapsed", elapsed, flush=True)

    supp = geometry_io.supplemental(final_gds, 120)
    (out / "final_120nm_checks.json").write_text(json.dumps(supp, indent=2), encoding="utf-8")
    drc_ok, drc_status = runner.check_drc(final_gds)
    print("[Audit] supplemental", supp, "DRC", drc_ok, drc_status, flush=True)
    drc_result_path = runner.scratch_dir / "trial_drc" / "result.json"
    shutil.copy2(drc_result_path, out / "final_DRC_result.json")
    (out / "final_DRC_result.json").write_text(
        json.dumps({
            "ok": bool(drc_ok), "status": drc_status,
            "result": json.loads(drc_result_path.read_text(encoding="utf-8")),
        }, indent=2), encoding="utf-8")

    baseline = metrics(REPO / "results" / "two_mode" / "strict" / "final_spectra.npz")
    v6b = metrics(out / "final_spectra.npz")
    summary = {"baseline": baseline, "v6b": v6b, "delta_J": v6b["J"] - baseline["J"],
               "supplemental": supp, "drc_ok": bool(drc_ok)}
    (out / "spectrum_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[Audit] spectrum audit", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
