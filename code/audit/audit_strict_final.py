"""Read-only audit of the strict DBS and Stage 2 artifacts.

This script never opens Lumerical and never runs DRC. It only reads the saved
JSON/CSV/NPZ/GDS artifacts and recomputes the reported scalar metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def optical_metrics(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        T = np.asarray(data["T"], dtype=float)
        J_saved = float(np.asarray(data["J"]))
        wavelength_nm = np.asarray(data["wavelength_nm"], dtype=float)

    t_te0 = T[0, 0]
    t_te1 = T[1, 1]
    x_te0 = T[0, 1]
    x_te1 = T[1, 0]
    t0 = float(t_te0.mean())
    t1 = float(t_te1.mean())
    c0 = float(x_te0.mean())
    c1 = float(x_te1.mean())
    return {
        "saved_J": J_saved,
        "recomputed_J": float((t0 + t1) / 2.0),
        "J_abs_error": float(abs(J_saved - (t0 + t1) / 2.0)),
        "shape": list(T.shape),
        "wavelength_nm": wavelength_nm.tolist(),
        "target_mean": {"top_to_TE0": t0, "bottom_to_TE1": t1},
        "target_min": {"top_to_TE0": float(t_te0.min()), "bottom_to_TE1": float(t_te1.min())},
        "target_max": {"top_to_TE0": float(t_te0.max()), "bottom_to_TE1": float(t_te1.max())},
        "insertion_loss_db_from_mean": {
            "top_to_TE0": float(-10.0 * np.log10(t0)),
            "bottom_to_TE1": float(-10.0 * np.log10(t1)),
            "mean_of_two_channel_dB": float((-10.0 * np.log10(t0) - 10.0 * np.log10(t1)) / 2.0),
        },
        "crosstalk_ratio_db_from_means": {
            "top_input_to_TE1": float(10.0 * np.log10(c0 / t0)),
            "bottom_input_to_TE0": float(10.0 * np.log10(c1 / t1)),
        },
        "crosstalk_worst_frequency_db": {
            "top_input_to_TE1": float(10.0 * np.log10(np.max(x_te0 / t_te0))),
            "bottom_input_to_TE0": float(10.0 * np.log10(np.max(x_te1 / t_te1))),
        },
        "cross_mean": {"top_input_to_TE1": c0, "bottom_input_to_TE0": c1},
    }


def history_metrics(path: Path, initial_J: float | None = None) -> dict[str, Any]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    accepted = [row for row in rows if row.get("accepted") == "1"]
    passed = [row for row in rows if row.get("drc_status") == "passed"]
    pdk_failed = [row for row in rows if row.get("drc_status", "").startswith("pdk_fail")]
    supplemental_failed = [
        row for row in rows if row.get("drc_status", "").startswith("supp_fail")
    ]
    evaluated = [row for row in rows if row.get("J_trial", "") != ""]

    reconstructed_before_mismatches: list[dict[str, Any]] = []
    reconstructed_deltas: list[float] = []
    if initial_J is not None:
        current = float(initial_J)
        for row in rows:
            trial_text = row.get("J_trial", "")
            if not trial_text:
                continue
            trial = float(trial_text)
            before = current
            if row.get("accepted") == "1":
                logged_before = float(row["J_before"])
                if abs(logged_before - before) > 5e-7:
                    reconstructed_before_mismatches.append(
                        {
                            "sweep": int(row["sweep"]),
                            "step": int(row["step"]),
                            "logged_J_before": logged_before,
                            "reconstructed_J_before": before,
                        }
                    )
                reconstructed_deltas.append(trial - before)
                current = trial

    return {
        "rows": len(rows),
        "accepted": len(accepted),
        "evaluated_after_drc": len(evaluated),
        "drc_passed": len(passed),
        "drc_pdk_failed": len(pdk_failed),
        "drc_supplemental_failed": len(supplemental_failed),
        "reported_forward_solves_last": int(rows[-1]["total_forward_solves"]) if rows else None,
        "reconstructed_accept_delta_min": min(reconstructed_deltas) if reconstructed_deltas else None,
        "reconstructed_accept_delta_max": max(reconstructed_deltas) if reconstructed_deltas else None,
        "reconstructed_J_before_mismatch_count": len(reconstructed_before_mismatches),
        "reconstructed_J_before_mismatches": reconstructed_before_mismatches,
        "max_evaluated_trial_minus_logged_before": (
            max(float(row["J_trial"]) - float(row["J_before"]) for row in evaluated)
            if evaluated
            else None
        ),
    }


def artifact_summary(root: Path, result_dir: str) -> dict[str, Any]:
    directory = root / "results" / result_dir
    drc = read_json(directory / "final_DRC" / "result.json")
    supplement = read_json(directory / "final_120nm_checks.json")
    final_metrics = optical_metrics(directory / "final_spectra.npz")
    initial_metrics = optical_metrics(directory / "initial_spectra.npz")
    history = history_metrics(
        directory / "history.csv",
        initial_J=initial_metrics["saved_J"],
    )
    final_gds = directory / "final.gds"
    final_state = directory / "final_state.npz"
    return {
        "directory": str(directory),
        "final_gds_sha256": sha256(final_gds),
        "drc_gds_sha256": drc.get("gds_sha256"),
        "drc_gds_hash_matches_local": sha256(final_gds) == str(drc.get("gds_sha256", "")).upper(),
        "drc_report_path_exists_from_current_workspace": Path(str(drc.get("gds", ""))).is_file(),
        "drc_rule_categories": drc.get("rule_categories"),
        "drc_violation_count": drc.get("violation_count"),
        "drc_status": drc.get("status"),
        "drc_scope_note": drc.get("scope_note"),
        "supplemental": supplement,
        "initial_optical": initial_metrics,
        "final_optical": final_metrics,
        "history": history,
        "final_state_sha256": sha256(final_state),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()

    strict = artifact_summary(root, "B_curve_DRC120_strict")
    stage2 = artifact_summary(root, "B_curve_stage2_joint")

    with np.load(root / "results/B_curve_DRC120_strict/final_state.npz", allow_pickle=False) as a:
        strict_state = np.asarray(a["silicon_state"])
    with np.load(root / "results/B_curve_stage2_joint/final_state.npz", allow_pickle=False) as b:
        stage2_state = np.asarray(b["silicon_state"])
    with np.load(root / "results/B_curve_DRC120_strict/final_spectra.npz", allow_pickle=False) as a:
        strict_T = np.asarray(a["T"])
    with np.load(root / "results/B_curve_stage2_joint/final_spectra.npz", allow_pickle=False) as b:
        stage2_T = np.asarray(b["T"])

    checkpoint_note: dict[str, Any] = {"exists": False}
    checkpoint_meta_path = root / "results/B_curve_DRC120_strict/latest_checkpoint.json"
    if checkpoint_meta_path.is_file():
        meta = read_json(checkpoint_meta_path)
        checkpoint_state_path = (
            root
            / "results/B_curve_DRC120_strict/checkpoints"
            / f"state_sweep{int(meta['sweep']) + 1}_step{int(meta['next_step'])}.npz"
        )
        checkpoint_note = {
            "exists": True,
            "metadata": meta,
            "state_path": str(checkpoint_state_path),
            "state_path_exists": checkpoint_state_path.is_file(),
        }
        if checkpoint_state_path.is_file():
            with np.load(checkpoint_state_path, allow_pickle=False) as data:
                checkpoint_note["state_diff_count_vs_final"] = int(
                    np.count_nonzero(np.asarray(data["silicon_state"]) != strict_state)
                )

    output = {
        "audit": "read_only_strict_final_and_stage2",
        "root": str(root),
        "fdtd_or_new_drc_invoked": False,
        "strict": strict,
        "stage2": stage2,
        "strict_stage2_state_equal": bool(np.array_equal(strict_state, stage2_state)),
        "strict_stage2_T_max_abs_difference": float(np.max(np.abs(strict_T - stage2_T))),
        "strict_stage2_state_diff_count": int(np.count_nonzero(strict_state != stage2_state)),
        "checkpoint_consistency": checkpoint_note,
        "config_status": read_json(root / "DBS_交接包_2026-10-11/config/B_strict.json").get("status"),
    }
    output_path = args.output or root / "analysis-recheck-20261013/strict_final_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
