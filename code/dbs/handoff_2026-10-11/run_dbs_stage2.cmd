@echo off
cd /d "F:\codex_project\inverse_design_sun\DBS_handoff_2026-10-11"
set OMP_NUM_THREADS=20
py -3.12 fix_ini_gpu.py
if not exist "results\B_curve_stage2_joint" mkdir "results\B_curve_stage2_joint"
py -3.12 -u run_dbs_stage2.py --scheme B --initial-state-npz "F:\codex_project\inverse_design_sun\DBS_handoff_2026-10-11\results\B_curve_DRC120_strict\final_state.npz" --output-dir "F:\codex_project\inverse_design_sun\DBS_handoff_2026-10-11\results\B_curve_stage2_joint" >> "results\B_curve_stage2_joint\dbs_run.log" 2>&1
