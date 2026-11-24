@echo off
cd /d "G:\codex_project\inverse_design_sun\EGADO-4mode"
if not exist "results_dbs_phase2" mkdir "results_dbs_phase2"
python -u run_dbs_4mode.py --output-dir "G:\codex_project\inverse_design_sun\EGADO-4mode\results_dbs_phase2" --max-sweeps 3 --tolerance 0.0001 >> "results_dbs_phase2\dbs_run.log" 2>&1
