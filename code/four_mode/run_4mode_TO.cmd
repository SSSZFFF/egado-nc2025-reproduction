@echo off
cd /d "F:\codex_project\inverse_design_sun\EGADO-4mode"
echo. >> "F:\codex_project\inverse_design_sun\EGADO-4mode\to_4mode_resume.log"
echo ====================================================================== >> "F:\codex_project\inverse_design_sun\EGADO-4mode\to_4mode_resume.log"
echo [RESUME] Restarting 4-mode TO from Checkpoint 266 (Run3_153) at %date% %time% >> "F:\codex_project\inverse_design_sun\EGADO-4mode\to_4mode_resume.log"
echo ====================================================================== >> "F:\codex_project\inverse_design_sun\EGADO-4mode\to_4mode_resume.log"
"D:\Program Files\ANSYS Inc\v251\Lumerical\python\python.exe" -u run_to_4mode.py 400 0 --resume "F:\codex_project\inverse_design_sun\EGADO-4mode\TO\x6000_y6000_f0239_m0000_3\parameters_153.npz" >> "F:\codex_project\inverse_design_sun\EGADO-4mode\to_4mode_resume.log" 2>&1
