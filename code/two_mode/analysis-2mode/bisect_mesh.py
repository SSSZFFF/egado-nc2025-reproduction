"""Bisect which of the new mesh overrides stops the simulation from running.

usage:  python bisect_mesh.py <repo>
"""
import os
import sys
import time
import glob

repo = os.path.abspath(sys.argv[1])
sys.path.insert(0, repo)
os.chdir(repo)

from func_MDM_inverse_design import lumapi  # noqa: E402

fsp = os.path.join(repo, "TO_base", "TE_mode_0_TO.fsp")
ALL = ["design_mesh", "source_mesh1", "source_mesh2", "fom_mesh"]

cases = [
    ("keep none (baseline)", []),
    ("keep fom_mesh only", ["fom_mesh"]),
    ("keep design_mesh only", ["design_mesh"]),
    ("keep source_mesh1 only", ["source_mesh1"]),
    ("keep source_mesh1+2", ["source_mesh1", "source_mesh2"]),
    ("keep design + source1", ["design_mesh", "source_mesh1"]),
    ("keep all (as saved)", ALL),
]

for label, keep in cases:
    before = set(glob.glob(os.path.join(repo, "TO_base", "*.log")))
    fdtd = lumapi.FDTD(hide=True)
    ok = False
    try:
        fdtd.load(fsp)
        for nm in ALL:
            if nm in keep:
                continue
            try:
                fdtd.eval('select("{}"); delete;'.format(nm))
            except Exception:
                pass
        t0 = time.time()
        try:
            fdtd.eval("run;")
        except Exception as exc:
            print("{:<26} run RAISED  {}".format(label, exc))
            continue
        dt = time.time() - t0
        new_logs = sorted(set(glob.glob(os.path.join(repo, "TO_base", "*.log"))) - before)
        ok = bool(new_logs) or dt > 1.0
        grid = ""
        if new_logs:
            import re
            txt = open(new_logs[-1], errors="replace").read()
            g = re.findall(r"Simulation size in gridpoints: (\d+ x \d+ x \d+)", txt)
            if g:
                grid = g[-1]
        print("{:<26} {:>7.1f} s   {}".format(label, dt, "RAN  grid=" + grid if ok else "DID NOT RUN"))
    finally:
        try:
            fdtd.close()
        except Exception:
            pass
