"""Measure the cost of the new meshes using the properly generated single-source .fsp files.

usage:  python time_mesh_impact.py <repo>
"""
import os
import re
import sys
import time
import glob

repo = os.path.abspath(sys.argv[1])
sys.path.insert(0, repo)
os.chdir(repo)

from func_MDM_inverse_design import lumapi  # noqa: E402

fsp = os.path.join(repo, "TO_base", "TE_mode_0_TO.fsp")
NEW_MESHES = ("design_mesh", "source_mesh1", "source_mesh2")
results = {}

for label, drop in (("without new meshes", True), ("with new meshes", False)):
    print("=" * 70)
    print(label)
    before = set(glob.glob(os.path.join(repo, "*.log"))) | set(glob.glob(os.path.join(repo, "TO_base", "*.log")))
    fdtd = lumapi.FDTD(hide=True)
    try:
        fdtd.load(fsp)
        if drop:
            for nm in NEW_MESHES:
                try:
                    fdtd.eval('select("{}"); delete;'.format(nm))
                except Exception as exc:
                    print("   could not delete {}: {}".format(nm, exc))
        t0 = time.time()
        try:
            fdtd.eval("run;")
            dt = time.time() - t0
            print("   run wall time : {:.1f} s".format(dt))
            results[label] = dt
        except Exception as exc:
            print("   run RAISED: {}".format(exc))
            results[label] = None

        after = set(glob.glob(os.path.join(repo, "*.log"))) | set(glob.glob(os.path.join(repo, "TO_base", "*.log")))
        new_logs = sorted(after - before, key=os.path.getmtime)
        if new_logs:
            txt = open(new_logs[-1], errors="replace").read()
            g = re.findall(r"Simulation size in gridpoints: (\d+ x \d+ x \d+)", txt)
            sp = re.findall(r"total FDTD solver speed on \d+ processes \(Mnodes/s\): ([\d.]+)", txt)
            w = re.findall(r"Overall wall time measurements in seconds: ([\d.]+)", txt)
            if g:
                print("   grid points   :", g[-1])
            if sp:
                print("   solver speed  : {} Mnodes/s".format(sp[-1]))
            if w:
                print("   engine wall   : {} s".format(w[-1]))
        else:
            print("   (no engine log produced -> the run did not actually happen)")
    finally:
        try:
            fdtd.close()
        except Exception:
            pass

print("=" * 70)
a, b = results.get("without new meshes"), results.get("with new meshes")
if a and b:
    print("without : {:7.1f} s".format(a))
    print("with    : {:7.1f} s".format(b))
    print("ratio   : {:.2f}x  ({:+.0f}%)".format(b / a, 100 * (b / a - 1)))
else:
    print("results:", results)
