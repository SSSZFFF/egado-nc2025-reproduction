"""One isolated load+delete+run per process.

usage:  python one_run.py <repo> <csv-of-meshes-to-delete|none>
"""
import os
import re
import sys
import time
import glob
import tempfile

repo = os.path.abspath(sys.argv[1])
todelete = [] if sys.argv[2].lower() == "none" else sys.argv[2].split(",")

sys.path.insert(0, repo)
os.chdir(repo)

from func_MDM_inverse_design import lumapi  # noqa: E402

fsp = os.path.join(repo, "TO_base", "TE_mode_0_TO.fsp")
tmp = tempfile.gettempdir()
before = set(glob.glob(os.path.join(tmp, "**", "*.log"), recursive=True))
before |= set(glob.glob(os.path.join(repo, "TO_base", "*.log")))

fdtd = lumapi.FDTD(hide=True)
try:
    fdtd.load(fsp)
    for nm in todelete:
        try:
            fdtd.eval('select("{}"); delete;'.format(nm))
        except Exception as exc:
            print("  delete {} failed: {}".format(nm, exc))
    t0 = time.time()
    try:
        fdtd.eval("run;")
        raised = None
    except Exception as exc:
        raised = exc
    dt = time.time() - t0

    after = set(glob.glob(os.path.join(tmp, "**", "*.log"), recursive=True))
    after |= set(glob.glob(os.path.join(repo, "TO_base", "*.log")))
    new = sorted(after - before, key=os.path.getmtime)
    grid = ""
    for p in reversed(new[-3:]):
        try:
            txt = open(p, errors="replace").read()
        except Exception:
            continue
        g = re.findall(r"Simulation size in gridpoints: (\d+ x \d+ x \d+)", txt)
        if g:
            grid = g[-1]
            break
    print("deleted={:<44s} {:>6.1f} s  {}".format(
        ",".join(todelete) or "none", dt,
        ("RAISED: " + str(raised)) if raised else ("grid=" + grid if grid else "no engine log")))
finally:
    try:
        fdtd.close()
    except Exception:
        pass
