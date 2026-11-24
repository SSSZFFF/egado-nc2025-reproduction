"""Session helper that can never raise a Lumerical "Save as..." dialog.

The project itself always does

    temp_fdtd.save(<explicit path>)   # see config_for_eg_a2d_fdtd_file()
    temp_fdtd.close()

Saving first means the session has no unsaved changes, so close() is silent.
Any probe script that creates its own session MUST go through this helper --
an eval-then-close() without a save pops a modal "Save as..." dialog on the
user's desktop.

Usage:

    from lum_session import scratch_session
    with scratch_session(repo) as fdtd:
        fdtd.eval(script)
        ...
"""
import contextlib
import os
import tempfile

import lumapi


@contextlib.contextmanager
def scratch_session(scratch_dir, hide=True):
    """Open an FDTD session; always save-then-close so nothing prompts."""
    os.makedirs(scratch_dir, exist_ok=True)
    scratch = os.path.join(scratch_dir, "_scratch_close_{}.fsp".format(os.getpid()))
    fdtd = lumapi.FDTD(hide=hide)
    try:
        yield fdtd
    finally:
        # project pattern: save to an explicit path first, then close
        try:
            fdtd.save(scratch)
        except Exception:
            pass
        try:
            fdtd.close()
        except Exception:
            pass
        for suffix in ("", ".log", "_p0.log"):
            try:
                os.remove(scratch + suffix)
            except Exception:
                pass


def scratch_dir_for(repo):
    return os.path.join(tempfile.gettempdir(), "egado_scratch")
