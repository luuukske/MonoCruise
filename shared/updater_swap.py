"""Applies a staged updater self-update (updater_pending/ -> updater/).

Stdlib only: imported by MonoCruise at boot, keep it Qt-free.

Why this lives in the APP and not the updater: a running one-folder
PyInstaller program holds an open handle to its own _internal/base_library.zip
without FILE_SHARE_DELETE (zipimport), so that file cannot be renamed while
the process lives, by any process. The updater therefore can never finish its
own swap in-process; MonoCruise applies it once the updater has exited
(single-instance mutex free). Verified empirically: of the updater's ~240
files, base_library.zip is the one rename that fails while it runs.

Layout, all directly under the install root (see updater/updater.py, which
stages these; the directory names must match on both sides):

  updater/          the updater program
  updater_pending/  new updater files staged by an update; a sentinel file
                    written last marks the stage complete
  updater_old/      previous updater files parked by a swap
"""

from __future__ import annotations

import logging
import os
import shutil
import sys

# Must match the constants in updater/updater.py (tests assert this).
PENDING_DIR = "updater_pending"
OLD_DIR = "updater_old"
UPDATER_DIR = "updater"
PENDING_SENTINEL = ".complete"

UPDATER_MUTEX_NAME = "MonoCruiseUpdaterSingleInstance"

log = logging.getLogger("updater_swap")


def updater_running() -> bool:
    """True while an updater instance holds its single-instance mutex."""
    if sys.platform != "win32":
        return False
    import ctypes

    kernel32 = ctypes.windll.kernel32
    kernel32.OpenMutexW.restype = ctypes.c_void_p
    kernel32.OpenMutexW.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_wchar_p)
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    SYNCHRONIZE = 0x00100000
    handle = kernel32.OpenMutexW(SYNCHRONIZE, False, UPDATER_MUTEX_NAME)
    if handle:
        kernel32.CloseHandle(handle)
        return True
    return False


def _move_tree(src: str, dst: str) -> None:
    """Move every file under src into dst via per-file renames."""
    for dirpath, _dirnames, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        target_dir = dst if rel == os.curdir else os.path.join(dst, rel)
        os.makedirs(target_dir, exist_ok=True)
        for name in filenames:
            os.replace(os.path.join(dirpath, name), os.path.join(target_dir, name))


def pending_swap_staged(root: str) -> bool:
    """True when a completed updater update is waiting to be swapped in.

    Cheap enough for a poll loop: one isdir when nothing is staged.
    """
    pending = os.path.join(root, PENDING_DIR)
    return os.path.isdir(pending)


def apply_pending_updater_swap(root: str, *, running_check=updater_running) -> bool:
    """Swap staged updater files into place. Returns True when applied.

    Call only while the updater is not running; rechecked here via its mutex.
    Never raises. Failure order is chosen so the stage stays retryable:

      1. park updater/ -> updater_old/   (fails => rollback, sentinel intact,
                                          the swap is retried later)
      2. remove the sentinel             (so it does not land in updater/)
      3. move updater_pending/ -> updater/
      4. delete leftovers

    A crash after step 2 leaves a sentinel-less pending dir, which is
    discarded here on the next attempt (the update can simply be re-run).
    """
    pending = os.path.join(root, PENDING_DIR)
    udir = os.path.join(root, UPDATER_DIR)
    old = os.path.join(root, OLD_DIR)

    if not os.path.isdir(pending):
        # No stage; still clean up a parked tree from a previous swap.
        shutil.rmtree(old, ignore_errors=True)
        return False
    sentinel = os.path.join(pending, PENDING_SENTINEL)
    if not os.path.isfile(sentinel):
        log.warning("discarding incomplete updater stage")
        shutil.rmtree(pending, ignore_errors=True)
        return False
    if running_check():
        return False  # updater still holds its files; retried by the caller

    shutil.rmtree(old, ignore_errors=True)
    try:
        _move_tree(udir, old)
    except OSError:
        log.warning("could not park the current updater; will retry", exc_info=True)
        try:
            _move_tree(old, udir)
        except OSError:
            log.error("rollback of updater park failed", exc_info=True)
        return False

    try:
        os.remove(sentinel)
        _move_tree(pending, udir)
    except OSError:
        log.error("updater swap failed; restoring previous updater", exc_info=True)
        try:
            _move_tree(old, udir)  # os.replace overwrites half-moved new files
        except OSError:
            log.error("rollback of updater swap failed", exc_info=True)
        shutil.rmtree(pending, ignore_errors=True)  # sentinel gone: not retryable
        return False

    shutil.rmtree(pending, ignore_errors=True)
    shutil.rmtree(old, ignore_errors=True)
    log.info("updater self-update applied")
    return True
