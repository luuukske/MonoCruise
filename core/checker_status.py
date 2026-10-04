"""Is the background game checker (checker/ets2_checker.py) running? Read-only mutex probe."""

from __future__ import annotations

import logging
import sys

# Must match CHECKER_MUTEX_NAME in checker/ets2_checker.py (pinned by a test).
CHECKER_MUTEX_NAME = "MonoCruiseCheckerSingleInstance"
_SYNCHRONIZE = 0x00100000

_log = logging.getLogger(__name__)


def checker_running() -> bool:
    """True while the checker holds its single-instance mutex. Never touches the registry."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        # Private WinDLL so these signatures cannot leak onto other kernel32 users.
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.OpenMutexW.restype = ctypes.c_void_p
        kernel32.OpenMutexW.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_wchar_p)
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel32.OpenMutexW(_SYNCHRONIZE, False, CHECKER_MUTEX_NAME)
        if handle:
            kernel32.CloseHandle(handle)
            return True
    except Exception:
        _log.debug("checker status probe failed", exc_info=True)
    return False
