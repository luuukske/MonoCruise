"""MonoCruise background game checker.

Tiny companion program that starts MonoCruise automatically when Euro Truck
Simulator 2 or American Truck Simulator is running. Installed to
``<install root>/checker/`` and (with the user's consent, via an installer
checkbox) added to Windows startup.

What it does, in full:
  1. Once per second, tries to open the SCS telemetry shared-memory block
     (``Local\\SCSTelemetry``) that the in-game SDK plugin publishes. The block
     only exists while the game is running; its first byte is the plugin's
     "SDK active" flag.
  2. When the game comes up and MonoCruise is not already open, reads the
     "Autostart MonoCruise" setting from MonoCruise's ``config.json`` and, if it
     is on, starts ``MonoCruise.exe`` from the install root. At most once per
     game session, so quitting MonoCruise mid-game does not relaunch it.
  3. When the game closes, goes back to waiting.

What it deliberately does NOT do:
  - no network access of any kind
  - no reading of input devices or personal data, and no files other than
    that one setting in ``config.json``
  - no registry access (the "start with Windows" entry is written by the
    installer, shown as a checkbox, and removed by the uninstaller)
  - no process enumeration; "is MonoCruise open?" is answered by checking the
    named mutex MonoCruise holds while it runs (see monocruise.py)

Everything it does is appended to ``checker.log`` next to the executable.
To stop it running at startup, disable "MonoCruiseChecker" under
Task Manager > Startup apps, or re-run the installer with the checkbox off.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from multiprocessing.shared_memory import SharedMemory
from subprocess import Popen

POLL_SECONDS = 1.0
TELEMETRY_SHM_NAME = "Local\\SCSTelemetry"
# Held by MonoCruise.exe for as long as it runs (created in monocruise.py).
APP_MUTEX_NAME = "MonoCruiseSingleInstance"
# Held by this checker; a second copy sees it and exits immediately.
# MonoCruise probes it too (core/checker_status.py) to grey out its autostart toggle.
CHECKER_MUTEX_NAME = "MonoCruiseCheckerSingleInstance"
# The in-app "Autostart MonoCruise" checkbox, stored in config.json at the install root.
AUTOSTART_KEY = "autostart_variable"

ERROR_ALREADY_EXISTS = 183
SYNCHRONIZE = 0x00100000

logger = logging.getLogger("checker")

_checker_mutex = None  # handle kept for the process lifetime

if sys.platform == "win32":
    import ctypes

    _kernel32 = ctypes.windll.kernel32
    # Explicit signatures: HANDLE is pointer-sized, the ctypes default int
    # return type would truncate it on 64-bit.
    _kernel32.CreateMutexW.restype = ctypes.c_void_p
    _kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
    _kernel32.OpenMutexW.restype = ctypes.c_void_p
    _kernel32.OpenMutexW.argtypes = (ctypes.c_uint, ctypes.c_int, ctypes.c_wchar_p)
    _kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)


def _base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _setup_logging() -> None:
    """Log to checker.log next to the exe, truncated on every start.

    The log only ever holds a handful of lines per session (start, launch,
    game closed), so users can open it and see exactly what the checker did.
    """
    handlers: list[logging.Handler] = []
    try:
        path = os.path.join(_base_dir(), "checker.log")
        handlers.append(logging.FileHandler(path, mode="w", encoding="utf-8"))
    except OSError:
        pass  # unwritable install dir: run silently rather than crash
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers or None,
    )


def acquire_checker_mutex() -> bool:
    """Single-instance guard, same idiom as the updater (updater/updater.py)."""
    global _checker_mutex
    _checker_mutex = _kernel32.CreateMutexW(None, False, CHECKER_MUTEX_NAME)
    return _kernel32.GetLastError() != ERROR_ALREADY_EXISTS


def monocruise_running() -> bool:
    """True while MonoCruise.exe holds its single-instance mutex."""
    handle = _kernel32.OpenMutexW(SYNCHRONIZE, False, APP_MUTEX_NAME)
    if handle:
        _kernel32.CloseHandle(handle)
        return True
    return False


def game_running() -> bool:
    """True while game runs with telemetry SDK active (read-only SHM open, create=False)."""
    try:
        shm = SharedMemory(name=TELEMETRY_SHM_NAME, create=False)
    except (FileNotFoundError, ValueError, OSError):
        return False
    try:
        return bool(shm.buf[0])
    finally:
        shm.close()


def autostart_enabled() -> bool:
    """The app's autostart setting; on when config.json is missing or unreadable, as before."""
    path = os.path.join(os.path.dirname(_base_dir()), "config.json")
    try:
        with open(path, encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return True
    return not (isinstance(data, dict) and data.get(AUTOSTART_KEY) is False)


def launch_monocruise() -> bool:
    """Start MonoCruise.exe from the install root (parent of this folder)."""
    exe = os.path.join(os.path.dirname(_base_dir()), "MonoCruise.exe")
    if not os.path.isfile(exe):
        return False
    Popen([exe], cwd=os.path.dirname(exe))
    return True


def handle_game_start() -> bool:
    """Start MonoCruise for a new game session if wanted. False means stop the checker."""
    if monocruise_running():
        logger.info("game detected, MonoCruise already open")
    elif not autostart_enabled():
        logger.info("game detected, autostart is off in MonoCruise settings")
    elif launch_monocruise():
        logger.info("game detected, started MonoCruise")
    else:
        logger.error(
            "MonoCruise.exe not found in the folder above the checker; "
            "was it uninstalled? exiting"
        )
        return False
    return True


def main() -> int:
    if sys.platform != "win32":
        print("The MonoCruise checker only works on Windows.")
        return 1
    _setup_logging()
    if not acquire_checker_mutex():
        logger.info("another checker instance is already running, exiting")
        return 0
    logger.info("checker started, polling for the game every %.0f s", POLL_SECONDS)
    while True:
        if game_running():
            if not handle_game_start():
                return 1
            while game_running():
                time.sleep(POLL_SECONDS)
            logger.info("game closed, waiting")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
