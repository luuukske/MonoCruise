"""Windows "Installed apps" entry, kept in step with the version on disk.

The Inno Setup installer writes the uninstall registry key once, at install
time (`installer/MonoCruise.iss`). The in-app updater replaces the program
files without ever rerunning the installer, so Windows keeps reporting
whichever version was first installed: someone who installed preview.9 and has
auto-updated ever since still reads "MonoCruise 1.1.0-preview.9" in
Settings > Apps. Two callers fix that:

* `updater/updater.py`, right after a successful install,
* `monocruise.py` at boot, which also heals installs that were updated before
  this existed.

Everything here is best-effort. A stale Apps entry is cosmetic: it must never
stop the app from starting or fail an otherwise good update.
"""
from __future__ import annotations

import logging
import ntpath
import os

try:  # Windows-only; the module must stay importable for the Linux test run.
    import winreg
except ImportError:  # pragma: no cover - exercised by the non-Windows path
    winreg = None

log = logging.getLogger("app_details")

APP_NAME = "MonoCruise"

# Inno names the key "{AppId}_is1" and MonoCruise.iss sets no AppId, so AppName
# stands in. PrivilegesRequired=lowest means HKCU; HKLM covers elevated builds.
UNINSTALL_KEY = rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{APP_NAME}_is1"

# DisplayName carries the version too (UninstallDisplayName builds it that way),
# and the Apps list shows the name far more prominently than the version column.
VALUE_NAMES = ("DisplayVersion", "DisplayName")


def display_name(version: str) -> str:
    """Apps-list name for *version*, in the shape MonoCruise.iss installs."""
    version = (version or "").strip()
    return f"{APP_NAME} {version}" if version else APP_NAME


def plan_values(version: str, current: dict) -> dict:
    """Uninstall-key values that need rewriting for *version*.

    *current* is what the key holds now, with missing values as None. Returns an
    empty dict when the entry already matches, so an in-sync install performs no
    registry writes at all.
    """
    version = (version or "").strip()
    if not version:
        return {}
    wanted = {"DisplayVersion": version, "DisplayName": display_name(version)}
    plan = {}
    for name, value in wanted.items():
        have = current.get(name)
        if have == value:
            continue
        # A display name that is not ours was set by hand or by another tool;
        # correct the version but leave someone else's rename standing.
        if name == "DisplayName" and have and not str(have).startswith(APP_NAME):
            continue
        plan[name] = value
    return plan


def entry_matches_install(location: str, install_root: str) -> bool:
    """True when an uninstall entry's InstallLocation points at *install_root*.

    Keeps a source checkout, or a second copy installed elsewhere, from
    relabelling an install it does not own. An entry with no InstallLocation
    recorded counts as ours: the key name already matched.
    """
    if not location or not install_root:
        return True
    return _norm(location) == _norm(install_root)


def _norm(path: str) -> str:
    # Registry paths are Windows paths whatever the host OS, so ntpath, not os.path.
    return ntpath.normcase(ntpath.normpath(path))


def _read(key, name: str):
    """One value from an open key, or None when it is not set."""
    try:
        return winreg.QueryValueEx(key, name)[0]
    except OSError:
        return None


def sync_app_details(version: str, install_root: str) -> bool:
    """Relabel the Windows uninstall entry for *install_root* to *version*.

    Returns True when something was written. Never raises: callers treat a
    failed relabel as a non-event.
    """
    if winreg is None or os.name != "nt" or not version:
        return False

    for hive, view in (
        (winreg.HKEY_CURRENT_USER, 0),
        (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_64KEY),
        (winreg.HKEY_LOCAL_MACHINE, winreg.KEY_WOW64_32KEY),
    ):
        access = winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE | view
        try:
            with winreg.OpenKey(hive, UNINSTALL_KEY, 0, access) as key:
                if not entry_matches_install(_read(key, "InstallLocation") or "", install_root):
                    continue  # a different MonoCruise install owns this entry
                plan = plan_values(version, {n: _read(key, n) for n in VALUE_NAMES})
                for name, value in plan.items():
                    winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
                if plan:
                    log.info("Windows app entry relabelled to %s", version)
                return bool(plan)
        except FileNotFoundError:
            continue  # no entry in this hive (zip install, or per-user only)
        except OSError:
            # Usually a per-machine entry an unelevated process cannot write.
            log.debug("could not update the Windows app entry", exc_info=True)
    return False
