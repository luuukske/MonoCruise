"""Find ETS2/ATS plugin folders (Windows registry/Steam; no-op elsewhere)."""

from __future__ import annotations

import logging
import re
import string
import sys
from pathlib import Path

import psutil

log = logging.getLogger("sdk")

_IS_WINDOWS = sys.platform == "win32"

# Steam app ids, install folder names and executables per supported game.
GAME_CONFIG: dict[str, dict[str, str]] = {
    "ets2": {
        "steam_id": "227300",
        "folder_name": "Euro Truck Simulator 2",
        "exe_name": "eurotrucks2.exe",
    },
    "ats": {
        "steam_id": "270880",
        "folder_name": "American Truck Simulator",
        "exe_name": "amtrucks.exe",
    },
}

GAME_TYPES: tuple[str, ...] = tuple(GAME_CONFIG)


def is_steam_installed() -> bool:
    """True if a Steam registry key is present."""
    if not _IS_WINDOWS:
        return False
    import winreg

    for hive_path in (
        r"SOFTWARE\WOW6432Node\Valve\Steam",
        r"SOFTWARE\Valve\Steam",
    ):
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, hive_path)
            winreg.CloseKey(key)
            return True
        except OSError:
            continue
    return False


def _steam_library_roots() -> list[Path]:
    """Every Steam library root we can find (install dir + extra libraries)."""
    if not _IS_WINDOWS:
        return []
    import winreg

    roots: list[Path] = []

    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam")
        install_path = winreg.QueryValueEx(key, "InstallPath")[0]
        winreg.CloseKey(key)
        roots.append(Path(install_path))
    except OSError:
        pass

    roots.extend([Path("C:/Program Files (x86)/Steam"), Path("C:/Program Files/Steam")])

    # libraryfolders.vdf lists every library the user added, on any drive.
    extra: list[Path] = []
    for root in roots:
        vdf = root / "steamapps" / "libraryfolders.vdf"
        if not vdf.exists():
            continue
        try:
            content = vdf.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lib in re.findall(r'"path"\s*"([^"]+)"', content):
            extra.append(Path(lib.replace("\\\\", "/")))
    roots.extend(extra)
    return roots


def _uninstall_key_path(game_type: str) -> Path | None:
    """Install location Steam records in the per-app uninstall registry key."""
    if not _IS_WINDOWS:
        return None
    import winreg

    steam_id = GAME_CONFIG[game_type]["steam_id"]
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            rf"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Steam App {steam_id}",
        )
        path = winreg.QueryValueEx(key, "InstallLocation")[0]
        winreg.CloseKey(key)
        candidate = Path(path)
        return candidate if candidate.exists() else None
    except OSError:
        return None


def validate_game_installation(game_path: Path, game_type: str) -> bool:
    """True if ``game_path`` holds the expected game executable."""
    config = GAME_CONFIG.get(game_type)
    if not config or not game_path:
        return False
    return (game_path / "bin" / "win_x64" / config["exe_name"]).exists()


def find_game_installations(game_type: str) -> list[Path]:
    """Registry, Steam libraries, and common paths; de-duplicated valid installs."""
    config = GAME_CONFIG.get(game_type)
    if not config or not is_steam_installed():
        return []

    folder = config["folder_name"]
    candidates: list[Path] = []

    uninstall = _uninstall_key_path(game_type)
    if uninstall is not None:
        candidates.append(uninstall)

    for root in _steam_library_roots():
        candidates.append(root / "steamapps" / "common" / folder)

    for drive in string.ascii_uppercase:
        candidates.extend(
            [
                Path(f"{drive}:/Program Files (x86)/Steam/steamapps/common/{folder}"),
                Path(f"{drive}:/Program Files/Steam/steamapps/common/{folder}"),
                Path(f"{drive}:/Steam/steamapps/common/{folder}"),
                Path(f"{drive}:/Games/Steam/steamapps/common/{folder}"),
                Path(f"{drive}:/SteamLibrary/steamapps/common/{folder}"),
            ]
        )

    valid: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if validate_game_installation(candidate, game_type):
            valid.append(candidate)
    return valid


def get_plugins_dir(game_path: Path) -> Path:
    """Plugin folder the SDK DLLs live in for a given installation."""
    return game_path / "bin" / "win_x64" / "plugins"


def is_game_running(game_type: str) -> bool:
    """True if the game's process is currently running."""
    config = GAME_CONFIG.get(game_type)
    if not config:
        return False
    target = config["exe_name"].lower()
    try:
        for proc in psutil.process_iter(["name"]):
            name = proc.info.get("name")
            if name and name.lower() == target:
                return True
    except psutil.Error:
        pass
    return False


def close_game(game_type: str) -> bool:
    """Terminate game process; kill if needed. Caller must have user consent."""
    config = GAME_CONFIG.get(game_type)
    if not config:
        return False
    target = config["exe_name"].lower()

    procs: list[psutil.Process] = []
    try:
        for proc in psutil.process_iter(["name"]):
            name = proc.info.get("name")
            if name and name.lower() == target:
                procs.append(proc)
    except psutil.Error:
        return False

    for proc in procs:
        try:
            proc.terminate()
        except psutil.Error:
            pass

    _, alive = psutil.wait_procs(procs, timeout=10)
    for proc in alive:
        try:
            proc.kill()
        except psutil.Error:
            pass

    return not is_game_running(game_type)
