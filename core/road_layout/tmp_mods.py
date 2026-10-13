"""TruckersMP's own map additions: the `.mp` archives its launcher keeps for the game.

Read only and local. Nothing is decrypted (the files are plain zlib or zip), copied,
persisted or sent anywhere, and no file is held open. Only the mods the launcher
currently lists in its file cache count: an event mod stays on disk long after the
event, and applying it would replace roads the server is not using. Among those the
newest file wins. See core/map_thread/README.md.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from pathlib import Path

from core.road_layout.archive import Archive, open_archive

log = logging.getLogger("road_layout")

_GAME_FOLDERS = {"ets2": "ets2", "ats": "ats"}


def _install_dir() -> Path | None:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    root = Path(appdata) / "TruckersMP"
    default = root / "installation"
    try:
        options = json.loads((root / "launcher-options.json").read_text(encoding="utf-8"))
        configured = options.get("installPath")
        if isinstance(configured, str) and Path(configured).is_dir():
            return Path(configured)
    except (OSError, ValueError, AttributeError):
        pass
    return default if default.is_dir() else None


def _listed_files(base: Path) -> list[str]:
    """Paths the launcher's file cache lists (names and hashes only), empty when unreadable."""
    try:
        listing = json.loads((base / "filecache.dat").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [key for key in listing if isinstance(key, str)] if isinstance(listing, dict) else []


def tmp_mod_paths(game: str) -> list[Path]:
    """The `.mp` archives the launcher currently lists for *game*, oldest first so the newest wins."""
    folder = _GAME_FOLDERS.get(game)
    base = _install_dir()
    if folder is None or base is None:
        return []
    wanted = (f"/data/{folder}/mods/", "/data/shared/mods/")
    found: list[Path] = []
    for key in _listed_files(base):
        if key.startswith(wanted) and key.endswith(".mp") and ".." not in key:
            path = base / key.lstrip("/")
            if path.is_file():
                found.append(path)
    try:
        return sorted(found, key=lambda p: (p.stat().st_mtime, p.name))
    except OSError:
        return []


class TmpOverlayOpener:
    """Opens the TruckersMP archives one per ``step``, like ``GameMapOpener``."""

    def __init__(self, game: str) -> None:
        self._pending = tmp_mod_paths(game)
        self.archives: list[Archive] = []
        self.done = False

    def step(self) -> bool:
        """Open one archive, or finish. True once nothing is left to do."""
        if self.done:
            return True
        if not self._pending:
            self.done = True
            return True
        path = self._pending.pop(0)
        try:
            self.archives.append(open_archive(path))
        except (OSError, ValueError, struct.error):
            log.warning("TruckersMP archive %s skipped", path.name)
        return False
