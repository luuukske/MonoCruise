"""Map packages the game itself mounted, read back from its log.

``game.log.txt`` stays on disk after the game exits, so a clip review can rebuild
the same packages without the game running. Only a path under that user
directory's ``mod`` folder, or the Steam workshop folder for ETS2 or ATS, is
opened. The log is read, never written, and it is not held open. Nothing here
is decrypted. See core/map_thread/README.md.
"""

from __future__ import annotations

import logging
import re
import struct
from pathlib import Path

from core.road_layout.archive import MountedMod, open_mounted
from core.scs_profile.reader import game_user_dir
from core.sdk_installer.game_paths import GAME_CONFIG

log = logging.getLogger("road_layout")

_MOUNT_RE = re.compile(r"\[fs\] device (.+?) mounted to mod pool\.?\s*$")
_PROFILE_MARK = "New profile selected:"
_SLICE = 1 << 20
_WORKSHOP = "workshop"
_CONTENT = "content"
_APP_IDS = {cfg["steam_id"] for cfg in GAME_CONFIG.values()}

# One live reader per game. Tools parse a log in one shot and do not use this.
_LIVE: dict[str, "ModLogScan"] = {}


def mod_log_path(game: str) -> Path | None:
    """``game.log.txt`` for *game*, if that user directory is on disk."""
    root = game_user_dir(game)
    if root is None:
        return None
    path = root / "game.log.txt"
    return path if path.is_file() else None


def mod_mount_paths(game: str) -> list[Path]:
    """Every mod package the current log says the game mounted, in mount order."""
    path = mod_log_path(game)
    root = game_user_dir(game)
    if path is None or root is None:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        log.warning("map mod log unreadable, base map stays")
        return []
    return mounts_in(text, root)


def mounts_in(text: str, user_dir: Path) -> list[Path]:
    """Packages named by *text* that are allowed for *user_dir*, after the last profile change."""
    found: list[Path] = []
    for line in text.splitlines():
        _take(line, user_dir, found)
    return found


def live_mounts(game: str) -> tuple[Path, ...] | None:
    """Mounts for the running session, reading at most 1 MB per call.

    ``None`` while the log still has unread bytes, so a caller does not layer a
    half-read list. A replaced log (its first bytes changed, or it shrank) starts over.
    """
    path = mod_log_path(game)
    root = game_user_dir(game)
    if path is None or root is None:
        return ()
    scan = _LIVE.get(game)
    if scan is None or scan.path != path:
        scan = ModLogScan(path, root)
        _LIVE[game] = scan
    if not scan.step():
        return None
    return tuple(scan.mounts)


class ModLogScan:
    """Incremental read of one ``game.log.txt``. The file is not held open."""

    def __init__(self, path: Path, user_dir: Path) -> None:
        self.path = path
        self._user_dir = user_dir
        self.mounts: list[Path] = []
        self._offset = 0
        self._tail = ""
        self._head = b""

    def step(self) -> bool:
        """Read one slice. True once this call reached the current end of the file."""
        try:
            with self.path.open("rb") as fh:
                head = fh.read(64)
                size = self.path.stat().st_size
                if head != self._head or size < self._offset:
                    self.mounts = []
                    self._offset = 0
                    self._tail = ""
                    self._head = head
                if self._offset >= size:
                    return True
                fh.seek(self._offset)
                blob = fh.read(_SLICE)
        except OSError:
            log.warning("map mod log unreadable")
            return True
        self._offset += len(blob)
        text = self._tail + blob.decode("utf-8", "replace")
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._tail = lines.pop()
        else:
            self._tail = ""
        for line in lines:
            _take(line, self._user_dir, self.mounts)
        try:
            return self._offset >= self.path.stat().st_size
        except OSError:
            return True


class ModOverlayOpener:
    """Opens the game's mounted mod packages one per ``step``."""

    def __init__(self, game: str, mounts: list[Path] | tuple[Path, ...] | None = None) -> None:
        self._pending = list(mod_mount_paths(game) if mounts is None else mounts)
        self.archives: list[MountedMod] = []
        self.done = False

    def step(self) -> bool:
        """Open one package, or finish. True once nothing is left to do."""
        if self.done:
            return True
        if not self._pending:
            self.done = True
            return True
        path = self._pending.pop(0)
        try:
            self.archives.append(open_mounted(path))
        except (OSError, ValueError, struct.error):
            log.warning("map mod %s skipped", path.name)
        return False


def _take(line: str, user_dir: Path, found: list[Path]) -> None:
    if _PROFILE_MARK in line:
        found.clear()
        return
    match = _MOUNT_RE.search(line)
    if match is None:
        return
    accepted = accept_mount(Path(match.group(1).strip()), user_dir)
    if accepted is not None:
        found.append(accepted)


def accept_mount(path: Path, user_dir: Path) -> Path | None:
    """*path* when it is a real package under the mod folder or the game's workshop folder."""
    try:
        if not path.is_file() and not path.is_dir():
            return None
        resolved = path.resolve()
    except OSError:
        return None
    mod_root = user_dir / "mod"
    if _is_under(resolved, mod_root) or _is_workshop(resolved):
        return path
    return None


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _is_workshop(path: Path) -> bool:
    parts = [part.casefold() for part in path.parts]
    ids = {app.casefold() for app in _APP_IDS}
    for index, part in enumerate(parts[:-1]):
        if part != _WORKSHOP or index + 2 >= len(parts):
            continue
        if parts[index + 1] == _CONTENT and parts[index + 2] in ids:
            return True
    return False
