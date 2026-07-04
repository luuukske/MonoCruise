"""Detect, fetch and install the ETS2 / ATS SDK DLLs.

This is the backend only: it decides *what* needs doing and can carry it out,
but never shows UI. A front-end drives it by calling :func:`check_sdk` (or
:func:`start_boot_check` for a non-blocking boot check), inspecting the returned
:class:`SdkCheckResult`, and calling :meth:`SdkManager.apply` once the user has
agreed.

Boot policy
-----------
* Locate every ETS2 / ATS install and check which managed files are present.
  This is entirely local and fast.
* The repository is only consulted (one JSON request) when a file is missing,
  or when this build opts into a forced refetch and the running version has not
  refetched yet. When every DLL is already installed on a normal build we never
  touch the network.

Forced refetch
--------------
Set :data:`FORCE_REFETCH` to ``True`` in a build that must re-pull the ETS2LA
plugin even though the DLLs look installed (e.g. shipping a fix for a bad plugin
build). It runs once: after the SDK is confirmed up to date the running
MonoCruise version is recorded and later boots skip the network again.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from core.settings import CONFIG_PATH
from core.version import __version__

from .game_paths import (
    GAME_TYPES,
    close_game,
    find_game_installations,
    get_plugins_dir,
    is_game_running,
    is_steam_installed,
)
from .remote import RemoteFile, SdkSource, SdkSourceError, git_blob_sha_of

log = logging.getLogger("sdk")

# Game engine version the bundled ETS2LA plugin supports. Bump this when the
# plugin gains support for a newer game version (a matching Assets/SDKs/<ver>
# folder must exist in the ETS2LA repository).
SUPPORTED_GAME_VERSION = "1.60"

# Flip to True in a build that must re-fetch the plugin even when the DLLs are
# already installed. See the module docstring.
FORCE_REFETCH = False

# DLLs that must be present for MonoCruise to talk to the game.
DLL_FILES: tuple[str, ...] = (
    "scs-telemetry.dll",
    "scs_sdk_controller.dll",
    "ets2la_plugin.dll",
)
# Informational file copied alongside the DLLs when available (documents where
# each DLL comes from). Its absence never counts as a problem.
COURTESY_FILES: tuple[str, ...] = ("sources.txt",)

_STATE_FILE = "sdk_state.json"
_CACHE_DIRNAME = "sdk_cache"


def _marker_name(version: str) -> str:
    """ETS2LA version-marker filename (the version lives in the name)."""
    return f"ets2la_{version}"


# Result types the front-end reads


@dataclass(frozen=True)
class ManagedFileState:
    name: str
    installed: bool
    up_to_date: bool  # meaningful only when the remote was consulted


@dataclass
class GameSdkState:
    game_type: str
    game_path: Path
    plugins_dir: Path
    running: bool
    files: list[ManagedFileState]

    @property
    def missing(self) -> list[str]:
        return [f.name for f in self.files if not f.installed]

    @property
    def outdated(self) -> list[str]:
        return [f.name for f in self.files if f.installed and not f.up_to_date]

    @property
    def needs_action(self) -> bool:
        return bool(self.missing or self.outdated)


@dataclass
class SdkCheckResult:
    supported_version: str
    steam_installed: bool
    consulted_remote: bool
    remote_error: str | None
    games: list[GameSdkState] = field(default_factory=list)

    @property
    def found_games(self) -> bool:
        return bool(self.games)

    @property
    def needs_action(self) -> bool:
        return any(g.needs_action for g in self.games)

    @property
    def games_needing_action(self) -> list[GameSdkState]:
        return [g for g in self.games if g.needs_action]


@dataclass
class GameApplyResult:
    game_type: str
    game_path: Path
    installed: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    skipped_running: bool = False

    @property
    def success(self) -> bool:
        return not self.errors and not self.skipped_running


class SdkManager:
    """Stateful helper that checks and installs the SDK for both games."""

    def __init__(
        self,
        *,
        supported_version: str = SUPPORTED_GAME_VERSION,
        data_dir: Path | None = None,
    ):
        self.version = supported_version
        self.marker = _marker_name(supported_version)
        self.tracked_files: tuple[str, ...] = DLL_FILES + (self.marker,)
        self.source = SdkSource(supported_version)

        base = data_dir if data_dir is not None else CONFIG_PATH.parent
        self.cache_dir = base / _CACHE_DIRNAME / supported_version
        self._state_path = base / _STATE_FILE

    # State (records which MonoCruise version last confirmed the SDK)

    def _read_last_checked(self) -> str | None:
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            value = data.get("last_checked_version")
            return value if isinstance(value, str) else None
        except (OSError, ValueError):
            return None

    def _mark_checked(self) -> None:
        try:
            self._state_path.write_text(
                json.dumps({"last_checked_version": __version__}),
                encoding="utf-8",
            )
        except OSError:
            log.debug("could not write SDK state file", exc_info=True)

    # Detection

    def check(self, *, force_remote: bool | None = None) -> SdkCheckResult:
        """Inspect every install and report what (if anything) needs doing.

        Read-only: never downloads, copies or deletes. Consults the repository
        only when needed (see the module docstring) unless ``force_remote`` is
        given explicitly.
        """
        if not is_steam_installed():
            return SdkCheckResult(self.version, False, False, None)

        located: list[tuple[str, Path]] = [
            (game_type, path)
            for game_type in GAME_TYPES
            for path in find_game_installations(game_type)
        ]
        if not located:
            return SdkCheckResult(self.version, True, False, None)

        # Local presence first (fast, offline).
        presence: list[tuple[str, Path, Path, dict[str, bool]]] = []
        any_missing = False
        for game_type, game_path in located:
            plugins = get_plugins_dir(game_path)
            present = {name: (plugins / name).exists() for name in self.tracked_files}
            any_missing = any_missing or not all(present.values())
            presence.append((game_type, game_path, plugins, present))

        forced = FORCE_REFETCH and self._read_last_checked() != __version__
        consult = force_remote if force_remote is not None else (any_missing or forced)

        remote_files: dict[str, RemoteFile] | None = None
        remote_error: str | None = None
        if consult:
            try:
                remote_files = self.source.list_files()
            except SdkSourceError as exc:
                remote_error = str(exc)
                log.warning("SDK check could not reach the ETS2LA source: %s", exc)

        games: list[GameSdkState] = []
        for game_type, game_path, plugins, present in presence:
            file_states: list[ManagedFileState] = []
            for name in self.tracked_files:
                installed = present[name]
                up_to_date = True
                if installed and remote_files and name in remote_files:
                    up_to_date = git_blob_sha_of(plugins / name) == remote_files[name].sha
                file_states.append(ManagedFileState(name, installed, up_to_date))
            games.append(
                GameSdkState(
                    game_type=game_type,
                    game_path=game_path,
                    plugins_dir=plugins,
                    running=is_game_running(game_type),
                    files=file_states,
                )
            )

        result = SdkCheckResult(
            supported_version=self.version,
            steam_installed=True,
            consulted_remote=remote_files is not None,
            remote_error=remote_error,
            games=games,
        )

        # Nothing to do and we actually looked: record the version so a forced
        # refetch does not repeat on every boot.
        if result.consulted_remote and not result.needs_action:
            self._mark_checked()
        return result

    def locate_games(self) -> list[GameSdkState]:
        """Every ETS2 / ATS install as a state object, without consulting the
        repository (per-file up-to-date flags are left optimistic). Used to
        drive a forced reinstall where the current file state does not matter.
        """
        states: list[GameSdkState] = []
        for game_type in GAME_TYPES:
            for game_path in find_game_installations(game_type):
                plugins = get_plugins_dir(game_path)
                files = [
                    ManagedFileState(name, (plugins / name).exists(), True)
                    for name in self.tracked_files
                ]
                states.append(
                    GameSdkState(
                        game_type=game_type,
                        game_path=game_path,
                        plugins_dir=plugins,
                        running=is_game_running(game_type),
                        files=files,
                    )
                )
        return states

    def reinstall_all(
        self,
        *,
        close_running: bool = True,
        on_progress: Callable[[str], None] | None = None,
    ) -> list[GameApplyResult]:
        """Force a fresh download and overwrite of every managed file for every
        detected install. Returns one result per game (empty if none found)."""
        return self.apply(
            self.locate_games(),
            close_running=close_running,
            on_progress=on_progress,
            force_all=True,
        )

    # Installation

    def _ensure_cached(self, remote: RemoteFile) -> Path:
        """Return a verified local copy of ``remote``, downloading if needed."""
        cached = self.cache_dir / remote.name
        if git_blob_sha_of(cached) == remote.sha:
            return cached
        self.source.download(remote, cached)
        return cached

    def apply(
        self,
        games: list[GameSdkState],
        *,
        close_running: bool = False,
        on_progress: Callable[[str], None] | None = None,
        force_all: bool = False,
    ) -> list[GameApplyResult]:
        """Install missing / outdated files into the given installations.

        Downloads each needed file once (verified) into the cache, then copies
        it into every game's plugin folder. If a game is running it is skipped
        unless ``close_running`` is True, in which case it is closed first
        (the caller must have obtained the user's consent).

        With ``force_all`` every managed file the repository publishes is
        (re)downloaded and overwritten regardless of the file's current state -
        used by the "reinstall SDK" action.
        """
        try:
            remote_files = self.source.list_files()
        except SdkSourceError as exc:
            log.error("cannot install SDK, source unreachable: %s", exc)
            return [
                GameApplyResult(g.game_type, g.game_path, errors=[("source", str(exc))])
                for g in games
            ]

        results: list[GameApplyResult] = []
        for game in games:
            result = GameApplyResult(game.game_type, game.game_path)

            if is_game_running(game.game_type):
                if not close_running:
                    result.skipped_running = True
                    results.append(result)
                    continue
                if on_progress:
                    on_progress(f"Closing {game.game_type.upper()}...")
                if not close_game(game.game_type):
                    result.errors.append((game.game_type, "could not close the running game"))
                    results.append(result)
                    continue

            wanted = self._files_to_install(game, remote_files, force_all=force_all)
            for name in wanted:
                remote = remote_files.get(name)
                if remote is None:
                    continue  # not published for this version; nothing to do
                try:
                    if on_progress:
                        on_progress(f"Installing {name} for {game.game_type.upper()}...")
                    source_file = self._ensure_cached(remote)
                    self._copy_into_place(source_file, game.plugins_dir / name)
                    result.installed.append(name)
                except (SdkSourceError, OSError) as exc:
                    log.error("failed to install %s for %s: %s", name, game.game_type, exc)
                    result.errors.append((name, str(exc)))

            results.append(result)

        if results and all(r.success for r in results):
            self._mark_checked()
        return results

    def _files_to_install(
        self,
        game: GameSdkState,
        remote_files: dict[str, RemoteFile],
        *,
        force_all: bool = False,
    ) -> list[str]:
        """Filenames to (re)install for one game: stale/missing tracked files,
        plus any courtesy file that is absent or differs from the source. With
        ``force_all``, every published managed file is returned."""
        if force_all:
            return [n for n in (*self.tracked_files, *COURTESY_FILES) if n in remote_files]
        wanted = [f.name for f in game.files if not f.installed or not f.up_to_date]
        for name in COURTESY_FILES:
            remote = remote_files.get(name)
            if remote is None:
                continue
            if git_blob_sha_of(game.plugins_dir / name) != remote.sha:
                wanted.append(name)
        return wanted

    @staticmethod
    def _copy_into_place(source: Path, dest: Path) -> None:
        """Atomically replace ``dest`` with the verified cached file."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        shutil.copyfile(source, tmp)
        os.replace(tmp, dest)


# Module-level convenience API


_default_manager: SdkManager | None = None
_default_lock = threading.Lock()


def get_manager() -> SdkManager:
    """Shared :class:`SdkManager` instance."""
    global _default_manager
    with _default_lock:
        if _default_manager is None:
            _default_manager = SdkManager()
        return _default_manager


def check_sdk(*, force_remote: bool | None = None) -> SdkCheckResult:
    """One-shot SDK check using the shared manager."""
    return get_manager().check(force_remote=force_remote)


def start_boot_check(
    on_result: Callable[[SdkCheckResult], None],
    *,
    force_remote: bool | None = None,
) -> threading.Thread:
    """Run :func:`check_sdk` on a daemon thread and hand the result back.

    Keeps the (possible) network call off the boot / Qt main thread. Any error
    is logged and swallowed so a failed check can never affect startup; in that
    case ``on_result`` is not called.
    """

    def _run() -> None:
        try:
            result = get_manager().check(force_remote=force_remote)
        except Exception:  # never let the boot check take the app down
            log.exception("SDK boot check failed")
            return
        try:
            on_result(result)
        except Exception:
            log.exception("SDK boot-check callback failed")

    thread = threading.Thread(target=_run, name="sdk_boot_check", daemon=True)
    thread.start()
    return thread


def start_reinstall(
    on_result: Callable[[list[GameApplyResult]], None],
    *,
    close_running: bool = True,
) -> threading.Thread:
    """Force-reinstall every managed file on a daemon thread, then hand the
    per-game results back. Keeps the download / file I/O off the caller's
    thread; ``on_result`` runs on the worker thread, so a Qt caller must marshal
    any UI work (or use the thread-safe logging popup). On failure ``on_result``
    receives an empty list.
    """

    def _run() -> None:
        try:
            results = get_manager().reinstall_all(close_running=close_running)
        except Exception:
            log.exception("SDK reinstall failed")
            results = []
        try:
            on_result(results)
        except Exception:
            log.exception("SDK reinstall callback failed")

    thread = threading.Thread(target=_run, name="sdk_reinstall", daemon=True)
    thread.start()
    return thread
