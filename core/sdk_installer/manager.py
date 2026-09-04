"""ETS2/ATS SDK install backend (no UI). See README.md in this package."""

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
    detect_game_version,
    find_game_installations,
    get_plugins_dir,
    is_game_running,
    is_steam_installed,
)
from .remote import (
    RemoteFile,
    SdkSource,
    SdkSourceError,
    SdkVersionUnsupported,
    git_blob_sha_of,
    unsupported_reason,
)

log = logging.getLogger("sdk")

# Only used when an install's own version cannot be read. The version that
# actually decides which plugin to fetch comes from the game executable.
DEFAULT_GAME_VERSION = "1.60"

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

# Plugins older MonoCruise versions installed, superseded by DLL_FILES. A
# leftover copy loads alongside the current one and fights it for the game.
LEGACY_FILES: tuple[str, ...] = (
    "input_semantical.dll",
    "ets2_la_plugin.dll",
)
_DISABLED_SUFFIX = ".monocruise-disabled"

_STATE_FILE = "sdk_state.json"
_CACHE_DIRNAME = "sdk_cache"
_MANIFEST_FILE = "manifest.json"


def _marker_name(version: str) -> str:
    """ETS2LA version-marker filename (the version lives in the name)."""
    return f"ets2la_{version}"


def _find_conflicting(plugins_dir: Path) -> list[str]:
    """Superseded plugin files still sitting in a game's plugins folder."""
    return [name for name in LEGACY_FILES if (plugins_dir / name).exists()]


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
    # Superseded plugins found in plugins_dir; detected locally, no remote needed.
    conflicting: list[str] = field(default_factory=list)
    # Engine version this install reports, and whether it was actually read
    # (False means DEFAULT_GAME_VERSION is a guess).
    game_version: str = DEFAULT_GAME_VERSION
    version_detected: bool = False
    # No plugin published upstream for game_version, in either direction.
    version_unsupported: bool = False
    unsupported_reason: str = ""
    # A verified local copy of that version's file set exists, so an
    # unsupported version can still be installed offline.
    cache_available: bool = False

    @property
    def missing(self) -> list[str]:
        return [f.name for f in self.files if not f.installed]

    @property
    def outdated(self) -> list[str]:
        return [f.name for f in self.files if f.installed and not f.up_to_date]

    @property
    def needs_action(self) -> bool:
        return bool(self.missing or self.outdated or self.conflicting)

    @property
    def installable(self) -> bool:
        """False when nothing, remote or cached, can supply this version."""
        return not self.version_unsupported or self.cache_available


@dataclass
class SdkCheckResult:
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

    @property
    def unsupported_games(self) -> list[GameSdkState]:
        """Installs whose game version has no plugin and no cached fallback."""
        return [g for g in self.games if g.version_unsupported and not g.cache_available]

    @property
    def version_unsupported(self) -> bool:
        return bool(self.unsupported_games)


@dataclass
class GameApplyResult:
    game_type: str
    game_path: Path
    installed: list[str] = field(default_factory=list)
    # Superseded plugins renamed aside so the game stops loading them.
    disabled: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    skipped_running: bool = False
    # Present-but-outdated DLLs while game runs (see deferred_running vs skipped_running).
    deferred_running: list[str] = field(default_factory=list)
    # True when this install's game version has no SDK folder upstream.
    unsupported: bool = False
    game_version: str = ""
    unsupported_reason: str = ""
    # Installed from the local cache because the source had nothing to offer.
    from_cache: bool = False

    @property
    def success(self) -> bool:
        return (
            not self.errors
            and not self.unsupported
            and not self.skipped_running
            and not self.deferred_running
        )


@dataclass(frozen=True)
class _LocalScan:
    """One install as seen offline, before any remote call."""

    game_type: str
    game_path: Path
    plugins_dir: Path
    version: str
    version_detected: bool
    present: dict[str, bool]


class SdkManager:
    """Stateful helper that checks and installs the SDK for both games."""

    def __init__(
        self,
        *,
        default_version: str = DEFAULT_GAME_VERSION,
        data_dir: Path | None = None,
    ):
        self.default_version = default_version

        base = data_dir if data_dir is not None else CONFIG_PATH.parent
        self._cache_root = base / _CACHE_DIRNAME
        self._state_path = base / _STATE_FILE
        self._sources: dict[str, SdkSource] = {}
        self._published: list[str] | None = None

    # Per-version helpers (one game version per install, so nothing is global)

    def tracked_files(self, version: str) -> tuple[str, ...]:
        """Files that must be present for one game version."""
        return DLL_FILES + (_marker_name(version),)

    def cache_dir(self, version: str) -> Path:
        return self._cache_root / version

    def _source(self, version: str) -> SdkSource:
        source = self._sources.get(version)
        if source is None:
            source = SdkSource(version)
            self._sources[version] = source
        return source

    def _unsupported_reason(self, version: str) -> str:
        """Why this version has no plugin, said in the right direction."""
        if self._published is None:
            self._published = self._source(version).list_versions()
        return unsupported_reason(version, self._published)

    def _game_version(self, game_type: str, game_path: Path) -> tuple[str, bool]:
        detected = detect_game_version(game_path, game_type)
        if detected:
            return detected, True
        log.warning(
            "could not read the %s game version, assuming %s", game_type, self.default_version
        )
        return self.default_version, False

    # Local cache (last known-good set, keyed by game version)

    def _manifest_path(self, version: str) -> Path:
        return self.cache_dir(version) / _MANIFEST_FILE

    def _read_manifest(self, version: str) -> dict[str, str]:
        try:
            data = json.loads(self._manifest_path(version).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        files = data.get("files") if isinstance(data, dict) else None
        if not isinstance(files, dict):
            return {}
        return {k: v for k, v in files.items() if isinstance(k, str) and isinstance(v, str)}

    def _record_cached(self, version: str, remote: RemoteFile) -> None:
        """Remember the SHA of a verified cached file so it can be reused offline."""
        files = self._read_manifest(version)
        if files.get(remote.name) == remote.sha:
            return
        files[remote.name] = remote.sha
        try:
            path = self._manifest_path(version)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"files": files}), encoding="utf-8")
        except OSError:
            log.debug("could not write the SDK cache manifest", exc_info=True)

    def cached_listing(self, version: str) -> dict[str, RemoteFile]:
        """Cached files that still hash to what was verified when they landed."""
        listing: dict[str, RemoteFile] = {}
        for name, sha in self._read_manifest(version).items():
            path = self.cache_dir(version) / name
            if git_blob_sha_of(path) != sha:
                continue
            # No URL: a cache entry is usable as-is or not at all, never fetched.
            listing[name] = RemoteFile(name, sha, 0, "")
        return listing

    def cache_is_complete(self, version: str) -> bool:
        """True when the cache alone can produce a working install."""
        listing = self.cached_listing(version)
        return all(name in listing for name in self.tracked_files(version))

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
        """Read-only install scan; remote only when missing or FORCE_REFETCH."""
        if not is_steam_installed():
            return SdkCheckResult(False, False, None)

        located: list[tuple[str, Path]] = [
            (game_type, path)
            for game_type in GAME_TYPES
            for path in find_game_installations(game_type)
        ]
        if not located:
            return SdkCheckResult(True, False, None)

        # Local presence first (fast, offline). The version marker carries the
        # game version, so a game update alone already shows up as missing.
        scans: list[_LocalScan] = []
        any_missing = False
        for game_type, game_path in located:
            plugins = get_plugins_dir(game_path)
            version, detected = self._game_version(game_type, game_path)
            present = {n: (plugins / n).exists() for n in self.tracked_files(version)}
            any_missing = any_missing or not all(present.values())
            scans.append(_LocalScan(game_type, game_path, plugins, version, detected, present))

        forced = FORCE_REFETCH and self._read_last_checked() != __version__
        consult = force_remote if force_remote is not None else (any_missing or forced)

        listings: dict[str, dict[str, RemoteFile]] = {}
        unsupported: dict[str, str] = {}
        remote_error: str | None = None
        if consult:
            for version in dict.fromkeys(scan.version for scan in scans):
                try:
                    listings[version] = self._source(version).list_files()
                except SdkVersionUnsupported as exc:
                    unsupported[version] = self._unsupported_reason(version)
                    log.warning("SDK check: %s", exc)
                except SdkSourceError as exc:
                    remote_error = str(exc)
                    log.warning("SDK check could not reach the ETS2LA source: %s", exc)

        games: list[GameSdkState] = []
        for scan in scans:
            listing = listings.get(scan.version)
            file_states: list[ManagedFileState] = []
            for name in self.tracked_files(scan.version):
                installed = scan.present[name]
                up_to_date = True
                if installed and listing and name in listing:
                    up_to_date = git_blob_sha_of(scan.plugins_dir / name) == listing[name].sha
                file_states.append(ManagedFileState(name, installed, up_to_date))
            is_unsupported = scan.version in unsupported
            games.append(
                GameSdkState(
                    game_type=scan.game_type,
                    game_path=scan.game_path,
                    plugins_dir=scan.plugins_dir,
                    running=is_game_running(scan.game_type),
                    files=file_states,
                    conflicting=_find_conflicting(scan.plugins_dir),
                    game_version=scan.version,
                    version_detected=scan.version_detected,
                    version_unsupported=is_unsupported,
                    unsupported_reason=unsupported.get(scan.version, ""),
                    cache_available=is_unsupported and self.cache_is_complete(scan.version),
                )
            )

        result = SdkCheckResult(
            steam_installed=True,
            consulted_remote=bool(listings),
            remote_error=remote_error,
            games=games,
        )

        # Nothing to do and we actually looked: record the version so a forced
        # refetch does not repeat on every boot.
        if result.consulted_remote and not result.needs_action:
            self._mark_checked()
        return result

    def locate_games(self) -> list[GameSdkState]:
        """All local installs without remote SHA checks (force reinstall path)."""
        states: list[GameSdkState] = []
        for game_type in GAME_TYPES:
            for game_path in find_game_installations(game_type):
                plugins = get_plugins_dir(game_path)
                version, detected = self._game_version(game_type, game_path)
                files = [
                    ManagedFileState(name, (plugins / name).exists(), True)
                    for name in self.tracked_files(version)
                ]
                states.append(
                    GameSdkState(
                        game_type=game_type,
                        game_path=game_path,
                        plugins_dir=plugins,
                        running=is_game_running(game_type),
                        files=files,
                        conflicting=_find_conflicting(plugins),
                        game_version=version,
                        version_detected=detected,
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

    def _ensure_cached(self, version: str, remote: RemoteFile) -> Path:
        """Return a verified local copy of ``remote``, downloading if needed."""
        cached = self.cache_dir(version) / remote.name
        if git_blob_sha_of(cached) != remote.sha:
            if not remote.download_url:
                raise SdkSourceError(f"{remote.name} is not in the local cache")
            self._source(version).download(remote, cached)
        self._record_cached(version, remote)
        return cached

    def _install_listing(
        self,
        game: GameSdkState,
        result: GameApplyResult,
        *,
        force_all: bool,
    ) -> dict[str, RemoteFile] | None:
        """Files to install from, for this install's own game version.

        Falls back to the verified cache when the source has nothing for that
        version, so a pruned upstream folder cannot strand a working install.
        """
        if not (force_all or game.missing or game.outdated):
            return {}  # only a legacy plugin to disable; that needs no network

        version = game.game_version
        try:
            return self._source(version).list_files()
        except SdkSourceError as exc:
            unsupported = isinstance(exc, SdkVersionUnsupported)
            cached = self.cached_listing(version)
            if all(n in cached for n in self.tracked_files(version)):
                log.info("installing the cached plugin set for game version %s: %s", version, exc)
                result.from_cache = True
                return cached
            if unsupported:
                # Not an error entry: a legacy plugin this pass disabled is
                # still worth reporting alongside the version warning.
                result.unsupported = True
                result.unsupported_reason = self._unsupported_reason(version)
                log.warning("cannot install the SDK: %s", result.unsupported_reason)
            else:
                log.error("cannot install SDK, source unreachable: %s", exc)
                result.errors.append(("source", str(exc)))
            return None

    def apply(
        self,
        games: list[GameSdkState],
        *,
        close_running: bool = False,
        on_progress: Callable[[str], None] | None = None,
        force_all: bool = False,
        allow_running_missing: bool = False,
    ) -> list[GameApplyResult]:
        """Copy verified files into plugin dirs; see README for running-game modes."""
        results: list[GameApplyResult] = []
        for game in games:
            result = GameApplyResult(
                game.game_type, game.game_path, game_version=game.game_version
            )

            restrict_to_missing = False
            if is_game_running(game.game_type):
                if close_running:
                    if on_progress:
                        on_progress(f"Closing {game.game_type.upper()}...")
                    if not close_game(game.game_type):
                        result.errors.append((game.game_type, "could not close the running game"))
                        results.append(result)
                        continue
                elif allow_running_missing:
                    restrict_to_missing = True
                else:
                    result.skipped_running = True
                    results.append(result)
                    continue

            # Renaming works even on a DLL the game holds loaded; it just takes
            # effect at the next game start, so no running-game special case.
            for name in _find_conflicting(game.plugins_dir):
                try:
                    if on_progress:
                        on_progress(f"Disabling {name} for {game.game_type.upper()}...")
                    self._disable_legacy(game.plugins_dir / name)
                    result.disabled.append(name)
                    log.info("disabled superseded plugin %s for %s", name, game.game_type)
                except OSError as exc:
                    log.error("could not disable %s for %s: %s", name, game.game_type, exc)
                    result.errors.append((name, str(exc)))

            # Never install another version's plugin: it loads, resolves
            # nothing, and the failure is silent.
            available = self._install_listing(game, result, force_all=force_all)
            if available is None:
                results.append(result)
                continue

            wanted = self._files_to_install(game, available, force_all=force_all)
            if restrict_to_missing:
                # Loaded DLLs stay deferred; absent files install for next game start.
                result.deferred_running = [
                    n for n in wanted
                    if n.endswith(".dll") and (game.plugins_dir / n).exists()
                ]
                wanted = [n for n in wanted if n not in result.deferred_running]
            for name in wanted:
                remote = available.get(name)
                if remote is None:
                    continue  # not published for this version; nothing to do
                try:
                    if on_progress:
                        on_progress(f"Installing {name} for {game.game_type.upper()}...")
                    source_file = self._ensure_cached(game.game_version, remote)
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
        """Stale/missing tracked files plus courtesy files; force_all returns all remote."""
        if force_all:
            tracked = self.tracked_files(game.game_version)
            return [n for n in (*tracked, *COURTESY_FILES) if n in remote_files]
        wanted = [f.name for f in game.files if not f.installed or not f.up_to_date]
        for name in COURTESY_FILES:
            remote = remote_files.get(name)
            if remote is None:
                continue
            if git_blob_sha_of(game.plugins_dir / name) != remote.sha:
                wanted.append(name)
        return wanted

    @staticmethod
    def _disable_legacy(path: Path) -> None:
        """Rename a superseded plugin aside. Never deletes; see the README."""
        os.replace(path, path.with_name(path.name + _DISABLED_SUFFIX))

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
    """Daemon check_sdk; errors swallowed; on_result not called on failure."""

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
    """Daemon reinstall_all; on_result([]) on failure; UI must marshal from worker."""

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
