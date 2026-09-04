"""The SDK installer must fetch the plugin for the game version actually installed.

Both directions of mismatch are covered: a game newer than anything published
(the 1.61 open beta, where the old code installed the 1.60 plugin and every AI
vehicle silently disappeared), and a game older than anything still published
(upstream prunes old SDK folders), where the local cache has to carry the
install. Nothing here touches the network or a real game folder.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.sdk_installer import manager as mgr
from core.sdk_installer.game_paths import detect_game_version
from core.sdk_installer.remote import (
    RemoteFile,
    SdkSourceError,
    SdkVersionUnsupported,
    git_blob_sha,
    unsupported_reason,
)

PUBLISHED = ("1.59", "1.60")


def _payload(version: str) -> dict[str, bytes]:
    """The upstream file set for one game version, contents tagged by version."""
    files = {name: f"{name}@{version}".encode() for name in mgr.DLL_FILES}
    files[f"ets2la_{version}"] = f"marker@{version}".encode()
    files["sources.txt"] = f"sources@{version}".encode()
    return files


class _FakeSource:
    """Stands in for SdkSource: serves payloads, 404s on anything else."""

    payloads: dict[str, dict[str, bytes]] = {}
    published: tuple[str, ...] = PUBLISHED
    offline: bool = False

    def __init__(self, version: str):
        self.version = version

    def list_files(self) -> dict[str, RemoteFile]:
        if self.offline:
            raise SdkSourceError("cannot reach GitHub")
        files = self.payloads.get(self.version)
        if files is None:
            raise SdkVersionUnsupported(
                f"no game plugin is published for game version {self.version}"
            )
        return {
            name: RemoteFile(name, git_blob_sha(data), len(data), f"test://{self.version}/{name}")
            for name, data in files.items()
        }

    def list_versions(self) -> list[str]:
        return list(self.published)

    def download(self, remote: RemoteFile, dest: Path) -> None:
        data = self.payloads[self.version][remote.name]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    """A manager wired to fake installs, a fake source, and a scratch cache."""
    installs: dict[str, Path] = {}
    versions: dict[str, str | None] = {}

    def _add_game(game_type: str, game_version: str | None) -> Path:
        root = tmp_path / "games" / game_type
        (root / "bin" / "win_x64" / "plugins").mkdir(parents=True, exist_ok=True)
        installs[game_type] = root
        versions[game_type] = game_version
        return root

    monkeypatch.setattr(mgr, "is_steam_installed", lambda: True)
    monkeypatch.setattr(mgr, "is_game_running", lambda game_type: False)
    monkeypatch.setattr(
        mgr, "find_game_installations", lambda game_type: [installs[game_type]]
        if game_type in installs else []
    )
    monkeypatch.setattr(
        mgr, "detect_game_version", lambda game_path, game_type: versions.get(game_type)
    )
    monkeypatch.setattr(mgr, "SdkSource", _FakeSource)
    monkeypatch.setattr(_FakeSource, "payloads", {v: _payload(v) for v in PUBLISHED})
    monkeypatch.setattr(_FakeSource, "published", PUBLISHED)
    monkeypatch.setattr(_FakeSource, "offline", False)

    manager = mgr.SdkManager(data_dir=tmp_path / "data")
    manager.add_game = _add_game  # type: ignore[attr-defined]
    return manager


def _plugins(game_root: Path) -> Path:
    return game_root / "bin" / "win_x64" / "plugins"


def _install_everything(manager) -> list:
    result = manager.check()
    return manager.apply(result.games_needing_action)


def test_installs_the_folder_matching_the_detected_game_version(sdk):
    """A 1.59 install gets the 1.59 plugin, not whatever the default happens to be."""
    game = sdk.add_game("ets2", "1.59")
    assert sdk.default_version != "1.59"

    results = _install_everything(sdk)

    assert [r.success for r in results] == [True]
    assert results[0].game_version == "1.59"
    plugins = _plugins(game)
    assert (plugins / "ets2la_plugin.dll").read_bytes() == b"ets2la_plugin.dll@1.59"
    assert (plugins / "ets2la_1.59").exists()
    assert not (plugins / "ets2la_1.60").exists()


def test_game_newer_than_any_plugin_is_never_given_an_older_one(sdk):
    """The 1.61 open beta case: warn, and install nothing rather than a 1.60 plugin."""
    game = sdk.add_game("ets2", "1.61")

    check = sdk.check()
    assert check.version_unsupported
    state = check.games[0]
    assert state.game_version == "1.61"
    assert state.version_detected
    assert not state.installable
    assert "newer" in state.unsupported_reason

    results = sdk.apply(check.games_needing_action)

    assert results[0].unsupported
    assert results[0].game_version == "1.61"
    assert "newer" in results[0].unsupported_reason
    assert not results[0].installed
    assert list(_plugins(game).iterdir()) == []


def test_one_unsupported_game_does_not_block_the_other(sdk):
    """ETS2 on the open beta must not stop ATS getting its own matching plugin."""
    ets2 = sdk.add_game("ets2", "1.61")
    ats = sdk.add_game("ats", "1.60")

    check = sdk.check()
    results = {r.game_type: r for r in sdk.apply(check.games_needing_action)}

    assert results["ets2"].unsupported
    assert list(_plugins(ets2).iterdir()) == []
    assert results["ats"].success
    assert (_plugins(ats) / "ets2la_plugin.dll").read_bytes() == b"ets2la_plugin.dll@1.60"


def test_pruned_upstream_version_reinstalls_from_the_cache(sdk):
    """Upstream drops 1.59 while a user is still on it; the cached set carries them."""
    game = sdk.add_game("ets2", "1.59")
    _install_everything(sdk)
    assert sdk.cache_is_complete("1.59")

    for path in _plugins(game).iterdir():
        path.unlink()
    _FakeSource.payloads.pop("1.59")
    _FakeSource.published = ("1.60", "1.61")

    check = sdk.check()
    assert check.games[0].version_unsupported
    assert check.games[0].cache_available
    assert check.games[0].installable
    assert not check.unsupported_games  # the cache covers it, so no warning

    results = sdk.apply(check.games_needing_action)

    assert results[0].success
    assert results[0].from_cache
    assert (_plugins(game) / "ets2la_plugin.dll").read_bytes() == b"ets2la_plugin.dll@1.59"
    assert (_plugins(game) / "ets2la_1.59").exists()


def test_cache_fallback_also_covers_an_unreachable_source(sdk):
    """A rate-limited or offline GitHub must not block a repair we can do locally."""
    game = sdk.add_game("ets2", "1.60")
    _install_everything(sdk)
    (_plugins(game) / "ets2la_plugin.dll").unlink()

    _FakeSource.offline = True
    results = sdk.apply(sdk.locate_games(), force_all=True)

    assert results[0].success
    assert results[0].from_cache
    assert (_plugins(game) / "ets2la_plugin.dll").read_bytes() == b"ets2la_plugin.dll@1.60"


def test_unsupported_version_without_a_cache_installs_nothing(sdk):
    """No plugin and no cached copy: report it, never fall back to another version."""
    game = sdk.add_game("ets2", "1.61")
    _FakeSource.offline = True

    results = sdk.apply(sdk.locate_games(), force_all=True)

    assert not results[0].success
    assert not results[0].installed
    assert list(_plugins(game).iterdir()) == []


def test_a_superseded_plugin_is_still_disabled_on_an_unsupported_version(sdk):
    """Disabling a 1.0 leftover is a local repair, and pedal control depends on it."""
    game = sdk.add_game("ets2", "1.61")
    legacy = _plugins(game) / mgr.LEGACY_FILES[0]
    legacy.write_bytes(b"old plugin")

    results = sdk.apply(sdk.check().games_needing_action)

    assert results[0].unsupported
    assert not results[0].success
    assert results[0].disabled == [mgr.LEGACY_FILES[0]]
    assert not results[0].errors  # the version is reported on its own field
    assert not legacy.exists()
    assert (_plugins(game) / f"{mgr.LEGACY_FILES[0]}.monocruise-disabled").exists()


def test_undetectable_version_falls_back_to_the_default(sdk):
    """A build with no readable version resource keeps working, and says so."""
    sdk.add_game("ets2", None)

    check = sdk.check()

    assert check.games[0].game_version == sdk.default_version
    assert not check.games[0].version_detected


def test_game_update_alone_makes_the_install_stale(sdk):
    """The version marker carries the version, so 1.59 files fail a 1.60 check."""
    game = sdk.add_game("ets2", "1.59")
    _install_everything(sdk)
    assert not sdk.check().needs_action

    sdk.add_game("ets2", "1.60")  # same folder, the game updated under it
    check = sdk.check()

    assert check.needs_action
    assert "ets2la_1.60" in check.games[0].missing
    assert "ets2la_plugin.dll" in check.games[0].outdated

    sdk.apply(check.games_needing_action)
    assert (_plugins(game) / "ets2la_plugin.dll").read_bytes() == b"ets2la_plugin.dll@1.60"


def test_unsupported_reason_reads_correctly_in_both_directions():
    assert "newer" in unsupported_reason("1.61", PUBLISHED)
    assert "no longer" in unsupported_reason("1.58", PUBLISHED)
    # Nothing to compare against: stay neutral rather than guess a direction.
    assert "no game plugin is published" in unsupported_reason("1.61", [])


def test_version_detection_is_optional_off_windows(monkeypatch):
    """Non-Windows and unknown game types return None, which the manager handles."""
    monkeypatch.setattr("core.sdk_installer.game_paths._IS_WINDOWS", False)
    assert detect_game_version(Path("anywhere"), "ets2") is None
    assert detect_game_version(Path("anywhere"), "not-a-game") is None
