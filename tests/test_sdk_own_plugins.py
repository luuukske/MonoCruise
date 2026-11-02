"""MonoCruise's own bundled plugins, installed beside ETS2LA's without the network.

See "MonoCruise's own plugins" in core/sdk_installer/README.md. Nothing here touches
the network or a real game folder.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.radar import tmp_state
from core.settings import Settings
from core.sdk_installer import manager as mgr
from core.sdk_installer import overrides as ov_mod
from core.sdk_installer import own as own_mod
from core.sdk_installer.remote import git_blob_sha, git_blob_sha_of
from tests.test_sdk_version_match import _FakeSource, _plugins, sdk  # noqa: F401

_NAME = "monocruise_tmp.dll"
_OURS = b"monocruise_tmp.dll test build"
_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def own(tmp_path, monkeypatch):
    bundle = tmp_path / "own"
    bundle.mkdir()
    (bundle / _NAME).write_bytes(_OURS)
    item = own_mod.OwnPlugin(_NAME, git_blob_sha(_OURS), bundle)
    monkeypatch.setattr(own_mod, "OWN_PLUGINS", (item,))
    return item


_FFB = "monocruise_ffb.dll"
_FFB_BYTES = b"monocruise_ffb.dll test build"


@pytest.fixture
def opt_in(tmp_path, monkeypatch):
    """Our plugin plus one wanted only while ``opt_in["on"]``, like the FFB plugin and the lane settings."""
    bundle = tmp_path / "own"
    bundle.mkdir()
    (bundle / _NAME).write_bytes(_OURS)
    (bundle / _FFB).write_bytes(_FFB_BYTES)
    want = {"on": False}
    items = (
        own_mod.OwnPlugin(_NAME, git_blob_sha(_OURS), bundle),
        own_mod.OwnPlugin(_FFB, git_blob_sha(_FFB_BYTES), bundle, wanted=lambda: want["on"]),
    )
    monkeypatch.setattr(own_mod, "OWN_PLUGINS", items)
    return want


def _ours(game: Path) -> bytes:
    return (_plugins(game) / _NAME).read_bytes()


def test_every_game_gets_our_plugin_beside_the_upstream_set(sdk, own):
    ets2 = sdk.add_game("ets2", "1.60")
    ats = sdk.add_game("ats", "1.59")

    results = sdk.apply(sdk.check().games_needing_action)

    assert all(r.success for r in results)
    assert _ours(ets2) == _OURS and _ours(ats) == _OURS
    assert (_plugins(ats) / "ets2la_plugin.dll").read_bytes() == b"ets2la_plugin.dll@1.59"
    assert not sdk.check().needs_action


@pytest.mark.parametrize("damage", ["delete", "stale"])
def test_a_missing_or_stale_copy_is_repaired_offline(sdk, own, damage):
    game = sdk.add_game("ets2", "1.60")
    sdk.apply(sdk.check().games_needing_action)
    if damage == "delete":
        (_plugins(game) / _NAME).unlink()
    else:
        (_plugins(game) / _NAME).write_bytes(b"an older monocruise_tmp.dll")
    _FakeSource.offline = True

    check = sdk.check()
    assert not check.consulted_remote
    assert [*check.games[0].missing, *check.games[0].outdated] == [_NAME]

    results = sdk.apply(check.games_needing_action)
    assert results[0].success and results[0].installed == [_NAME]
    assert _ours(game) == _OURS


def test_a_running_game_keeps_its_loaded_copy_until_restart(sdk, own, monkeypatch):
    game = sdk.add_game("ets2", "1.60")
    sdk.apply(sdk.check().games_needing_action)
    (_plugins(game) / _NAME).write_bytes(b"loaded by the game")
    monkeypatch.setattr(mgr, "is_game_running", lambda game_type: True)

    result = sdk.apply(sdk.check().games_needing_action, allow_running_missing=True)[0]

    assert result.deferred_running == [_NAME]
    assert _ours(game) == b"loaded by the game"


def test_a_damaged_bundle_is_never_installed(sdk, own):
    game = sdk.add_game("ets2", "1.60")
    own.path.write_bytes(b"truncated")

    result = sdk.apply(sdk.check().games_needing_action)[0]

    assert [name for name, _ in result.errors] == [_NAME]
    assert not (_plugins(game) / _NAME).exists()
    assert (_plugins(game) / "ets2la_plugin.dll").exists()


def test_reinstall_rewrites_our_plugin(sdk, own):
    game = sdk.add_game("ets2", "1.60")
    sdk.apply(sdk.check().games_needing_action)
    (_plugins(game) / _NAME).write_bytes(b"hand edited")

    results = sdk.apply(sdk.locate_games(), force_all=True)

    assert results[0].success and _NAME in results[0].installed
    assert _ours(game) == _OURS


def test_an_unsupported_game_version_gets_none_of_ours(sdk, own):
    game = sdk.add_game("ets2", "1.61")

    sdk.apply(sdk.check().games_needing_action)

    assert not (_plugins(game) / _NAME).exists()


def test_a_retired_override_goes_back_to_stock_offline(sdk, own, monkeypatch):
    game = sdk.add_game("ets2", "1.60")
    sdk.apply(sdk.check().games_needing_action)
    old = b"ets2la_plugin.dll monocruise ncz build"
    (_plugins(game) / "ets2la_plugin.dll").write_bytes(old)
    monkeypatch.setattr(ov_mod, "RETIRED", {("1.60", "ets2la_plugin.dll"): frozenset({git_blob_sha(old)})})
    _FakeSource.offline = True

    check = sdk.check()
    assert check.games[0].outdated == ["ets2la_plugin.dll"]

    results = sdk.apply(check.games_needing_action)
    assert results[0].success
    assert (_plugins(game) / "ets2la_plugin.dll").read_bytes() == b"ets2la_plugin.dll@1.60"


def test_every_shipped_own_plugin_matches_its_bundle():
    for item in own_mod.OWN_PLUGINS:
        assert git_blob_sha_of(item.path) == item.sha, f"rebuild changed {item.name}: update its sha"
        assert (item.path.parent / "LICENSES.txt").exists()


def test_the_plugin_writes_the_layout_the_radar_reads():
    src = (_REPO / "tmp_plugin" / "src" / "main.cpp").read_text(encoding="utf-8")
    name = re.search(r'kStateName\[\]\s*=\s*L"([^"]+)"', src).group(1).replace("\\\\", "\\")
    assert name == tmp_state._STATE_TAG
    assert f"sizeof( StateData ) == {tmp_state._STATE_SIZE}" in src
    assert re.search(r"kStateVersion\s*=\s*(\d+)", src).group(1) == str(tmp_state._STATE_VERSION)


def test_an_opt_in_plugin_is_installed_only_while_wanted(sdk, opt_in):
    game = sdk.add_game("ets2", "1.60")
    sdk.apply(sdk.check().games_needing_action)
    assert _ours(game) == _OURS and not (_plugins(game) / _FFB).exists()
    _FakeSource.offline = True
    assert not sdk.check().needs_action, "nobody asked for it: nothing to do"

    opt_in["on"] = True
    check = sdk.check()
    assert not check.consulted_remote and check.games[0].missing == [_FFB]
    assert sdk.apply(check.games_needing_action)[0].installed == [_FFB]
    assert (_plugins(game) / _FFB).read_bytes() == _FFB_BYTES


def test_an_opt_in_plugin_left_installed_is_kept_current(sdk, opt_in):
    game = sdk.add_game("ets2", "1.60")
    opt_in["on"] = True
    sdk.apply(sdk.check().games_needing_action)
    opt_in["on"] = False
    (_plugins(game) / _FFB).write_bytes(b"an older monocruise_ffb.dll")

    check = sdk.check()
    assert check.games[0].outdated == [_FFB], "switched off but still in the game folder"
    sdk.apply(check.games_needing_action)
    assert (_plugins(game) / _FFB).read_bytes() == _FFB_BYTES


def test_reinstall_leaves_out_an_opt_in_plugin_nobody_wants(sdk, opt_in):
    game = sdk.add_game("ets2", "1.60")

    results = sdk.apply(sdk.locate_games(), force_all=True)

    assert results[0].success and results[0].installed and _FFB not in results[0].installed
    assert not (_plugins(game) / _FFB).exists()


def test_switching_it_on_installs_it_at_once_beside_a_running_game(sdk, opt_in, monkeypatch):
    game = sdk.add_game("ets2", "1.60")
    sdk.apply(sdk.check().games_needing_action)
    monkeypatch.setattr(mgr, "_default_manager", sdk)
    monkeypatch.setattr(mgr, "is_game_running", lambda game_type: True)
    opt_in["on"] = True
    done: list = []

    mgr.start_own_install(_FFB, done.append).join(5)
    assert [r.installed for r in done[0]] == [[_FFB]], "absent from the game, so it installs for the next start"
    assert (_plugins(game) / _FFB).read_bytes() == _FFB_BYTES

    done.clear()
    mgr.start_own_install(_FFB, done.append).join(5)
    assert done == [[]], "already there and current: nothing to do"


@pytest.mark.parametrize("key", ["ldw_enabled", "lka_enabled", "la_enabled"])
def test_the_shipped_ffb_plugin_is_wanted_by_any_lane_feature(monkeypatch, key):
    ffb, s = own_mod.own_plugin(_FFB), Settings.instance()
    for name in ("ldw_enabled", "lka_enabled", "la_enabled"):
        monkeypatch.setattr(s, name, False)
    assert ffb.label == "Lane departure plugin" and not ffb.wanted()
    monkeypatch.setattr(s, key, True)
    assert ffb.wanted()
