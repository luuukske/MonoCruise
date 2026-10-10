"""Bundled plugin overrides: installed over the stock build only, upstream updates win.

See the "Plugin overrides" section of core/sdk_installer/README.md. Nothing here
touches the network or a real game folder.
"""
from __future__ import annotations

import pytest

from core.sdk_installer import overrides as ov_mod
from core.sdk_installer.remote import git_blob_sha, git_blob_sha_of
from tests.test_sdk_version_match import _FakeSource, _plugins, sdk  # noqa: F401

_OURS = b"ets2la_plugin.dll monocruise ncz build"
_STOCK_160 = b"ets2la_plugin.dll@1.60"


@pytest.fixture
def override(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    (bundle / "1.60").mkdir(parents=True)
    (bundle / "1.60" / "ets2la_plugin.dll").write_bytes(_OURS)
    item = ov_mod.PluginOverride(
        version="1.60",
        name="ets2la_plugin.dll",
        sha=git_blob_sha(_OURS),
        replaces=frozenset({git_blob_sha(_STOCK_160)}),
        bundle_dir=bundle,
    )
    monkeypatch.setattr(ov_mod, "OVERRIDES", (item,))
    return item


def _dll(game):
    return (_plugins(game) / "ets2la_plugin.dll").read_bytes()


def test_a_fresh_install_gets_the_override_and_upstream_for_the_rest(sdk, override):
    game = sdk.add_game("ets2", "1.60")

    results = sdk.apply(sdk.check().games_needing_action)

    assert results[0].success
    assert _dll(game) == _OURS
    assert (_plugins(game) / "scs-telemetry.dll").read_bytes() == b"scs-telemetry.dll@1.60"
    assert (_plugins(game) / "ets2la_1.60").exists()
    assert not sdk.check().needs_action


def test_the_stock_build_is_replaced_on_an_offline_boot(sdk, override, monkeypatch):
    game = sdk.add_game("ets2", "1.60")
    monkeypatch.setattr(ov_mod, "OVERRIDES", ())
    sdk.apply(sdk.check().games_needing_action)
    assert _dll(game) == _STOCK_160

    monkeypatch.setattr(ov_mod, "OVERRIDES", (override,))
    _FakeSource.offline = True
    check = sdk.check()
    assert not check.consulted_remote
    assert check.games[0].outdated == ["ets2la_plugin.dll"]

    results = sdk.apply(check.games_needing_action)

    assert results[0].success and results[0].installed == ["ets2la_plugin.dll"]
    assert _dll(game) == _OURS


def test_an_upstream_update_wins_over_the_override(sdk, override):
    game = sdk.add_game("ets2", "1.60")
    sdk.apply(sdk.check().games_needing_action)
    assert _dll(game) == _OURS

    _FakeSource.payloads["1.60"]["ets2la_plugin.dll"] = b"ets2la_plugin.dll@1.60 with tmp sdk"
    check = sdk.check(force_remote=True)
    assert check.games[0].outdated == ["ets2la_plugin.dll"]

    sdk.apply(check.games_needing_action)
    assert _dll(game) == b"ets2la_plugin.dll@1.60 with tmp sdk"


def test_reinstall_keeps_the_override_while_upstream_is_stock(sdk, override):
    game = sdk.add_game("ets2", "1.60")
    sdk.apply(sdk.check().games_needing_action)

    results = sdk.apply(sdk.locate_games(), force_all=True)

    assert results[0].success
    assert _dll(game) == _OURS


def test_a_build_nobody_replaces_is_left_alone(sdk, override):
    game = sdk.add_game("ets2", "1.60")
    sdk.apply(sdk.check().games_needing_action)
    (_plugins(game) / "ets2la_plugin.dll").write_bytes(b"someone's dev build")

    assert not sdk.check().needs_action


def test_a_damaged_bundle_is_never_installed(sdk, override, monkeypatch):
    game = sdk.add_game("ets2", "1.60")
    monkeypatch.setattr(ov_mod, "OVERRIDES", ())
    sdk.apply(sdk.check().games_needing_action)
    monkeypatch.setattr(ov_mod, "OVERRIDES", (override,))
    override.path.write_bytes(b"truncated")

    results = sdk.apply(sdk.check().games_needing_action)

    assert [name for name, _ in results[0].errors] == ["ets2la_plugin.dll"]
    assert _dll(game) == _STOCK_160


def test_other_game_versions_are_untouched(sdk, override):
    ats = sdk.add_game("ats", "1.59")

    sdk.apply(sdk.check().games_needing_action)

    assert _dll(ats) == b"ets2la_plugin.dll@1.59"


def test_every_shipped_override_matches_its_bundle():
    for item in ov_mod.OVERRIDES:
        assert git_blob_sha_of(item.path) == item.sha, f"rebuild changed {item.path.name}: update its sha"
        assert item.sha not in item.replaces
        assert (item.path.parent / "LICENSES.txt").exists()


def test_results_say_which_files_came_from_an_override(sdk, override):
    sdk.add_game("ets2", "1.60")

    result = sdk.apply(sdk.check().games_needing_action)[0]

    assert result.overrides == ["ets2la_plugin.dll"]
    assert set(result.installed) > set(result.overrides)
