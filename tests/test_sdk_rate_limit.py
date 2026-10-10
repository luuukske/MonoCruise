"""The game plugin must install when GitHub cannot be asked: offline, or a rate-limited API.

VPN exit addresses share the API's 60 requests per hour with everyone on them. The live
fetch stays first; the plugin set shipped with the build is the fallback. See "Bundled
fallback" in core/sdk_installer/README.md. Nothing here touches the network.
"""
from __future__ import annotations

import json

import pytest
from requests.structures import CaseInsensitiveDict

from core.sdk_installer import bundled as bundled_mod
from core.sdk_installer import overrides as ov_mod
from core.sdk_installer import remote as remote_mod
from core.sdk_installer.remote import (
    RemoteFile,
    SdkSource,
    SdkSourceError,
    git_blob_sha,
    git_blob_sha_of,
)
from tests.test_sdk_version_match import _FakeSource, _payload, _plugins, sdk  # noqa: F401
from tools import fetch_sdk_bundle


class _Response:
    def __init__(self, status: int, *, headers=None, text: str = "", content: bytes = b""):
        self.status_code = status
        self.headers = CaseInsensitiveDict(headers or {})
        self.text = text
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            raise remote_mod.requests.HTTPError(f"HTTP {self.status_code}")


_RATE_LIMITED = _Response(
    403,
    headers={"x-ratelimit-remaining": "0"},
    text='{"message":"API rate limit exceeded for 203.0.113.7."}',
)


def _bundled(version: str) -> dict[str, bytes]:
    """What a build shipped for one version: older than the live set in the fake source."""
    return {name: data + b" (bundled)" for name, data in _payload(version).items()}


def _ship(version: str) -> None:
    """Write one version's set and manifest into the bundle dir, as the build tool does."""
    files = _bundled(version)
    version_dir = bundled_mod.BUNDLE_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (version_dir / name).write_bytes(data)
    manifest = {"files": {name: git_blob_sha(data) for name, data in files.items()}}
    (version_dir / bundled_mod.MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def downloads(monkeypatch):
    """Record every file the installer downloads."""
    names: list[str] = []
    real = _FakeSource.download

    def _recording(self, remote: RemoteFile, dest):
        names.append(remote.name)
        real(self, remote, dest)

    monkeypatch.setattr(_FakeSource, "download", _recording)
    return names


def _dll(game) -> bytes:
    return (_plugins(game) / "ets2la_plugin.dll").read_bytes()


def test_a_reachable_api_installs_the_live_set_over_the_bundle(sdk, downloads):
    game = sdk.add_game("ets2", "1.60")
    _ship("1.60")

    results = sdk.apply(sdk.check().games_needing_action)

    assert results[0].success and not results[0].from_bundle
    assert _dll(game) == b"ets2la_plugin.dll@1.60"
    assert "ets2la_plugin.dll" in downloads


def test_offline_or_rate_limited_installs_the_bundle(sdk, downloads):
    game = sdk.add_game("ets2", "1.60")
    _ship("1.60")
    _FakeSource.offline = True

    check = sdk.check()
    assert check.remote_error
    results = sdk.apply(check.games_needing_action)

    assert results[0].success
    assert results[0].from_bundle
    assert _dll(game) == b"ets2la_plugin.dll@1.60 (bundled)"
    assert (_plugins(game) / "ets2la_1.60").exists()
    assert (_plugins(game) / "sources.txt").exists()
    assert downloads == []


def test_a_complete_cache_wins_over_the_bundle(sdk):
    game = sdk.add_game("ets2", "1.60")
    sdk.apply(sdk.check().games_needing_action)
    (_plugins(game) / "ets2la_plugin.dll").unlink()
    _ship("1.60")
    _FakeSource.offline = True

    results = sdk.apply(sdk.check().games_needing_action)

    assert results[0].success and results[0].from_cache and not results[0].from_bundle
    assert _dll(game) == b"ets2la_plugin.dll@1.60"


def test_a_partial_cache_is_topped_up_from_the_bundle(sdk):
    """Cached files are newer than the build, so they keep priority file by file."""
    game = sdk.add_game("ets2", "1.60")
    sdk.apply(sdk.check().games_needing_action)
    (sdk.cache_dir("1.60") / "scs-telemetry.dll").unlink()
    for path in _plugins(game).iterdir():
        path.unlink()
    _ship("1.60")
    _FakeSource.offline = True

    results = sdk.apply(sdk.check().games_needing_action)

    assert results[0].success and results[0].from_bundle
    assert _dll(game) == b"ets2la_plugin.dll@1.60"
    assert (_plugins(game) / "scs-telemetry.dll").read_bytes() == (
        b"scs-telemetry.dll@1.60 (bundled)"
    )


def test_a_damaged_bundled_file_is_never_installed(sdk):
    game = sdk.add_game("ets2", "1.60")
    _ship("1.60")
    (bundled_mod.BUNDLE_DIR / "1.60" / "ets2la_plugin.dll").write_bytes(b"truncated")
    _FakeSource.offline = True

    results = sdk.apply(sdk.check().games_needing_action)

    assert not results[0].success
    assert not (_plugins(game) / "ets2la_plugin.dll").exists()


def test_a_bundle_without_a_manifest_is_ignored(sdk):
    sdk.add_game("ets2", "1.60")
    _ship("1.60")
    (bundled_mod.BUNDLE_DIR / "1.60" / bundled_mod.MANIFEST_FILE).unlink()
    _FakeSource.offline = True

    assert not sdk.apply(sdk.check().games_needing_action)[0].success


def test_a_pruned_version_installs_from_the_bundle(sdk):
    """Upstream dropped 1.58 (the API 404s) after this build shipped it."""
    game = sdk.add_game("ets2", "1.58")
    _ship("1.58")

    check = sdk.check()
    assert check.games[0].version_unsupported
    assert check.games[0].cache_available
    assert not check.unsupported_games

    results = sdk.apply(check.games_needing_action)

    assert results[0].success and results[0].from_bundle
    assert _dll(game) == b"ets2la_plugin.dll@1.58 (bundled)"


def test_the_bundled_stock_build_still_gives_way_to_its_override(sdk, tmp_path, monkeypatch):
    """Offline, a bundled stock plugin is swapped for an override build."""
    ours = b"monocruise ncz build"
    bundle = tmp_path / "override"
    (bundle / "1.60").mkdir(parents=True)
    (bundle / "1.60" / "ets2la_plugin.dll").write_bytes(ours)
    stock = _bundled("1.60")["ets2la_plugin.dll"]
    item = ov_mod.PluginOverride(
        "1.60", "ets2la_plugin.dll", git_blob_sha(ours),
        frozenset({git_blob_sha(stock)}), bundle_dir=bundle,
    )
    monkeypatch.setattr(ov_mod, "OVERRIDES", (item,))
    game = sdk.add_game("ets2", "1.60")
    _ship("1.60")
    _FakeSource.offline = True

    results = sdk.apply(sdk.check().games_needing_action)

    assert results[0].success and results[0].overrides == ["ets2la_plugin.dll"]
    assert _dll(game) == ours


def test_a_rate_limit_is_named_and_the_address_is_not_logged(monkeypatch):
    monkeypatch.setattr(remote_mod.requests, "get", lambda *a, **k: _RATE_LIMITED)

    with pytest.raises(SdkSourceError) as info:
        SdkSource("1.61").list_files()

    assert "rate limit" in str(info.value)
    assert "203.0.113.7" not in str(info.value)


def test_the_token_only_goes_to_the_api(monkeypatch, tmp_path):
    seen: dict[str, dict] = {}

    def _get(url, headers=None, **_k):
        seen[url] = dict(headers or {})
        if "api.github.com" in url:
            return _Response(200)
        return _Response(200, content=b"x")

    monkeypatch.setattr(remote_mod.requests, "get", _get)
    monkeypatch.setattr(_Response, "json", lambda self: [], raising=False)
    source = SdkSource("1.61", token="secret")
    source.list_files()
    source.download(RemoteFile("a", git_blob_sha(b"x"), 1, "https://raw.example/a"), tmp_path / "a")

    api = [h for url, h in seen.items() if "api.github.com" in url]
    raw = [h for url, h in seen.items() if "api.github.com" not in url]
    assert api and all(h.get("Authorization") == "Bearer secret" for h in api)
    assert raw and all("Authorization" not in h for h in raw)


def test_the_build_tool_writes_a_verified_set_per_version(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_sdk_bundle, "SdkSource", _FakeSource)
    monkeypatch.setattr(_FakeSource, "payloads", {v: _payload(v) for v in ("1.59", "1.60")})
    monkeypatch.setattr(_FakeSource, "published", ("1.58", "1.59", "1.60"))
    monkeypatch.setattr(_FakeSource, "offline", False)
    monkeypatch.setattr(_FakeSource, "__init__", lambda self, version, token=None: setattr(
        self, "version", version))
    out = tmp_path / "bundled"
    (out / "1.57").mkdir(parents=True)  # stale set from an earlier fetch
    (out / "LICENSES.txt").write_text("kept")

    fetched = fetch_sdk_bundle.fetch(out, token=None)

    assert sorted(fetched) == ["1.59", "1.60"]  # 1.58 has no Windows set upstream
    assert not (out / "1.57").exists()
    assert (out / "LICENSES.txt").read_text() == "kept"
    for version in fetched:
        listing = bundled_mod.bundled_listing(version, out)
        assert set(listing) == set(_payload(version))
        assert listing["ets2la_plugin.dll"].local_path.read_bytes() == _payload(version)[
            "ets2la_plugin.dll"
        ]


def test_the_build_tool_fails_when_upstream_is_incomplete(tmp_path, monkeypatch):
    payload = _payload("1.60")
    del payload["scs_sdk_controller.dll"]
    monkeypatch.setattr(fetch_sdk_bundle, "SdkSource", _FakeSource)
    monkeypatch.setattr(_FakeSource, "payloads", {"1.60": payload})
    monkeypatch.setattr(_FakeSource, "published", ("1.60",))
    monkeypatch.setattr(_FakeSource, "offline", False)
    monkeypatch.setattr(_FakeSource, "__init__", lambda self, version, token=None: setattr(
        self, "version", version))

    with pytest.raises(SdkSourceError, match="scs_sdk_controller.dll"):
        fetch_sdk_bundle.fetch(tmp_path / "bundled", token=None)


def test_a_shipped_bundle_matches_its_manifest():
    """Holds for whatever tools/fetch_sdk_bundle.py last wrote; nothing fetched is fine."""
    assert (bundled_mod.BUNDLE_DIR / "LICENSES.txt").exists()
    for version_dir in (p for p in bundled_mod.BUNDLE_DIR.iterdir() if p.is_dir()):
        manifest = bundled_mod.read_manifest(version_dir)
        for name, sha in manifest.items():
            assert git_blob_sha_of(version_dir / name) == sha, f"{version_dir.name}/{name}"
