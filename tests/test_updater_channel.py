"""Unit tests for the stable/preview channel selection and version logic.

Channels are decided by the GitHub `prerelease` flag (not target_commitish),
and the updater reads the channel + installed version from files the app writes
into the install dir.
"""

from __future__ import annotations

import json

import pytest
from packaging.version import Version

# Import the loader first: it puts updater/ on sys.path so `import github_api`
# (a sibling-absolute import from the updater package) resolves.
from tests._updater_loader import load_updater

import github_api  # noqa: E402

mc = load_updater()

RELEASES = [
    {"id": 5, "tag_name": "v1.1.0-preview.2", "prerelease": True},
    {"id": 4, "tag_name": "v1.1.0-preview.1", "prerelease": True},
    {"id": 3, "tag_name": "v1.0.4", "prerelease": False},
    {"id": 2, "tag_name": "v1.0.3", "prerelease": False},
]


def make_api(releases):
    api = github_api.GitHubAPI("luuukske", "MonoCruise")
    api._releases_cache = releases
    return api


def test_preview_only_prereleases_stable_only_stable():
    api = make_api(RELEASES)
    assert [r["id"] for r in api.get_releases_for_channel("preview")] == [5, 4]
    assert [r["id"] for r in api.get_releases_for_channel("stable")] == [3, 2]


def test_latest_per_channel():
    api = make_api(RELEASES)
    assert api.get_latest_release_for_channel("preview")["tag_name"] == "v1.1.0-preview.2"
    assert api.get_latest_release_for_channel("stable")["tag_name"] == "v1.0.4"


def test_latest_stable_none_when_only_prereleases():
    api = make_api([RELEASES[0], RELEASES[1]])  # both prerelease
    assert api.get_latest_release_for_channel("stable") is None
    assert api.get_latest_release_for_channel("preview")["tag_name"] == "v1.1.0-preview.2"


def test_read_channel(tmp_path):
    # The app writes config.json to the install root; the updater reads it
    # from there directly.
    root = str(tmp_path)

    assert mc.read_channel(root) == "stable"  # no config -> default

    (tmp_path / "config.json").write_text(json.dumps({"update_channel": "preview"}))
    assert mc.read_channel(root) == "preview"

    (tmp_path / "config.json").write_text(json.dumps({"update_channel": "Preview"}))
    assert mc.read_channel(root) == "preview"  # case-insensitive

    (tmp_path / "config.json").write_text(json.dumps({"update_channel": "nightly"}))
    assert mc.read_channel(root) == "stable"  # invalid -> default

    (tmp_path / "config.json").write_text("not valid json")
    assert mc.read_channel(root) == "stable"  # unreadable -> default


def test_installed_version(tmp_path):
    root = str(tmp_path)

    assert mc.installed_version(root) is None
    assert mc.installed_version_text(root) == ""

    (tmp_path / "installed_version.txt").write_text("1.1.0-preview.1", encoding="utf-8")
    assert mc.installed_version(root) == Version("1.1.0-preview.1")
    assert mc.installed_version_text(root) == "1.1.0-preview.1"


def test_release_version_parsing():
    assert mc.release_version({"tag_name": "v1.1.0-preview.1"}) == Version("1.1.0-preview.1")
    assert mc.release_version({"tag_name": "1.0.4"}) == Version("1.0.4")
    assert mc.release_version({"tag_name": "garbage"}) is None
    assert mc.release_version({}) is None


def test_prerelease_orders_below_final():
    # PEP 440: a preview of 1.1.0 is older than 1.1.0 final, so a stable 1.1.0
    # is correctly offered as an upgrade to a preview user on 1.1.0-preview.1.
    # (packaging normalizes the "preview" label to "rc"; ordering still holds.)
    assert Version("1.1.0-preview.1") < Version("1.1.0")
    assert Version("1.1.0-preview.1") < Version("1.1.0-preview.2")
    assert Version("1.1.0-preview.1") == Version("1.1.0rc1")
