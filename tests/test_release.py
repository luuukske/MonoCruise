"""Unit tests for tools/release.py (version bumping + CHANGELOG promotion).

The release helper reads/writes module-level VERSION_FILE and CHANGELOG paths.
Tests point those at synthetic files in a tmp_path so nothing real is touched,
and use the --dry-run codepath / a stubbed _run so no git operations fire.
"""

from __future__ import annotations

import argparse
import datetime as dt
import types

import pytest

import tools.release as release

VERSION_TMPL = '__version__ = "{v}"\n'


def write_changelog(path, unreleased_body: str = "", extra_sections: str = "") -> None:
    path.write_text(
        "# Changelog\n\n"
        f"## [Unreleased]\n{unreleased_body}\n"
        f"{extra_sections}",
        encoding="utf-8",
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    version_file = tmp_path / "version.py"
    changelog = tmp_path / "CHANGELOG.md"
    version_file.write_text(VERSION_TMPL.format(v="1.1.0"), encoding="utf-8")
    write_changelog(changelog)  # empty [Unreleased]
    monkeypatch.setattr(release, "VERSION_FILE", version_file)
    monkeypatch.setattr(release, "CHANGELOG", changelog)
    return types.SimpleNamespace(version_file=version_file, changelog=changelog)


@pytest.fixture
def fixed_today(monkeypatch):
    class _FakeDate(dt.date):
        @classmethod
        def today(cls):
            return cls(2026, 6, 26)

    monkeypatch.setattr(release, "_dt", types.SimpleNamespace(date=_FakeDate))


@pytest.mark.parametrize(
    "current, level, pre, expected",
    [
        ("1.1.0", "patch", None, "1.1.1"),
        ("1.1.0", "minor", None, "1.2.0"),
        ("1.9.4", "major", None, "2.0.0"),
        ("1.1.0", None, "preview", "1.1.0-preview.1"),
        ("1.1.0-preview.1", None, "preview", "1.1.0-preview.2"),
        ("1.1.0-preview.3", "patch", None, "1.1.0"),   # finalize -> drop suffix
        ("1.1.0-preview.3", "minor", None, "1.1.0"),   # any level finalizes
        ("1.1.0-preview.1", None, "rc", "1.1.0-rc.1"), # label change resets counter
        ("1.1.0", "minor", "preview", "1.2.0-preview.1"),
    ],
)
def test_bump(current, level, pre, expected):
    assert release._bump(current, level, pre) == expected


def test_read_version_handles_pre_release_suffix(repo):
    repo.version_file.write_text(VERSION_TMPL.format(v="1.1.0-preview.2"), encoding="utf-8")
    assert release._read_version() == "1.1.0-preview.2"


def test_unreleased_empty(repo):
    assert release._unreleased_has_content() is False


def test_unreleased_with_content(repo):
    write_changelog(repo.changelog, unreleased_body="\n### Added\n- new thing\n")
    assert release._unreleased_has_content() is True


def test_extract_section(repo):
    write_changelog(
        repo.changelog,
        extra_sections="## [1.1.0] - 2025-01-01\n\n### Added\n- thing one\n",
    )
    body = release._extract_section("1.1.0")
    assert body is not None
    assert "thing one" in body


def test_extract_section_missing(repo):
    assert release._extract_section("9.9.9") is None


def test_promote_unreleased(repo, fixed_today):
    write_changelog(repo.changelog, unreleased_body="\n### Added\n- thing one\n")
    release._promote_unreleased("1.1.0-preview.1")
    text = repo.changelog.read_text(encoding="utf-8")
    assert "## [Unreleased]" in text
    assert "## [1.1.0-preview.1] - 2026-06-26" in text
    assert "thing one" in text
    # Fresh [Unreleased] sits above the newly promoted section.
    assert text.index("## [Unreleased]") < text.index("## [1.1.0-preview.1]")


def test_cmd_bump_refuses_empty_unreleased(repo):
    args = argparse.Namespace(set=None, level="patch", pre=None, dry_run=True)
    with pytest.raises(SystemExit):
        release.cmd_bump(args)


def test_cmd_bump_pre_dry_run(repo, fixed_today, monkeypatch):
    write_changelog(repo.changelog, unreleased_body="\n### Added\n- new\n")
    monkeypatch.setattr(release, "_run", lambda *a: pytest.fail("git must not run in dry-run"))
    args = argparse.Namespace(set=None, level=None, pre="preview", dry_run=True)
    release.cmd_bump(args)
    assert release._read_version() == "1.1.0-preview.1"
    assert "## [1.1.0-preview.1] - 2026-06-26" in repo.changelog.read_text(encoding="utf-8")


def test_cmd_notes(repo, capsys):
    write_changelog(
        repo.changelog,
        extra_sections="## [1.1.0] - 2025-01-01\n\n### Added\n- thing one\n",
    )
    release.cmd_notes(argparse.Namespace(version="1.1.0"))
    assert "thing one" in capsys.readouterr().out


def test_cmd_notes_unknown_version(repo):
    with pytest.raises(SystemExit):
        release.cmd_notes(argparse.Namespace(version="9.9.9"))
