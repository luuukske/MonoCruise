"""Unit tests for the updater's extract/preserve/skip logic.

This is the highest-risk surface: a bug here can wipe a user's config or write
outside the install dir. Covers preserve-on-extract, the exact-match skip rules,
and the zip-slip guard.
"""

from __future__ import annotations

import zipfile
from unittest.mock import Mock

import pytest

from tests._updater_loader import load_updater

mc = load_updater()


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    # UpdateWorker is a QThread; a QApplication must exist for signal machinery.
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def make_zip(path, entries: dict) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


def worker(install_dir):
    return mc.UpdateWorker(Mock(), {}, str(install_dir))


def test_extract_writes_app_files_skips_preserved(tmp_path):
    install = tmp_path / "install"
    install.mkdir()
    zip_path = tmp_path / "update.zip"
    make_zip(zip_path, {
        "config.json": "{}",
        "config.json.bak": "{}",
        "logs/run-1.log": "log",
        "logs/": "",
        "MonoCruise.exe": "exe",
        "updater.exe": "exe",
        "updater.py": "py",
        "_internal/foo.dll": "dll",
    })

    worker(install)._extract_update(str(zip_path))

    assert (install / "MonoCruise.exe").exists()
    assert (install / "_internal" / "foo.dll").exists()
    for skipped in ("config.json", "config.json.bak", "logs/run-1.log",
                    "updater.exe", "updater.py"):
        assert not (install / skipped).exists(), f"{skipped} should not be extracted"


def test_extract_preserves_existing_user_state(tmp_path):
    install = tmp_path / "install"
    (install / "logs").mkdir(parents=True)
    (install / "config.json").write_text("CUSTOM", encoding="utf-8")
    (install / "logs" / "run-old.log").write_text("OLD", encoding="utf-8")

    zip_path = tmp_path / "update.zip"
    make_zip(zip_path, {
        "config.json": "OVERWRITTEN",
        "logs/run-1.log": "new",
        "MonoCruise.exe": "exe",
    })

    worker(install)._extract_update(str(zip_path))

    assert (install / "config.json").read_text(encoding="utf-8") == "CUSTOM"
    assert (install / "logs" / "run-old.log").read_text(encoding="utf-8") == "OLD"
    assert (install / "MonoCruise.exe").exists()


def test_zip_slip_is_blocked(tmp_path):
    install = tmp_path / "install"
    install.mkdir()
    zip_path = tmp_path / "evil.zip"
    make_zip(zip_path, {"../../evil.exe": "bad"})

    with pytest.raises(ValueError, match="Blocked malicious zip entry"):
        worker(install)._extract_update(str(zip_path))

    # Nothing written outside the install dir.
    assert not (tmp_path / "evil.exe").exists()
    assert not (tmp_path.parent / "evil.exe").exists()


def test_should_skip_rules(tmp_path):
    w = worker(tmp_path)
    # Preserved / updater files -> skipped.
    assert w._should_skip("config.json") is True
    assert w._should_skip("Config.JSON") is True            # case-insensitive
    assert w._should_skip("config.json.bak") is True
    assert w._should_skip("logs/run.log") is True
    assert w._should_skip("updater.exe") is True
    assert w._should_skip("updater.py") is True
    # Not preserved -> extracted.
    assert w._should_skip("config.json.malicious") is False  # exact match only
    assert w._should_skip("subdir/config.json") is False     # only top-level preserved
    assert w._should_skip("MonoCruise.exe") is False
