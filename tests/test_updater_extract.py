"""Unit tests for the updater's install/preserve/skip and self-update logic.

This is the highest-risk surface: a bug here can wipe a user's config, write
outside the install dir, or brick the updater. Covers preserve-on-install,
the skip rules, the zip-slip guard, staging of the updater's own files to
updater_pending/, and the startup swap that applies them.
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


def worker(install_root):
    return mc.UpdateWorker(Mock(), {}, str(install_root))


def test_install_writes_app_files_skips_preserved(tmp_path):
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

    worker(install)._install_update(str(zip_path))

    assert (install / "MonoCruise.exe").exists()
    assert (install / "_internal" / "foo.dll").exists()
    for skipped in ("config.json", "config.json.bak", "logs/run-1.log",
                    "updater.exe", "updater.py"):
        assert not (install / skipped).exists(), f"{skipped} should not be installed"
    # Staging dir is cleaned up after the install.
    assert not (install / mc.STAGING_DIR).exists()


def test_install_preserves_existing_user_state(tmp_path):
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

    worker(install)._install_update(str(zip_path))

    assert (install / "config.json").read_text(encoding="utf-8") == "CUSTOM"
    assert (install / "logs" / "run-old.log").read_text(encoding="utf-8") == "OLD"
    assert (install / "MonoCruise.exe").exists()


def test_install_stages_updater_files_to_pending(tmp_path):
    install = tmp_path / "install"
    install.mkdir()
    zip_path = tmp_path / "update.zip"
    make_zip(zip_path, {
        "MonoCruise.exe": "exe",
        "updater/updater.exe": "new-updater",
        "updater/_internal/bar.dll": "dll",
    })

    worker(install)._install_update(str(zip_path))

    pending = install / mc.PENDING_DIR
    assert (pending / "updater.exe").read_text(encoding="utf-8") == "new-updater"
    assert (pending / "_internal" / "bar.dll").exists()
    # Sentinel marks the staged tree as complete for the next-launch swap.
    assert (pending / mc.PENDING_SENTINEL).exists()
    # Nothing lands in the live updater dir during the install itself.
    assert not (install / "updater" / "updater.exe").exists()


def test_install_rejects_zip_without_app_exe(tmp_path):
    install = tmp_path / "install"
    install.mkdir()
    zip_path = tmp_path / "update.zip"
    make_zip(zip_path, {"_internal/foo.dll": "dll"})

    with pytest.raises(ValueError, match="missing MonoCruise.exe"):
        worker(install)._install_update(str(zip_path))

    # Nothing was moved into the install dir.
    assert not (install / "_internal").exists()
    assert not (install / mc.STAGING_DIR).exists()


def test_zip_slip_is_blocked(tmp_path):
    install = tmp_path / "install"
    install.mkdir()
    zip_path = tmp_path / "evil.zip"
    make_zip(zip_path, {"../../evil.exe": "bad", "MonoCruise.exe": "exe"})

    with pytest.raises(ValueError, match="Blocked malicious zip entry"):
        worker(install)._install_update(str(zip_path))

    # Nothing written outside the install dir, nothing moved into it.
    assert not (tmp_path / "evil.exe").exists()
    assert not (tmp_path.parent / "evil.exe").exists()
    assert not (install / "MonoCruise.exe").exists()


def test_should_skip_rules(tmp_path):
    w = worker(tmp_path)
    # Preserved / legacy root-level updater files -> skipped.
    assert w._should_skip("config.json") is True
    assert w._should_skip("Config.JSON") is True            # case-insensitive
    assert w._should_skip("config.json.bak") is True
    assert w._should_skip("logs/run.log") is True
    assert w._should_skip("updater.exe") is True
    assert w._should_skip("updater.py") is True
    # Not skipped -> installed (updater/ subtree goes to updater_pending).
    assert w._should_skip("updater/updater.exe") is False
    assert w._should_skip("updater/_internal/bar.dll") is False
    assert w._should_skip("config.json.malicious") is False  # exact match only
    assert w._should_skip("subdir/config.json") is False     # only top-level preserved
    assert w._should_skip("MonoCruise.exe") is False


# The swap itself is applied by MonoCruise (shared/updater_swap.py), not by
# the updater: a running updater cannot rename its own tree.
from shared import updater_swap as swap


def _not_running() -> bool:
    return False


def _make_updater_tree(root, marker: str):
    udir = root / swap.UPDATER_DIR
    (udir / "_internal").mkdir(parents=True)
    (udir / "updater.exe").write_text(marker, encoding="utf-8")
    (udir / "_internal" / "lib.dll").write_text(marker, encoding="utf-8")
    return udir


def test_swap_constants_match_updater():
    # The updater stages into these paths; the app swaps them. A rename on
    # either side silently breaks self-updates.
    assert mc.PENDING_DIR == swap.PENDING_DIR
    assert mc.OLD_DIR == swap.OLD_DIR
    assert mc.PENDING_SENTINEL == swap.PENDING_SENTINEL


def test_swap_applies_complete_pending(tmp_path):
    udir = _make_updater_tree(tmp_path, "old")
    pending = tmp_path / swap.PENDING_DIR
    (pending / "_internal").mkdir(parents=True)
    (pending / "updater.exe").write_text("new", encoding="utf-8")
    (pending / "_internal" / "lib.dll").write_text("new", encoding="utf-8")
    (pending / swap.PENDING_SENTINEL).write_text("", encoding="utf-8")

    assert swap.apply_pending_updater_swap(str(tmp_path), running_check=_not_running) is True

    assert (udir / "updater.exe").read_text(encoding="utf-8") == "new"
    assert (udir / "_internal" / "lib.dll").read_text(encoding="utf-8") == "new"
    assert not (udir / swap.PENDING_SENTINEL).exists()
    assert not pending.exists()
    assert not (tmp_path / swap.OLD_DIR).exists()


def test_swap_discards_incomplete_pending(tmp_path):
    udir = _make_updater_tree(tmp_path, "old")
    pending = tmp_path / swap.PENDING_DIR
    pending.mkdir()
    (pending / "updater.exe").write_text("half", encoding="utf-8")
    # No sentinel: the stage never completed and must not be swapped in.

    assert swap.apply_pending_updater_swap(str(tmp_path), running_check=_not_running) is False

    assert (udir / "updater.exe").read_text(encoding="utf-8") == "old"
    assert not pending.exists()


def test_swap_skipped_while_updater_runs(tmp_path):
    udir = _make_updater_tree(tmp_path, "old")
    pending = tmp_path / swap.PENDING_DIR
    pending.mkdir()
    (pending / "updater.exe").write_text("new", encoding="utf-8")
    (pending / swap.PENDING_SENTINEL).write_text("", encoding="utf-8")

    assert swap.apply_pending_updater_swap(str(tmp_path), running_check=lambda: True) is False

    # Untouched and fully retryable: sentinel still in place.
    assert (udir / "updater.exe").read_text(encoding="utf-8") == "old"
    assert (pending / swap.PENDING_SENTINEL).exists()


def test_swap_cleans_stale_old_dir(tmp_path):
    udir = _make_updater_tree(tmp_path, "current")
    old = tmp_path / swap.OLD_DIR
    old.mkdir()
    (old / "updater.exe").write_text("stale", encoding="utf-8")

    assert swap.apply_pending_updater_swap(str(tmp_path), running_check=_not_running) is False

    assert not old.exists()
    assert (udir / "updater.exe").read_text(encoding="utf-8") == "current"


def test_pending_swap_staged(tmp_path):
    assert swap.pending_swap_staged(str(tmp_path)) is False
    (tmp_path / swap.PENDING_DIR).mkdir()
    assert swap.pending_swap_staged(str(tmp_path)) is True
