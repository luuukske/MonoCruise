"""Windows Apps-list (uninstall entry) relabelling after an in-place update.

The registry side is Windows-only, so what is asserted here is the decision
logic (which values need writing, and whether an entry belongs to this install)
plus the updater actually calling it. Those run everywhere, including the Linux
CI job."""
from __future__ import annotations

import zipfile
from unittest.mock import Mock

import pytest

from shared import windows_app_details as details
from tests._updater_loader import load_updater

mc = load_updater()


@pytest.fixture(scope="module", autouse=True)
def _qapp():
    # UpdateWorker is a QThread; a QApplication must exist for signal machinery.
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


# Value planning

def test_display_name_matches_installer_shape():
    # UninstallDisplayName in installer/MonoCruise.iss is "MonoCruise {version}".
    assert details.display_name("1.1.0-preview.23") == "MonoCruise 1.1.0-preview.23"
    assert details.display_name("") == "MonoCruise"


def test_plan_rewrites_both_values_after_an_update():
    current = {
        "DisplayVersion": "1.1.0-preview.9",
        "DisplayName": "MonoCruise 1.1.0-preview.9",
    }
    assert details.plan_values("1.1.0-preview.23", current) == {
        "DisplayVersion": "1.1.0-preview.23",
        "DisplayName": "MonoCruise 1.1.0-preview.23",
    }


def test_plan_is_empty_when_already_in_sync():
    # An in-sync install must not write to the registry on every boot.
    current = {
        "DisplayVersion": "1.1.0-preview.23",
        "DisplayName": "MonoCruise 1.1.0-preview.23",
    }
    assert details.plan_values("1.1.0-preview.23", current) == {}


def test_plan_fills_in_missing_values():
    assert details.plan_values("1.0.4", {"DisplayVersion": None, "DisplayName": None}) == {
        "DisplayVersion": "1.0.4",
        "DisplayName": "MonoCruise 1.0.4",
    }


def test_plan_keeps_a_display_name_someone_else_set():
    current = {"DisplayVersion": "1.0.3", "DisplayName": "Truck cruise thing"}
    # Version still corrected; the hand-set name is left alone.
    assert details.plan_values("1.0.4", current) == {"DisplayVersion": "1.0.4"}


def test_plan_does_nothing_without_a_version():
    assert details.plan_values("", {"DisplayVersion": "1.0.3"}) == {}


# Which install an entry belongs to

def test_entry_matches_same_install_root():
    assert details.entry_matches_install(
        r"C:\Apps\Local\Programs\MonoCruise" + "\\",
        r"C:\Apps\Local\Programs\MonoCruise",
    ) is True


def test_entry_does_not_match_another_install():
    assert details.entry_matches_install(
        r"C:\Program Files\MonoCruise", r"D:\dev\MonoCruise"
    ) is False


def test_entry_without_install_location_counts_as_ours():
    # Older installers did not record InstallLocation; the key name matched.
    assert details.entry_matches_install("", r"C:\Apps\MonoCruise") is True


def test_sync_is_a_no_op_without_a_version():
    # Guard runs before any registry access, so this is safe on a real machine.
    assert details.sync_app_details("", r"C:\Apps\MonoCruise") is False


# Updater wiring

def test_release_version_text_strips_the_tag_prefix():
    assert mc.release_version_text({"tag_name": "v1.1.0-preview.23"}) == "1.1.0-preview.23"
    assert mc.release_version_text({"tag_name": "1.0.4"}) == "1.0.4"
    assert mc.release_version_text({}) == ""


def test_updater_relabels_windows_entry_after_installing(tmp_path, monkeypatch):
    install = tmp_path / "install"
    install.mkdir()
    zip_path = tmp_path / "update.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("MonoCruise.exe", "exe")
    payload = zip_path.read_bytes()

    api = Mock()
    api.get_release_asset_url.return_value = "https://example.invalid/Update.zip"
    api.download_asset.side_effect = (
        lambda url, dest, progress_callback=None: open(dest, "wb").write(payload)
    )

    synced = []
    monkeypatch.setattr(mc, "sync_app_details",
                        lambda release, root: synced.append((release, root)))

    release = {"tag_name": "v1.1.0-preview.23"}
    worker = mc.UpdateWorker(api, release, str(install))
    errors = []
    worker.error.connect(errors.append)
    worker.run()

    assert errors == []
    assert (install / "MonoCruise.exe").exists()
    assert synced == [(release, str(install))]


def test_updater_skips_relabel_when_the_install_fails(tmp_path, monkeypatch):
    install = tmp_path / "install"
    install.mkdir()
    zip_path = tmp_path / "update.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("_internal/foo.dll", "dll")  # no MonoCruise.exe: rejected
    payload = zip_path.read_bytes()

    api = Mock()
    api.get_release_asset_url.return_value = "https://example.invalid/Update.zip"
    api.download_asset.side_effect = (
        lambda url, dest, progress_callback=None: open(dest, "wb").write(payload)
    )

    synced = []
    monkeypatch.setattr(mc, "sync_app_details",
                        lambda release, root: synced.append((release, root)))

    worker = mc.UpdateWorker(api, {"tag_name": "v1.1.0-preview.23"}, str(install))
    errors = []
    worker.error.connect(errors.append)
    worker.run()

    # Files never landed, so Windows must keep reporting the old version.
    assert errors and "MonoCruise.exe" in errors[0]
    assert synced == []
