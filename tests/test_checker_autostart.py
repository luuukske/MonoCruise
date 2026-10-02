"""The in-app autostart toggle: the checker honours it, and the app greys it out
when no checker is running to act on it."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QWidget

import checker.ets2_checker as ets2_checker
from core import checker_status
from core.settings import Settings
from ui.main_window.settings_panel import SettingsPanel


@pytest.fixture()
def install_root(tmp_path, monkeypatch):
    """A fake install root: the checker lives in <root>/checker, config.json in <root>."""
    (tmp_path / "checker").mkdir()
    monkeypatch.setattr(ets2_checker, "_base_dir", lambda: str(tmp_path / "checker"))
    return tmp_path


def _write_config(root, payload) -> None:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    (root / "config.json").write_text(text, encoding="utf-8")


def test_checker_launches_only_when_autostart_is_on(install_root):
    _write_config(install_root, {"autostart_variable": True})
    assert ets2_checker.autostart_enabled() is True
    _write_config(install_root, {"autostart_variable": False})
    assert ets2_checker.autostart_enabled() is False


@pytest.mark.parametrize("payload", [None, "{not json", {}, [], {"autostart_variable": None}])
def test_checker_keeps_launching_when_the_setting_is_unreadable(install_root, payload):
    # Missing, mid-write or odd config must not silently turn autostart off.
    if payload is not None:
        _write_config(install_root, payload)
    assert ets2_checker.autostart_enabled() is True


def test_game_start_respects_the_setting(monkeypatch):
    launched: list[bool] = []
    monkeypatch.setattr(ets2_checker, "monocruise_running", lambda: False)
    monkeypatch.setattr(ets2_checker, "launch_monocruise", lambda: launched.append(True) or True)

    monkeypatch.setattr(ets2_checker, "autostart_enabled", lambda: False)
    assert ets2_checker.handle_game_start() is True
    assert launched == []

    monkeypatch.setattr(ets2_checker, "autostart_enabled", lambda: True)
    assert ets2_checker.handle_game_start() is True
    assert launched == [True]


def test_game_start_stops_when_monocruise_is_gone(monkeypatch):
    monkeypatch.setattr(ets2_checker, "monocruise_running", lambda: False)
    monkeypatch.setattr(ets2_checker, "autostart_enabled", lambda: True)
    monkeypatch.setattr(ets2_checker, "launch_monocruise", lambda: False)
    assert ets2_checker.handle_game_start() is False


def test_app_probes_the_checker_mutex_by_the_same_name():
    assert checker_status.CHECKER_MUTEX_NAME == ets2_checker.CHECKER_MUTEX_NAME


def test_app_never_writes_the_startup_registry():
    # AV heuristics flagged v1.0.x for this; only the installer may write the Run key.
    source = "".join(
        Path(mod.__file__).read_text(encoding="utf-8") for mod in (checker_status, ets2_checker)
    )
    assert "winreg" not in source
    assert "RegSetValue" not in source


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _panel(monkeypatch, running: bool) -> tuple[QWidget, SettingsPanel]:
    monkeypatch.setattr(checker_status, "checker_running", lambda: running)
    settings = Settings.instance()
    monkeypatch.setattr(settings, "autostart_variable", True)
    host = QWidget()
    p = SettingsPanel(
        host, settings,
        on_save=lambda: None, on_reset=lambda: None,
        show_confirm=lambda *a, **k: None, show_consent=lambda **kw: None,
    )
    return host, p


def test_toggle_is_greyed_out_without_the_checker(qapp, monkeypatch):
    host, p = _panel(monkeypatch, running=False)
    try:
        assert not p.chk_autostart.isEnabled()
        assert not p.chk_autostart.isChecked()
        assert p._autostart_subtext.isVisibleTo(p)
        assert "installation config" in p._autostart_subtext.text()
        # Greying out must not rewrite the stored setting.
        assert Settings.instance().autostart_variable is True
    finally:
        host.deleteLater()


def test_toggle_comes_back_once_the_checker_runs(qapp, monkeypatch):
    host, p = _panel(monkeypatch, running=False)
    try:
        monkeypatch.setattr(checker_status, "checker_running", lambda: True)
        p.refresh_autostart_availability()
        assert p.chk_autostart.isEnabled()
        assert p.chk_autostart.isChecked()
        assert not p._autostart_subtext.isVisibleTo(p)
        p.chk_autostart.setChecked(False)
        assert Settings.instance().autostart_variable is False
    finally:
        host.deleteLater()
