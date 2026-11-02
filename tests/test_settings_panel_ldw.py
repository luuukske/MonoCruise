"""The settings panel's lane departure dropdown: None, Warn and Steer onto ldw_enabled and lka_enabled.

A pick other than None installs the FFB plugin (core/sdk_installer/README.md, MonoCruise's own plugins).
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QWidget

import core.sdk_installer as sdk_installer
from core.settings import Settings
from ui.main_window.settings_panel import SettingsPanel

_TOUCHED = ("ldw_enabled", "lka_enabled")


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def installs(monkeypatch):
    """Plugin installs the panel started, by file name; nothing reaches a game folder."""
    started: list[str] = []
    monkeypatch.setattr(sdk_installer, "start_own_install", lambda name, on_result: started.append(name))
    return started


@pytest.fixture()
def panel(qapp, installs, monkeypatch):
    s = Settings.instance()
    saved = {name: getattr(s, name) for name in _TOUCHED}
    s.ldw_enabled, s.lka_enabled = False, False
    monkeypatch.setattr(SettingsPanel, "_get_thread", staticmethod(lambda name: None))
    host = QWidget()
    p = SettingsPanel(
        host, s,
        on_save=lambda: None, on_reset=lambda: None,
        show_confirm=lambda *a, **k: None, show_consent=lambda **kw: None,
    )
    p._bind_timer.stop()
    yield p
    host.deleteLater()
    for name, value in saved.items():
        setattr(s, name, value)


def _load(p: SettingsPanel, ldw: bool, lka: bool) -> None:
    s = Settings.instance()
    s.ldw_enabled, s.lka_enabled = ldw, lka
    p.apply_settings(s)


@pytest.mark.parametrize(
    "ldw, lka, shown",
    [(False, False, "None"), (True, False, "Warn"), (True, True, "Steer"), (False, True, "Steer")],
)
def test_it_shows_the_two_switches(panel, installs, ldw, lka, shown):
    _load(panel, ldw, lka)
    assert panel.opt_ldw.currentText() == shown
    assert installs == [], "loading settings never installs anything"


@pytest.mark.parametrize(
    "mode, ldw, lka",
    [("Warn", True, False), ("Steer", True, True), ("None", False, False)],
)
def test_a_pick_sets_both_switches_and_installs_the_plugin_unless_none(panel, installs, mode, ldw, lka):
    was_on = mode == "None"
    _load(panel, was_on, was_on)
    panel.opt_ldw.setCurrentText(mode)
    s = Settings.instance()
    assert (s.ldw_enabled, s.lka_enabled) == (ldw, lka)
    assert installs == ([] if mode == "None" else ["monocruise_ffb.dll"])
