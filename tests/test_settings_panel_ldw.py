"""The settings panel's lane departure dropdown: None and Warn onto ldw_enabled.

Warn installs the FFB plugin (core/sdk_installer/README.md, MonoCruise's own plugins). Steer is config only for
now (lka_enabled, Lukas 2026-11-20): the dropdown neither offers it nor touches it.
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
def saves():
    return []


@pytest.fixture()
def panel(qapp, installs, saves, monkeypatch):
    s = Settings.instance()
    saved = {name: getattr(s, name) for name in _TOUCHED}
    s.ldw_enabled, s.lka_enabled = False, False
    monkeypatch.setattr(SettingsPanel, "_get_thread", staticmethod(lambda name: None))
    host = QWidget()
    p = SettingsPanel(
        host, s,
        on_save=lambda: saves.append(1), on_reset=lambda: None,
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


def test_steer_is_not_offered(panel):
    offered = [panel.opt_ldw.itemText(i) for i in range(panel.opt_ldw.count())]
    assert offered == ["None", "Warn"]


@pytest.mark.parametrize(
    "ldw, lka, shown",
    [(False, False, "None"), (True, False, "Warn"), (True, True, "Warn"), (False, True, "None")],
)
def test_it_shows_the_rumble_switch_only(panel, installs, ldw, lka, shown):
    _load(panel, ldw, lka)
    assert panel.opt_ldw.currentText() == shown
    assert installs == [], "loading settings never installs anything"


@pytest.mark.parametrize("lka", [False, True])
@pytest.mark.parametrize("mode, ldw", [("Warn", True), ("None", False)])
def test_a_pick_sets_the_rumble_keeps_the_config_lka_and_installs_unless_none(panel, installs, saves, mode, ldw, lka):
    was_on = mode == "None"
    _load(panel, was_on, lka)
    panel.opt_ldw.setCurrentText(mode)
    s = Settings.instance()
    assert (s.ldw_enabled, s.lka_enabled) == (ldw, lka), "a config-set lka_enabled survives every pick"
    assert saves, "the pick is saved to config.json"
    assert installs == ([] if mode == "None" else ["monocruise_ffb.dll"])
