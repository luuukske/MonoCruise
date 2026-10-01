"""Settings-panel wiring for the acceleration style dropdown.

The row is hidden in Speed limiter mode on purpose: the envelope only shapes the
CC bid, and the limiter's positive bid caps the user's own pedal instead.
"""
from __future__ import annotations

import pytest

from PySide6.QtWidgets import QApplication, QWidget

from core.longitudinal.accel_envelope import PROFILE_LABELS
from core.settings import Settings
from ui.main_window.settings_panel import SettingsPanel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def panel(qapp):
    settings = Settings.instance()
    settings.cc_mode = "Cruise control"
    settings.cc_accel_profile = "Normal"
    settings.cc_start_button = None

    host = QWidget()
    p = SettingsPanel(
        host,
        settings,
        on_save=lambda: None,
        on_reset=lambda: None,
        show_confirm=lambda *a, **k: None,
        show_consent=lambda **kw: None,
    )
    yield p, settings
    host.deleteLater()


def _row_hidden(p) -> bool:
    item = p._grid.itemAtPosition(p._accel_style_row, 1)
    return item.widget().isHidden()


def test_dropdown_offers_every_profile(panel):
    p, _ = panel
    values = [p.opt_accel_style.itemText(i) for i in range(p.opt_accel_style.count())]
    assert values == list(PROFILE_LABELS)


def test_dropdown_starts_on_the_saved_profile(panel):
    p, _ = panel
    assert p.opt_accel_style.currentText() == "Normal"


def test_selecting_a_profile_writes_the_setting(panel):
    p, settings = panel
    p.opt_accel_style.setCurrentText("Sport")
    assert settings.cc_accel_profile == "Sport"


def test_row_hides_in_speed_limiter_mode_and_returns(panel):
    p, settings = panel
    assert not _row_hidden(p)

    p._set_cruise_mode("Speed limiter")
    assert settings.cc_mode == "Speed limiter"
    assert _row_hidden(p)

    p._set_cruise_mode("Cruise control")
    assert not _row_hidden(p)


def test_bulk_reload_repaints_the_dropdown_without_echoing(panel):
    p, settings = panel
    settings.cc_accel_profile = "Efficiency"
    p.apply_settings(settings)
    assert p.opt_accel_style.currentText() == "Efficiency"
    assert settings.cc_accel_profile == "Efficiency"


def test_a_junk_stored_profile_falls_back_in_the_widget(panel):
    p, settings = panel
    settings.cc_accel_profile = "ludicrous"
    p.apply_settings(settings)
    assert p.opt_accel_style.currentText() == "Normal"
