"""The global-limit box speaks mph on ATS and still saves km/h."""

from __future__ import annotations

import pytest

from PySide6.QtWidgets import QApplication, QWidget

from core.settings import Settings
from core.speed_units import MPH_TO_KMH
from ui.main_window.settings_panel import SettingsPanel


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def panel(qapp, monkeypatch):
    settings = Settings.instance()
    monkeypatch.setattr(settings, "cc_mode", "Cruise control")
    monkeypatch.setattr(settings, "cc_start_button", None)
    monkeypatch.setattr(settings, "global_speed_limit_kmh", 90.0)

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


def test_ats_shows_the_nearest_mph_and_does_not_rewrite_the_cap(panel, monkeypatch):
    p, settings = panel
    monkeypatch.setattr(settings, "last_game", 2)
    monkeypatch.setattr(settings, "global_speed_limit_kmh", 90.0)
    p.refresh_speed_unit()

    assert p.ent_global_limit.text() == "56"
    assert p.ent_global_limit._mc_unit_label.text() == "mph"
    assert (p.ent_global_limit._mc_minimum, p.ent_global_limit._mc_maximum) == (38, 80)
    assert settings.global_speed_limit_kmh == 90.0
    assert p.opt_short.currentText().endswith("mph")

    p.ent_global_limit.setText("100")
    p.ent_global_limit.returnPressed.emit()
    assert p.ent_global_limit.text() == "80"
    assert settings.global_speed_limit_kmh == pytest.approx(80 * MPH_TO_KMH)

    p._on_global_limit(65)
    assert settings.global_speed_limit_kmh == pytest.approx(65 * MPH_TO_KMH)


def test_ets2_box_stays_kmh_and_stores_the_typed_number(panel, monkeypatch):
    p, settings = panel
    monkeypatch.setattr(settings, "last_game", 1)
    monkeypatch.setattr(settings, "global_speed_limit_kmh", 90.0)
    p.refresh_speed_unit()

    assert p.ent_global_limit.text() == "90.0"
    assert p.ent_global_limit._mc_unit_label.text() == "km/h"
    assert (p.ent_global_limit._mc_minimum, p.ent_global_limit._mc_maximum) == (60, 130)

    p.ent_global_limit.setText("20")
    p.ent_global_limit.returnPressed.emit()
    assert p.ent_global_limit.text() == "60"
    assert settings.global_speed_limit_kmh == 60

    p._on_global_limit(80)
    assert settings.global_speed_limit_kmh == 80

    p._on_global_limit(None)
    assert settings.global_speed_limit_kmh is None
