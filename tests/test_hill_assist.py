"""Hill assist is the stop brake. Off, a stopped truck is not held."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from core.sending_thread.hold_controller import (
    STATE_HOLDING,
    STATE_ROLLING,
    HoldController,
    release_stationary_auto_brake,
)
from core.settings import Settings
from ui.main_window.settings_panel import SettingsPanel

DT = 0.01


def _hold() -> HoldController:
    return HoldController(lambda d: min(1.0, max(0.0, d) / 10.0))


def _stopped(h: HoldController, *, enabled: bool = True):
    return h.update(
        speed_kmh=0.0,
        gear=1,
        pitch_norm=0.0,
        commanded_accel_ms2=0.0,
        gasval=0.0,
        opdgasval=0.0,
        offset=0.0,
        park_brake=False,
        aeb_active=False,
        dt=DT,
        enabled=enabled,
    )


def test_a_stop_still_holds_when_hill_assist_is_on():
    out = _stopped(_hold())
    assert out.state == STATE_HOLDING
    assert out.brake_pedal > 0.0
    assert out.active is True


def test_turning_hill_assist_off_drops_the_hold_and_the_brake():
    h = _hold()
    assert _stopped(h).state == STATE_HOLDING
    out = _stopped(h, enabled=False)
    assert out.state == STATE_ROLLING
    assert out.brake_pedal == 0.0
    assert out.active is False
    assert h.state == STATE_ROLLING


def test_turning_hill_assist_back_on_holds_again():
    h = _hold()
    _stopped(h, enabled=False)
    out = _stopped(h, enabled=True)
    assert out.state == STATE_HOLDING
    assert out.brake_pedal > 0.0


def test_off_releases_a_stop_that_is_not_asking_for_brake():
    assert release_stationary_auto_brake(False, False, 0.2, 0.0) is True
    assert release_stationary_auto_brake(False, False, 0.0, 0.4) is True


def test_off_still_brakes_while_slowing_or_already_rolling():
    assert release_stationary_auto_brake(False, False, 0.2, -0.4) is False
    assert release_stationary_auto_brake(False, False, 3.0, 0.0) is False


def test_on_and_aeb_never_release_the_stop():
    assert release_stationary_auto_brake(True, False, 0.0, 0.0) is False
    assert release_stationary_auto_brake(False, True, 0.0, 0.0) is False


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_settings_switch_writes_hill_assist(qapp):
    settings = Settings.instance()
    previous = settings.hill_assist
    settings.hill_assist = True
    host = QWidget()
    try:
        panel = SettingsPanel(
            host,
            settings,
            on_save=lambda: None,
            on_reset=lambda: None,
            show_confirm=lambda *a, **k: None,
            show_consent=lambda **kw: None,
        )
        assert panel.chk_hill_assist.isChecked() is True
        panel.chk_hill_assist.setChecked(False)
        assert settings.hill_assist is False
        settings.hill_assist = True
        panel.apply_settings(settings)
        assert panel.chk_hill_assist.isChecked() is True
    finally:
        settings.hill_assist = previous
        host.deleteLater()
