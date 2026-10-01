"""ATS button steps are mph. ETS2 tests pin last_game themselves."""

from __future__ import annotations

import pytest

from core.cruise_control_thread.thread import CruiseControlThread
from core.settings import Settings
from core.speed_units import MPH_TO_KMH
from core.thread_management.registry import registry
from tests.longitudinal.harness import (
    FakeThread,
    pedal_data,
    sending_data,
    telemetry_data,
)


def _ms(mph: float) -> float:
    return mph * MPH_TO_KMH / 3.6


@pytest.fixture
def rig(monkeypatch):
    tel = FakeThread("telemetry_thread", telemetry_data(speed=_ms(65.0)))
    pedal = FakeThread("main_pedal_thread", pedal_data())
    sending = FakeThread("sending_thread", sending_data())
    for t in (tel, pedal, sending):
        registry.replace(t)

    settings = Settings.instance()
    for key, value in (
        ("cc_mode", "Cruise control"),
        ("last_game", 2),
        ("global_speed_limit_kmh", None),
        ("acc_enabled", False),
        ("short_increments", 1),
        ("long_increments", 5),
        ("long_press_reset", True),
        ("cc_start_button", "btn_start"),
        ("cc_inc_button", "btn_inc"),
        ("cc_dec_button", "btn_dec"),
    ):
        monkeypatch.setattr(settings, key, value)

    thread = CruiseControlThread()
    thread.running = True
    yield thread, tel, pedal, settings

    for name in ("telemetry_thread", "main_pedal_thread", "sending_thread"):
        registry.unregister(name)


_HELD_TO_BINDING = {
    "cc_dec_held": "cc_dec_button",
    "cc_inc_held": "cc_inc_button",
    "cc_start_held": "cc_start_button",
}


def _press(pedal, button: str, thread, ticks: int = 1) -> None:
    thread.loop()
    counts = dict(getattr(pedal.data, "cc_button_press_counts", None) or {})
    name = _HELD_TO_BINDING[button]
    counts[name] = counts.get(name, 0) + 1
    pedal.data.set(**{button: True, "cc_button_press_counts": counts})
    for _ in range(ticks):
        thread.loop()
    pedal.data.set(**{button: False})
    thread.loop()


def _engage(thread, mph: float) -> None:
    thread._cc_ctrl.enable()
    thread._cc_ctrl.set_target_kmh(mph * MPH_TO_KMH)


def test_short_inc_sets_cruise_to_the_current_mph(rig):
    thread, tel, pedal, _ = rig
    tel.data.set(speed=_ms(65.4))
    _press(pedal, "cc_inc_held", thread)
    assert thread._cc_ctrl.enabled is True
    assert thread._cc_ctrl.target_speed_kmh == pytest.approx(65 * MPH_TO_KMH)


def test_short_step_is_one_mph(rig):
    thread, _, pedal, _ = rig
    _engage(thread, 65)
    _press(pedal, "cc_inc_held", thread)
    assert thread._cc_ctrl.target_speed_kmh == pytest.approx(66 * MPH_TO_KMH)


def test_five_mph_snaps_to_the_grid(rig, monkeypatch):
    thread, _, pedal, settings = rig
    monkeypatch.setattr(settings, "short_increments", 5)
    _engage(thread, 65)
    _press(pedal, "cc_inc_held", thread)
    assert thread._cc_ctrl.target_speed_kmh == pytest.approx(70 * MPH_TO_KMH)

    _press(pedal, "cc_dec_held", thread)
    assert thread._cc_ctrl.target_speed_kmh == pytest.approx(65 * MPH_TO_KMH)


def test_set_speed_stops_at_80_mph(rig):
    thread, _, pedal, _ = rig
    _engage(thread, 80)
    _press(pedal, "cc_inc_held", thread)
    assert thread._cc_ctrl.target_speed_kmh == pytest.approx(80 * MPH_TO_KMH)


def test_slow_capture_floors_at_19_mph(rig):
    thread, tel, pedal, _ = rig
    tel.data.set(speed=_ms(10.0))
    _press(pedal, "cc_inc_held", thread)
    assert thread._cc_ctrl.target_speed_kmh == pytest.approx(19 * MPH_TO_KMH)


def test_global_limit_in_mph_stops_the_step(rig, monkeypatch):
    thread, _, pedal, settings = rig
    monkeypatch.setattr(settings, "global_speed_limit_kmh", 60 * MPH_TO_KMH)
    monkeypatch.setattr(settings, "short_increments", 5)
    _engage(thread, 60)
    _press(pedal, "cc_inc_held", thread)
    assert thread._cc_ctrl.target_speed_kmh == pytest.approx(60 * MPH_TO_KMH)
