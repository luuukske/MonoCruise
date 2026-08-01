"""Cruise-mode orchestration: user-gas override latch, disengage conditions, button FSM.

Driven through CruiseControlThread.loop() against fake siblings. Everything the
orchestrator reads from Settings is pinned here so the tests do not depend on
whatever config.json happens to hold.
"""
from __future__ import annotations

import pytest

from core.cruise_control_thread.thread import CruiseControlThread
from core.settings import Settings
from core.thread_management.registry import registry
from tests.longitudinal.harness import (
    FakeThread,
    pedal_data,
    sending_data,
    telemetry_data,
)

SPEED_KMH = 80.0
TARGET_KMH = 80.0


@pytest.fixture
def rig(monkeypatch):
    """CruiseControlThread in cruise mode, CC engaged at 80, global limit 90."""
    tel = FakeThread("telemetry_thread", telemetry_data(speed=SPEED_KMH / 3.6))
    pedal = FakeThread("main_pedal_thread", pedal_data())
    sending = FakeThread("sending_thread", sending_data())
    for t in (tel, pedal, sending):
        registry.replace(t)

    settings = Settings.instance()
    for key, value in (
        ("cc_mode", "Cruise control"),
        ("global_speed_limit_kmh", 90.0),
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
    yield thread, tel, pedal, sending, settings

    for name in ("telemetry_thread", "main_pedal_thread", "sending_thread"):
        registry.unregister(name)


def _engage_cc(thread, target_kmh: float = TARGET_KMH) -> None:
    thread._cc_ctrl.enable()
    thread._cc_ctrl.set_target_kmh(target_kmh)


def _press(pedal, button: str, thread, ticks: int = 1) -> None:
    pedal.data.set(**{button: True})
    for _ in range(ticks):
        thread.loop()
    pedal.data.set(**{button: False})
    thread.loop()


def test_cc_bids_and_is_labelled_cc(rig):
    thread, *_ = rig
    _engage_cc(thread)
    thread.loop()
    assert thread.data.active is True
    assert thread.data.active_controller == "cc"


def test_user_gas_override_latches_and_hands_the_cap_to_the_limiter(rig):
    """Without this handover a user gas override bypasses the global limit."""
    thread, _, pedal, sending, _ = rig
    _engage_cc(thread)
    thread.loop()

    pedal.data.set(opdgasval=0.9)
    sending.data.set(user_gas_above_mapper=True)
    thread.loop()

    assert thread._cc_user_override is True
    assert thread.data.active_controller == "limiter"


def test_no_latch_without_the_sending_thread_flag(rig):
    """Gas alone is not an override: the mapper must actually have been beaten."""
    thread, _, pedal, _, _ = rig
    _engage_cc(thread)
    thread.loop()

    pedal.data.set(opdgasval=0.9)
    thread.loop()
    assert thread._cc_user_override is False
    assert thread.data.active_controller == "cc"


def test_no_latch_while_well_below_the_cc_target(rig):
    """Below target the driver is catching up to CC, not overriding it."""
    thread, tel, pedal, sending, _ = rig
    _engage_cc(thread, TARGET_KMH)
    tel.data.set(speed=(TARGET_KMH - 20.0) / 3.6)
    pedal.data.set(opdgasval=0.9)
    sending.data.set(user_gas_above_mapper=True)
    thread.loop()
    assert thread._cc_user_override is False


def test_latch_releases_when_the_driver_lifts_off(rig):
    thread, _, pedal, sending, _ = rig
    _engage_cc(thread)
    thread.loop()
    pedal.data.set(opdgasval=0.9)
    sending.data.set(user_gas_above_mapper=True)
    thread.loop()
    assert thread._cc_user_override is True

    pedal.data.set(opdgasval=0.0)
    thread.loop()
    assert thread._cc_user_override is False
    assert thread.data.active_controller == "cc"


def test_latch_releases_when_speed_falls_below_the_target(rig):
    thread, tel, pedal, sending, _ = rig
    _engage_cc(thread)
    thread.loop()
    pedal.data.set(opdgasval=0.9)
    sending.data.set(user_gas_above_mapper=True)
    thread.loop()
    assert thread._cc_user_override is True

    tel.data.set(speed=(TARGET_KMH - 10.0) / 3.6)
    thread.loop()
    assert thread._cc_user_override is False


def test_no_latch_without_an_active_limiter(rig, monkeypatch):
    """With no global limit there is nothing to hand the cap over to."""
    thread, _, pedal, sending, settings = rig
    monkeypatch.setattr(settings, "global_speed_limit_kmh", None)
    _engage_cc(thread)
    thread.loop()

    pedal.data.set(opdgasval=0.9)
    sending.data.set(user_gas_above_mapper=True)
    thread.loop()
    assert thread._cc_user_override is False


def test_user_brake_disengages_cc(rig):
    thread, _, pedal, _, _ = rig
    _engage_cc(thread)
    thread.loop()

    pedal.data.set(brakeval=0.5)
    thread.loop()
    assert thread._cc_ctrl.enabled is False


def test_the_mappers_own_commanded_brake_does_not_disengage_cc(rig):
    """CC braking through the mapper must not read as the driver braking."""
    thread, tel, _, sending, _ = rig
    _engage_cc(thread)
    thread.loop()

    tel.data.set(gameBrake=0.6)
    sending.data.set(recent_brake_outputs=(0.6, 0.6, 0.6))
    thread.loop()
    assert thread._cc_ctrl.enabled is True

    sending.data.set(recent_brake_outputs=(0.0, 0.0, 0.0))
    thread.loop()
    assert thread._cc_ctrl.enabled is False


def test_park_brake_disengages_cc(rig):
    thread, tel, _, _, _ = rig
    _engage_cc(thread)
    thread.loop()

    tel.data.set(parkBrake=True)
    thread.loop()
    assert thread._cc_ctrl.enabled is False


def test_a_crash_then_a_stop_disarms_cc(rig):
    """Speed collapse followed by standstill: do not resume on the driver's behalf."""
    thread, tel, _, _, _ = rig
    _engage_cc(thread)
    thread.loop()

    tel.data.set(speed=SPEED_KMH / 3.6 - 8.0)
    thread.loop()
    assert thread._cc_ctrl.enabled is True, "disarm must wait for the stop"

    tel.data.set(speed=0.05)
    thread.loop()
    assert thread._cc_ctrl.enabled is False


def test_limiter_mode_ignores_the_disengage_conditions(rig, monkeypatch):
    thread, _, pedal, _, settings = rig
    monkeypatch.setattr(settings, "cc_mode", "Speed limiter")
    _engage_cc(thread)
    thread.loop()

    pedal.data.set(brakeval=1.0)
    for _ in range(3):
        thread.loop()
    assert thread._cc_ctrl.enabled is True
    assert thread._limiter_ctrl.active is True


def test_short_inc_press_enables_cc_at_the_current_speed(rig):
    thread, _, pedal, _, _ = rig
    assert thread._cc_ctrl.enabled is False
    _press(pedal, "cc_inc_held", thread)
    assert thread._cc_ctrl.enabled is True
    assert thread._cc_ctrl.target_speed_kmh == pytest.approx(round(SPEED_KMH))


def test_short_inc_press_steps_by_short_increments(rig):
    thread, _, pedal, _, _ = rig
    _engage_cc(thread)
    _press(pedal, "cc_inc_held", thread)
    assert thread._cc_ctrl.target_speed_kmh == TARGET_KMH + 1


def test_short_dec_press_steps_down(rig):
    thread, _, pedal, _, _ = rig
    _engage_cc(thread)
    _press(pedal, "cc_dec_held", thread)
    assert thread._cc_ctrl.target_speed_kmh == TARGET_KMH - 1


def test_start_press_toggles_cc(rig):
    thread, _, pedal, _, _ = rig
    _engage_cc(thread)
    _press(pedal, "cc_start_held", thread)
    assert thread._cc_ctrl.enabled is False
    _press(pedal, "cc_start_held", thread)
    assert thread._cc_ctrl.enabled is True


def test_inc_is_blocked_with_the_park_brake_on(rig):
    thread, tel, pedal, _, _ = rig
    tel.data.set(parkBrake=True)
    _press(pedal, "cc_inc_held", thread)
    assert thread._cc_ctrl.enabled is False


def test_buttons_are_ignored_while_the_pedal_device_is_lost(rig):
    thread, _, pedal, _, _ = rig
    pedal.data.set(device_lost=True)
    _press(pedal, "cc_inc_held", thread)
    assert thread._cc_ctrl.enabled is False


def test_unassigned_buttons_do_not_engage_cc(rig, monkeypatch):
    thread, _, pedal, _, settings = rig
    monkeypatch.setattr(settings, "cc_inc_button", None)
    _press(pedal, "cc_inc_held", thread)
    assert thread._cc_ctrl.enabled is False
