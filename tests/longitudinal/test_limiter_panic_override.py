"""Panic bypass: a held floor stays capped, a quick lift-and-press does not.

The limiter is the safety net for a limit set lower than the driver meant.
Flooring into that limit must still stop there. The bypass is a deliberate blip
at the cap, and it has to give the pedal back afterwards.
"""
from __future__ import annotations

import pytest

from core.cruise_control_thread.thread import CruiseControlThread
from core.longitudinal.limiter_override import LimiterPanicOverride
from core.settings import Settings
from core.thread_management.registry import registry
from tests.longitudinal.harness import FakeThread, pedal_data, sending_data, telemetry_data

LIMIT = 80.0
DT = 0.02


def _step(
    ov: LimiterPanicOverride,
    gas: float,
    speed: float,
    *,
    dt: float = DT,
    n: int = 1,
    limit: float | None = LIMIT,
    active: bool = True,
) -> bool:
    out = False
    for _ in range(n):
        out = ov.update(
            gas=gas,
            speed_kmh=speed,
            limit_kmh=limit,
            limiter_active=active,
            dt=dt,
        )
    return out


def _hold(ov: LimiterPanicOverride, speed: float = LIMIT) -> None:
    assert _step(ov, 1.0, speed, n=25) is False  # 0.50 s, past the arm hold


def _blip(ov: LimiterPanicOverride, speed: float = LIMIT) -> bool:
    _step(ov, 0.0, speed, n=8)  # 0.16 s off, a stab back down
    return _step(ov, 1.0, speed)


def test_holding_the_floor_at_the_cap_never_bypasses():
    ov = LimiterPanicOverride()
    assert _step(ov, 1.0, LIMIT + 15.0, n=250) is False
    assert ov.overridden is False


def test_a_quick_lift_and_press_at_the_cap_bypasses():
    ov = LimiterPanicOverride()
    _hold(ov, LIMIT + 10.0)
    assert _blip(ov, LIMIT + 10.0) is True


def test_a_fast_stab_works_on_the_way_into_the_cap():
    """Floor time on the approach counts. The stab itself still has to be fast."""
    ov = LimiterPanicOverride()
    _hold(ov, LIMIT - 30.0)
    assert _blip(ov, LIMIT) is True


def test_a_fast_small_stab_bypasses():
    """A short flick qualifies. The pedal does not have to go to the stop or to zero."""
    ov = LimiterPanicOverride()
    _hold(ov)
    assert _step(ov, 0.45, LIMIT, n=4) is False  # 0.08 s across the minimum lift
    assert _step(ov, 0.80, LIMIT) is True


def test_the_same_small_lift_is_rejected_when_it_is_slow():
    ov = LimiterPanicOverride()
    _hold(ov)
    assert _step(ov, 0.60, LIMIT, n=12) is False  # 0.24 s, same gap, too slow
    _step(ov, 0.45, LIMIT, n=3)
    assert _step(ov, 0.80, LIMIT) is False


def test_a_blip_well_below_the_cap_does_not_bypass():
    ov = LimiterPanicOverride()
    _hold(ov, LIMIT - 30.0)
    assert _blip(ov, LIMIT - 30.0) is False
    assert ov.overridden is False


def test_a_below_cap_blip_does_not_leave_the_pedal_armed():
    ov = LimiterPanicOverride()
    _hold(ov, LIMIT - 30.0)
    _blip(ov, LIMIT - 30.0)
    _step(ov, 1.0, LIMIT, n=2)
    assert _blip(ov, LIMIT) is False


def test_a_slow_lift_then_press_does_not_bypass():
    ov = LimiterPanicOverride()
    _hold(ov)
    assert _step(ov, 0.6, LIMIT, n=25) is False  # 0.50 s, past the quick-release edge
    _step(ov, 0.0, LIMIT, n=5)
    assert _step(ov, 1.0, LIMIT) is False


def test_a_press_that_was_not_held_does_not_arm():
    ov = LimiterPanicOverride()
    _step(ov, 1.0, LIMIT, n=18)  # 0.36 s, under the arm hold
    assert _blip(ov) is False


def test_a_merely_quick_lift_is_not_a_panic_stab():
    ov = LimiterPanicOverride()
    _hold(ov)
    assert _step(ov, 0.0, LIMIT, n=20) is False  # 0.40 s, past the stab window
    assert _step(ov, 1.0, LIMIT) is False


def test_a_one_sample_dip_is_not_a_blip_and_does_not_drop_the_arm():
    ov = LimiterPanicOverride()
    _hold(ov)
    _step(ov, 0.0, LIMIT, n=1)  # 0.02 s, under the minimum lift
    assert _step(ov, 1.0, LIMIT) is False
    assert _blip(ov) is True


def test_returning_only_part_way_does_not_bypass():
    ov = LimiterPanicOverride()
    _hold(ov)
    _step(ov, 0.0, LIMIT, n=8)
    assert _step(ov, 0.5, LIMIT, n=10) is False
    assert ov.overridden is False


def test_a_lift_held_past_the_window_does_not_bypass():
    ov = LimiterPanicOverride()
    _hold(ov)
    assert _step(ov, 0.0, LIMIT, n=40) is False  # 0.80 s, past the window
    assert _step(ov, 1.0, LIMIT) is False


def test_a_floored_pedal_keeps_the_bypass():
    ov = LimiterPanicOverride()
    _hold(ov, LIMIT + 20.0)
    assert _blip(ov, LIMIT + 20.0) is True
    assert _step(ov, 1.0, LIMIT + 25.0, n=200) is True


def test_partial_throttle_keeps_the_bypass():
    ov = LimiterPanicOverride()
    _hold(ov, LIMIT + 20.0)
    assert _blip(ov, LIMIT + 20.0) is True
    assert _step(ov, 0.55, LIMIT + 25.0, n=200) is True


def test_staying_off_the_pedal_does_not_restore_the_limiter():
    ov = LimiterPanicOverride()
    _hold(ov)
    assert _blip(ov) is True
    assert _step(ov, 0.0, LIMIT + 5.0, n=200) is True


def test_a_stab_at_the_cap_holds_until_you_fall_back_through_5_above():
    ov = LimiterPanicOverride()
    _hold(ov, LIMIT)
    assert _blip(ov, LIMIT) is True
    assert _step(ov, 1.0, LIMIT, n=40) is True
    assert _step(ov, 1.0, LIMIT + 6.0) is True
    assert _step(ov, 1.0, LIMIT + 5.0) is True
    assert _step(ov, 1.0, LIMIT + 4.9) is False


def test_slowing_below_5_above_the_cap_restores_the_limiter():
    ov = LimiterPanicOverride()
    _hold(ov, LIMIT + 10.0)
    assert _blip(ov, LIMIT + 10.0) is True
    assert _step(ov, 1.0, LIMIT + 5.0) is True
    assert _step(ov, 1.0, LIMIT + 4.9) is False


def test_a_stalled_tick_cannot_finish_the_hold_by_itself():
    ov = LimiterPanicOverride()
    for _ in range(4):
        _step(ov, 1.0, LIMIT, dt=5.0)
    assert _blip(ov) is False


def test_the_latch_clears_when_the_limiter_is_gone():
    ov = LimiterPanicOverride()
    _hold(ov)
    assert _blip(ov) is True
    assert _step(ov, 1.0, LIMIT, active=False) is False
    assert ov.overridden is False


@pytest.fixture
def limiter_rig(monkeypatch):
    tel = FakeThread("telemetry_thread", telemetry_data(speed=LIMIT / 3.6))
    pedal = FakeThread("main_pedal_thread", pedal_data())
    sending = FakeThread("sending_thread", sending_data())
    for t in (tel, pedal, sending):
        registry.replace(t)

    settings = Settings.instance()
    monkeypatch.setattr(settings, "cc_mode", "Speed limiter")
    monkeypatch.setattr(settings, "global_speed_limit_kmh", LIMIT)
    monkeypatch.setattr(settings, "last_game", 1)

    thread = CruiseControlThread()
    thread.running = True
    yield thread, tel, pedal

    for name in ("telemetry_thread", "main_pedal_thread", "sending_thread"):
        registry.unregister(name)


def _clock(monkeypatch):
    import time as time_mod

    from core.cruise_control_thread import thread as cc_mod

    clock = {"t": 10_000.0}
    monkeypatch.setattr(time_mod, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(cc_mod.time, "monotonic", lambda: clock["t"])
    return clock


def _advance(thread, clock, seconds: float, step: float = 0.05) -> None:
    n = int(round(seconds / step))
    for _ in range(n):
        clock["t"] += step
        thread.loop()


def test_orchestrator_holds_a_floored_pedal_at_the_cap(limiter_rig, monkeypatch):
    thread, tel, pedal = limiter_rig
    clock = _clock(monkeypatch)
    tel.data.set(speed=(LIMIT + 20.0) / 3.6)
    pedal.data.set(opdgasval=1.0)
    thread.loop()
    _advance(thread, clock, 1.0)
    assert thread._limiter_panic.overridden is False
    assert thread.data.active_controller == "limiter"
    assert thread.data.wanted_accel_ms2 < 0.0
    assert thread.data.wanted_accel_ms2 >= -1.0 - 1e-9


def test_orchestrator_blip_drops_the_bid_and_restores_on_slowing(limiter_rig, monkeypatch):
    thread, tel, pedal = limiter_rig
    clock = _clock(monkeypatch)
    popups: list[tuple] = []

    def _emit(title, message, message_type, duration_ms=5000, priority=0):
        popups.append((title, message, message_type, priority))

    monkeypatch.setattr("ui.popup.popup_window.PopupWindow.emit", _emit)
    tel.data.set(speed=(LIMIT + 10.0) / 3.6)
    pedal.data.set(opdgasval=1.0)
    thread.loop()
    _advance(thread, clock, 0.80)

    pedal.data.set(opdgasval=0.0)
    _advance(thread, clock, 0.16)
    pedal.data.set(opdgasval=1.0)
    _advance(thread, clock, 0.05)

    assert thread._limiter_panic.overridden is True
    assert thread.data.active is False
    assert thread.data.wanted_accel_ms2 == pytest.approx(0.0)

    tel.data.set(speed=(LIMIT - 8.0) / 3.6)
    _advance(thread, clock, 0.05)
    assert thread._limiter_panic.overridden is False
    assert thread.data.active_controller == "limiter"
    assert popups == [("Limiter bypassed", "Slow down to restore", "w", 2)]


def test_orchestrator_blip_in_cruise_mode_drops_cc_without_disabling_it(monkeypatch):
    tel = FakeThread("telemetry_thread", telemetry_data(speed=(LIMIT + 5.0) / 3.6))
    pedal = FakeThread("main_pedal_thread", pedal_data(opdgasval=1.0))
    sending = FakeThread("sending_thread", sending_data())
    for t in (tel, pedal, sending):
        registry.replace(t)
    settings = Settings.instance()
    monkeypatch.setattr(settings, "cc_mode", "Cruise control")
    monkeypatch.setattr(settings, "global_speed_limit_kmh", LIMIT)
    monkeypatch.setattr(settings, "acc_enabled", False)
    monkeypatch.setattr(settings, "last_game", 1)
    monkeypatch.setattr(settings, "cc_start_button", "btn_start")
    monkeypatch.setattr(settings, "cc_inc_button", "btn_inc")
    monkeypatch.setattr(settings, "cc_dec_button", "btn_dec")

    thread = CruiseControlThread()
    thread.running = True
    thread._cc_ctrl.enable()
    thread._cc_ctrl.set_target_kmh(LIMIT)
    try:
        clock = _clock(monkeypatch)
        thread.loop()
        _advance(thread, clock, 0.80)
        pedal.data.set(opdgasval=0.0)
        _advance(thread, clock, 0.16)
        pedal.data.set(opdgasval=1.0)
        _advance(thread, clock, 0.05)
        assert thread._limiter_panic.overridden is True
        assert thread._cc_ctrl.enabled is True
        assert thread.data.active is False
    finally:
        for name in ("telemetry_thread", "main_pedal_thread", "sending_thread"):
            registry.unregister(name)
