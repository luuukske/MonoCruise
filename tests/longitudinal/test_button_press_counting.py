"""Short presses fire per counted press, not per observed level.

At a low polling rate the cruise thread samples slower than a tap is long, so a
tap can fall entirely between two of its ticks. Reproduces the 10 Hz case where
6 of 22 rapid taps were dropped. See core/cruise_control_thread/README.md.
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
    """CruiseControlThread in cruise mode, all CC buttons assigned."""
    tel = FakeThread("telemetry_thread", telemetry_data(speed=SPEED_KMH / 3.6))
    pedal = FakeThread("main_pedal_thread", pedal_data())
    sending = FakeThread("sending_thread", sending_data())
    for t in (tel, pedal, sending):
        registry.replace(t)

    settings = Settings.instance()
    for key, value in (
        ("cc_mode", "Cruise control"),
        ("global_speed_limit_kmh", 120.0),
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


def _engage(thread, target_kmh: float = TARGET_KMH) -> None:
    thread._cc_ctrl.enable()
    thread._cc_ctrl.set_target_kmh(target_kmh)
    thread.loop()


def _tap_unseen(pedal, thread, binding: str, count: int = 1) -> None:
    """Press and release entirely between two cruise ticks: level never True."""
    counts = dict(getattr(pedal.data, "cc_button_press_counts", None) or {})
    counts[binding] = counts.get(binding, 0) + count
    # The level is already back to False by the time this thread next looks.
    pedal.data.set(cc_button_press_counts=counts)
    thread.loop()


def test_tap_missed_by_the_level_still_steps_the_target(rig):
    thread, _, pedal, _, _ = rig
    _engage(thread)
    _tap_unseen(pedal, thread, "cc_dec_button")
    assert thread._cc_ctrl.target_speed_kmh == TARGET_KMH - 1


def test_several_taps_between_ticks_all_apply(rig):
    thread, _, pedal, _, _ = rig
    _engage(thread)
    _tap_unseen(pedal, thread, "cc_inc_button", count=6)
    assert thread._cc_ctrl.target_speed_kmh == TARGET_KMH + 6


def test_no_double_step_when_the_level_is_also_seen(rig):
    """A press the thread does observe must still count exactly once."""
    thread, _, pedal, _, _ = rig
    _engage(thread)
    counts = dict(pedal.data.cc_button_press_counts)
    counts["cc_dec_button"] = counts.get("cc_dec_button", 0) + 1
    pedal.data.set(cc_dec_held=True, cc_button_press_counts=counts)
    thread.loop()
    thread.loop()
    pedal.data.set(cc_dec_held=False)
    thread.loop()
    thread.loop()
    assert thread._cc_ctrl.target_speed_kmh == TARGET_KMH - 1


def test_presses_while_paused_are_dropped_not_replayed(rig):
    """Unpausing must not release a burst of queued speed changes."""
    thread, tel, pedal, _, _ = rig
    _engage(thread)
    tel.data.set(paused=True)
    counts = dict(pedal.data.cc_button_press_counts)
    counts["cc_inc_button"] = counts.get("cc_inc_button", 0) + 5
    pedal.data.set(cc_button_press_counts=counts)
    thread.loop()
    tel.data.set(paused=False)
    thread.loop()
    thread.loop()
    assert thread._cc_ctrl.target_speed_kmh == TARGET_KMH


def test_long_press_does_not_also_fire_a_short_step(rig, monkeypatch):
    """The press a long hold serviced must not fire again on release."""
    import time as time_mod

    from core.cruise_control_thread import thread as cc_mod

    clock = {"t": 1000.0}
    monkeypatch.setattr(time_mod, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(cc_mod.time, "monotonic", lambda: clock["t"])

    thread, _, pedal, _, _ = rig
    _engage(thread)
    counts = dict(pedal.data.cc_button_press_counts)
    counts["cc_inc_button"] = counts.get("cc_inc_button", 0) + 1
    pedal.data.set(cc_inc_held=True, cc_button_press_counts=counts)
    thread.loop()

    # Hold past the long-press threshold: one long step lands.
    clock["t"] += 0.4
    thread.loop()
    after_long = thread._cc_ctrl.target_speed_kmh
    assert after_long == TARGET_KMH + 5

    clock["t"] += 0.05
    pedal.data.set(cc_inc_held=False)
    thread.loop()
    thread.loop()
    assert thread._cc_ctrl.target_speed_kmh == after_long
