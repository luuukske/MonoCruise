"""Orchestrator invariants for Speed-limiter mode, driven through CruiseControlThread.loop().

Covers the always-on cap, the mid-session global-limit re-clamp (CC.step never
runs in this mode, so the orchestrator owns it), and the rule that limiter mode
is immune to the CC disengage conditions.
"""
from __future__ import annotations

import threading

import pytest

from core.cruise_control_thread.thread import CruiseControlThread
from core.settings import Settings
from core.thread_management.registry import registry


class _Data:
    def __init__(self, **fields):
        self._lock = threading.RLock()
        self.__dict__.update(fields)


class _FakeThread:
    """Minimal stand-in for a registered sibling thread."""

    def __init__(self, name: str, data: _Data):
        self.name = name
        self.data = data

    def is_alive(self) -> bool:
        return True


def _telemetry_data(**overrides) -> _Data:
    fields = dict(
        is_connected=True,
        paused=False,
        speed=80.0 / 3.6,
        gear_dashboard=1,
        parkBrake=False,
        gameClutch=0.0,
        gameThrottle=0.0,
        gameBrake=0.0,
        commanded_accel_ms2=0.0,
    )
    fields.update(overrides)
    return _Data(**fields)


def _pedal_data(**overrides) -> _Data:
    fields = dict(
        device_lost=False,
        em_stop=False,
        cc_dec_held=False,
        cc_inc_held=False,
        cc_start_held=False,
        acc_dist_inc_held=False,
        acc_dist_dec_held=False,
        brakeval=0.0,
        opdgasval=0.0,
    )
    fields.update(overrides)
    return _Data(**fields)


def _sending_data(**overrides) -> _Data:
    fields = dict(
        recent_brake_outputs=(0.0, 0.0, 0.0),
        auto_neutral_holding=False,
        user_gas_above_mapper=False,
        hold_active=False,
    )
    fields.update(overrides)
    return _Data(**fields)


@pytest.fixture
def rig(monkeypatch):
    """CruiseControlThread wired to fake siblings, in limiter mode at a 90 cap."""
    tel = _FakeThread("telemetry_thread", _telemetry_data())
    pedal = _FakeThread("main_pedal_thread", _pedal_data())
    sending = _FakeThread("sending_thread", _sending_data())
    for t in (tel, pedal, sending):
        registry.replace(t)

    settings = Settings.instance()
    monkeypatch.setattr(settings, "cc_mode", "Speed limiter")
    monkeypatch.setattr(settings, "global_speed_limit_kmh", 90.0)

    thread = CruiseControlThread()
    thread.running = True

    yield thread, tel, pedal, settings

    for name in ("telemetry_thread", "main_pedal_thread", "sending_thread"):
        registry.unregister(name)


def test_global_limit_alone_activates_the_limiter(rig):
    """No button press needed: the cap is always on in limiter mode."""
    thread, _, _, _ = rig
    thread.loop()
    assert thread._limiter_ctrl.active is True
    assert thread._limiter_ctrl.target_speed_kmh == 90.0
    assert thread.data.active_controller == "limiter"


def test_clearing_the_global_limit_deactivates_the_limiter(rig, monkeypatch):
    thread, _, _, settings = rig
    thread.loop()
    monkeypatch.setattr(settings, "global_speed_limit_kmh", None)
    thread.loop()
    assert thread._limiter_ctrl.active is False
    assert thread.data.active_controller == "none"


def test_tightening_the_limit_reclamps_a_stale_button_target(rig, monkeypatch):
    """The invariant: a stale higher button target must not survive the change."""
    thread, _, _, settings = rig
    thread._cc_ctrl.enable()
    thread._cc_ctrl.set_target_kmh(90.0)
    thread.loop()
    assert thread._limiter_ctrl.target_speed_kmh == 90.0

    monkeypatch.setattr(settings, "global_speed_limit_kmh", 70.0)
    thread.loop()
    assert thread._limiter_ctrl.target_speed_kmh == 70.0, (
        "stale button target still caps the truck above the new global limit"
    )


def test_limiter_survives_a_brake_press(rig):
    """Disengage conditions are CC-only; the limiter stays on through braking."""
    thread, tel, pedal, _ = rig
    thread._cc_ctrl.enable()
    thread._cc_ctrl.set_target_kmh(90.0)
    thread.loop()

    with tel.data._lock:
        tel.data.gameBrake = 1.0
    with pedal.data._lock:
        pedal.data.brakeval = 1.0
    for _ in range(5):
        thread.loop()

    assert thread._limiter_ctrl.active is True
    assert thread._cc_ctrl.enabled is True


@pytest.mark.parametrize("missing", ["telemetry_thread", "main_pedal_thread", "sending_thread"])
def test_a_missing_sibling_does_not_break_the_loop(rig, missing):
    """A thread must never crash or stop looping because a sibling is gone."""
    thread, _, _, _ = rig
    thread.loop()
    registry.unregister(missing)
    for _ in range(3):
        thread.loop()  # must not raise
    if missing == "sending_thread":
        # Only an optional reader is gone: the limiter keeps capping.
        assert thread._limiter_ctrl.active is True
    else:
        assert thread.data.active is False
