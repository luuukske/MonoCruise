"""Pedal dead man: a stalled sending thread must not leave the gas on or the driver without brake.

2026-11-20: the sending thread froze four times with the driver's gas at 95 %, and the game kept
that gas while the driver braked (core/sending_thread/README.md, pedal dead man).
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import core.sending_thread.deadman as dm
from core.scs_profile.intensity import apply_brake_intensity


class _Thread:
    def __init__(self, heartbeat_at: float, loop_interval: float = 0.01, alive: bool = True) -> None:
        self.heartbeat_at = heartbeat_at
        self.loop_interval = loop_interval
        self.running = alive
        self._alive = alive

    def is_alive(self) -> bool:
        return self._alive


class _Controller:
    def __init__(self, gas: float, brake: float) -> None:
        self.aforward = gas
        self.abackward = brake

    def close(self) -> None:
        pass


def _pedal(brake_output=0.0, brakeval=0.0, em_stop=False):
    data = SimpleNamespace(brake_output=brake_output, brakeval=brakeval, em_stop=em_stop,
                           _lock=threading.Lock())
    return SimpleNamespace(data=data, is_alive=lambda: True)


def _telemetry(connected=True):
    return SimpleNamespace(data=SimpleNamespace(is_connected=connected), is_alive=lambda: True)


@pytest.fixture
def world(monkeypatch):
    threads: dict[str, object] = {"telemetry_thread": _telemetry(), "main_pedal_thread": _pedal()}

    def get_thread(name):
        if name not in threads:
            raise KeyError(name)
        return threads[name]

    monkeypatch.setattr(dm.registry, "get_thread", get_thread)
    monkeypatch.setattr(dm, "last_brake_intensity", lambda: 1.1)
    clock = {"t": 100.0}
    monkeypatch.setattr(dm.time, "monotonic", lambda: clock["t"])
    deadman = dm.PedalDeadmanThread()
    deadman.running = True
    controller = _Controller(gas=0.95, brake=0.0)
    deadman._controller = controller
    return SimpleNamespace(threads=threads, clock=clock, deadman=deadman, controller=controller)


def test_state_follows_the_sending_heartbeat_scaled_by_its_own_loop():
    assert dm.sending_state(_Thread(99.9), True, False, 100.0) == dm.FRESH
    assert dm.sending_state(_Thread(99.6), True, True, 100.0) == dm.STALE
    assert dm.sending_state(_Thread(99.6), True, False, 100.0) == dm.IDLE, "never seen writing: startup"
    assert dm.sending_state(_Thread(98.0, loop_interval=1.0), True, True, 100.0) == dm.FRESH, \
        "a 1 Hz disconnected loop is not a freeze"
    assert dm.sending_state(_Thread(99.9), False, True, 100.0) == dm.IDLE
    assert dm.sending_state(_Thread(0.0), True, True, 100.0) == dm.STALE, "a replacement still in setup"
    assert dm.sending_state(None, True, True, 100.0) == dm.STALE


def test_driver_brake_goes_through_the_slider_remap_and_a_slam_keeps_full_authority():
    assert dm.driver_brake_sent(0.0, 0.4, False, 1.0, 2.2) == pytest.approx(apply_brake_intensity(0.4, 2.2))
    assert dm.driver_brake_sent(0.0, 0.5, False, 2.0, 1.1) == pytest.approx(0.25)
    assert dm.driver_brake_sent(1.0, 0.9, True, 1.0, 3.0) == pytest.approx(1.0)


def test_a_frozen_sending_thread_has_its_gas_released_and_the_driver_brake_passed(world):
    sending = _Thread(heartbeat_at=100.0)
    world.threads["sending_thread"] = sending
    world.deadman.loop()
    assert world.controller.aforward == pytest.approx(0.95), "fresh: the dead man never writes"

    world.clock["t"] = 100.5
    world.threads["main_pedal_thread"] = _pedal(brakeval=0.6)
    world.deadman.loop()
    assert world.controller.aforward == 0.0
    assert world.controller.abackward == pytest.approx(0.6)
    assert world.deadman.data.engaged and world.deadman.data.engagements == 1

    world.controller.aforward = 0.4
    sending.heartbeat_at = 100.5
    world.deadman.loop()
    assert world.controller.aforward == pytest.approx(0.4), "back to the sending thread's own output"
    assert not world.deadman.data.engaged


def test_a_brake_held_at_the_freeze_is_kept_while_the_thread_lives_and_dropped_once_it_is_gone(world):
    sending = _Thread(heartbeat_at=100.0)
    world.threads["sending_thread"] = sending
    world.controller.abackward = 0.7
    world.deadman.loop()

    world.clock["t"] = 101.0
    world.controller.abackward = 0.0
    world.deadman.loop()
    assert world.controller.abackward == pytest.approx(0.7), "hold or AEB brake from before the freeze"

    sending._alive = False
    sending.running = False
    world.threads["main_pedal_thread"] = _pedal(brakeval=0.2)
    world.deadman.loop()
    assert world.controller.abackward == pytest.approx(0.2), "a dead thread never resumes its hold"
    assert world.controller.aforward == 0.0


def test_nothing_is_written_before_the_sending_thread_ever_ran_or_without_the_game(world):
    world.threads["sending_thread"] = _Thread(heartbeat_at=0.0)
    world.deadman.loop()
    assert world.controller.aforward == pytest.approx(0.95)

    world.threads["sending_thread"] = _Thread(heartbeat_at=100.0)
    world.deadman.loop()
    world.threads["telemetry_thread"] = _telemetry(connected=False)
    world.clock["t"] = 110.0
    world.deadman.loop()
    assert world.controller.aforward == pytest.approx(0.95)
