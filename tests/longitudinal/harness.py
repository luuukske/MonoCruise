"""Fakes for driving CruiseControlThread without real threads. See core/longitudinal/README.md."""
from __future__ import annotations

import threading

from core.longitudinal.base import LongCtx


def make_ctx(speed_ms: float, *, dt: float = 0.02, now: float = 0.0, **overrides) -> LongCtx:
    """Nominal driving context: connected, unpaused, no pedals, drive gear."""
    fields = dict(
        now=now,
        dt=dt,
        speed_ms=speed_ms,
        gear_dashboard=1,
        park_brake=False,
        game_throttle=0.0,
        game_clutch=0.0,
        game_brake=0.0,
        aeb_brake=False,
        connected=True,
        paused=False,
        em_stop=False,
        device_lost=False,
    )
    fields.update(overrides)
    return LongCtx(**fields)


class Data:
    """Stand-in for a ThreadData: plain attributes behind the same lock protocol."""

    def __init__(self, **fields):
        self._lock = threading.RLock()
        self.__dict__.update(fields)

    def set(self, **fields) -> None:
        with self._lock:
            self.__dict__.update(fields)


class FakeThread:
    """Minimal stand-in for a registered sibling thread."""

    def __init__(self, name: str, data: Data):
        self.name = name
        self.data = data

    def is_alive(self) -> bool:
        return True


def telemetry_data(**overrides) -> Data:
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
    return Data(**fields)


def pedal_data(**overrides) -> Data:
    fields = dict(
        device_lost=False,
        em_stop=False,
        cc_dec_held=False,
        cc_inc_held=False,
        cc_start_held=False,
        acc_dist_inc_held=False,
        acc_dist_dec_held=False,
        # Real main_pedal_thread publishes a rising-edge count per binding,
        # seeded at zero; short presses fire from it, not from the held level.
        cc_button_press_counts={
            "cc_start_button": 0,
            "cc_inc_button": 0,
            "cc_dec_button": 0,
            "acc_dist_inc_button": 0,
            "acc_dist_dec_button": 0,
        },
        brakeval=0.0,
        opdgasval=0.0,
    )
    fields.update(overrides)
    return Data(**fields)


def sending_data(**overrides) -> Data:
    fields = dict(
        recent_brake_outputs=(0.0, 0.0, 0.0),
        auto_neutral_holding=False,
        user_gas_above_mapper=False,
        hold_active=False,
        mapper_est_max_accel_ms2=0.0,
    )
    fields.update(overrides)
    return Data(**fields)
