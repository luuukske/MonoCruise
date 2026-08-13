"""Mapper road-load FF: grade smoothing and launch invert clamp."""
from __future__ import annotations

import math

import pytest

from core.sending_thread import accel_to_pedals as atp
from core.sending_thread.accel_to_pedals import AccelToPedals
from core.settings import Settings

DT = 0.02
HIGHWAY_MS = 28.0
MASS_KG = 24_000.0
DOWN_RAD = -0.04
STEEP_RAD = -0.11


class _FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def clock(monkeypatch):
    clk = _FakeClock()
    monkeypatch.setattr("core.sending_thread.accel_to_pedals.time.monotonic", clk)
    return clk


@pytest.fixture()
def crr():
    inst = Settings.instance()
    old = inst.mapper_rolling_resistance
    inst.mapper_rolling_resistance = 0.07
    try:
        yield 0.07
    finally:
        inst.mapper_rolling_resistance = old


def _pitch_norm(rad: float) -> float:
    return rad / (2.0 * math.pi)


def _step(
    mapper: AccelToPedals,
    clock: _FakeClock,
    pitch_rad: float,
    wanted: float = 0.0,
    raw: float = 0.0,
    speed: float = HIGHWAY_MS,
):
    clock.t += DT
    return mapper.step(
        wanted,
        raw,
        speed,
        MASS_KG,
        True,
        cruise_commanding=True,
        road_pitch=_pitch_norm(pitch_rad),
        gear_dashboard=12,
        learn=True,
    )


def test_tiny_grade_error_stays_on_the_slow_tau():
    mapper = AccelToPedals()
    slow = mapper._ema_alpha(DT, atp._GRADE_SMOOTH_TAU_S)
    fast = mapper._ema_alpha(DT, atp._GRADE_FAST_TAU_S)
    alpha = mapper._grade_ema_alpha(DT, 0.03)
    assert abs(alpha - slow) < 0.15 * abs(fast - slow)


def test_large_grade_error_blends_toward_the_fast_tau():
    mapper = AccelToPedals()
    slow = mapper._ema_alpha(DT, atp._GRADE_SMOOTH_TAU_S)
    fast = mapper._ema_alpha(DT, atp._GRADE_FAST_TAU_S)
    alpha = mapper._grade_ema_alpha(DT, 0.40)
    assert (alpha - slow) > 0.85 * (fast - slow)


def test_pitch_noise_does_not_move_grade_ff_much(crr, clock):
    mapper = AccelToPedals()
    mapper._shared.slow_integral = 0.0
    _step(mapper, clock, 0.0)
    baseline = mapper._shared.grade_accel_smooth
    wobble = 0.003
    for i in range(40):
        _step(mapper, clock, wobble if i % 2 == 0 else -wobble)
    assert abs(mapper._shared.grade_accel_smooth - baseline) < 0.04


def test_a_real_hill_still_tracks_within_a_second(crr, clock):
    mapper = AccelToPedals()
    mapper._shared.slow_integral = 0.0
    _step(mapper, clock, 0.0)
    target = mapper._road_load_parts(
        HIGHWAY_MS, 0.0, _pitch_norm(STEEP_RAD), 12,
    )[1]
    tracked = None
    for _ in range(40):
        tracked = _step(mapper, clock, STEEP_RAD)
    assert tracked is not None
    # Last 10% of the step is a small error, so the slow tau owns it on purpose.
    assert mapper._shared.grade_accel_smooth / target > 0.85


def test_positive_wanted_on_a_steep_descent_does_not_brake(crr, clock):
    """11% downhill launch: gravity FF must not invert a +1 m/s2 bid into brake."""
    mapper = AccelToPedals()
    targets = None
    for _ in range(10):
        targets = _step(mapper, clock, STEEP_RAD, wanted=1.0, speed=0.5)
    assert targets is not None
    assert targets.brake == pytest.approx(0.0, abs=1e-6)
    assert targets.gas > 0.0


def test_climb_leftover_slow_i_bleeds_on_a_crest(crr, clock):
    mapper = AccelToPedals()
    mapper._shared.slow_integral = 0.66
    _step(mapper, clock, 0.0, wanted=0.0, raw=0.15)
    for _ in range(50):
        _step(mapper, clock, DOWN_RAD, wanted=0.0, raw=0.15)
    assert mapper._shared.slow_integral < 0.25
    assert mapper._shared.slow_integral > 0.0
