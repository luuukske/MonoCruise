"""Low-speed strength scale: a fast truck stops hunting once the pull is visible."""
from __future__ import annotations

import math
from collections import deque

import pytest

from core.sending_thread.accel_to_pedals import AccelToPedals, weight_factor
from core.sending_thread.strength_scale import StrengthScale, speed_weight, update

DT = 0.02
OFFSET = 0.50
TRUE_SLOPE = 2.77
MODEL_CAP = 1.54
MASS = 20_000.0


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


def test_weight_is_full_in_a_pull_and_gone_at_cruise():
    assert speed_weight(5.0) == 1.0
    assert speed_weight(28.0) == 0.0
    assert 0.0 < speed_weight(14.0) < 1.0


def test_a_sag_does_not_raise_the_scale():
    state = StrengthScale()
    for _ in range(40):
        update(state, DT, 0.4, -0.6, 0.7, MODEL_CAP, 6.0, 6, 1.0, 0.0, OFFSET)
    assert state.scale == pytest.approx(1.0)


def _pull(clock, speed: float, seconds: float = 6.0):
    mapper = AccelToPedals()
    mapper._shared.slow_integral = 0.0
    wf = weight_factor(MASS, False)
    delay: deque[float] = deque([0.0] * 16, maxlen=16)
    meas = 0.0
    last = None
    tail_a: list[float] = []
    tail_g: list[float] = []
    ticks = int(seconds / DT)
    for i in range(ticks):
        clock.t += DT
        road = last.road_load_ms2 if last is not None else 0.7
        accel = OFFSET + TRUE_SLOPE * delay[0] - road
        meas += (1.0 - math.exp(-DT / 0.30)) * (accel - meas)
        last = mapper.step(
            0.45, meas, speed, MASS, False,
            max_accel_ms2=MODEL_CAP * wf,
            accel_offset_ms2=OFFSET,
            cruise_commanding=True,
            gear_dashboard=6,
            learn=True,
        )
        delay.appendleft(last.gas)
        if i > ticks * 2 // 3:
            tail_a.append(accel)
            tail_g.append(last.gas)
    assert last is not None
    return mapper, tail_a, tail_g


def test_a_strong_low_speed_pull_stops_hunting(clock):
    mapper, accels, gases = _pull(clock, speed=6.0)
    assert mapper._shared.strength.scale > 1.6
    mean_a = sum(accels) / len(accels)
    assert abs(mean_a - 0.45) < 0.25
    mid = sum(gases) / len(gases)
    std = math.sqrt(sum((g - mid) ** 2 for g in gases) / len(gases))
    assert std < 0.04


def test_highway_speed_does_not_learn_the_low_speed_scale(clock):
    mapper, _, _ = _pull(clock, speed=28.0, seconds=4.0)
    assert mapper._shared.strength.scale < 1.15
