"""Gas-side capacity learning on a light truck holding speed. See core/sending_thread/README.md.

2026-10-20: a 10 t truck cruising at 100 km/h had 0.1-0.3 m/s2 of authority against
0.4-1.1 of modelled road load; the anchor fell 2.56 -> 0.59 and CC pumped the gas.
"""
from __future__ import annotations

import pytest

import core.sending_thread.pedal_capacity as pc


class _FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def clock(monkeypatch):
    clk = _FakeClock()
    monkeypatch.setattr(pc.time, "monotonic", clk)
    return clk


@pytest.fixture(autouse=True)
def _no_settings_io(monkeypatch):
    monkeypatch.setattr(pc.PedalCapacityTracker, "_maybe_save", lambda self, now: None)

def _light_truck(anchor=2.4, ratio=1.067, offset=0.69):
    t = pc.PedalCapacityTracker()
    t._accel_anchor_gain_ms2, t._accel_ratio_step, t._accel_zero_offset_ms2 = anchor, ratio, offset
    return t


def _hold_speed(t, clk, pedal, accel, seconds, gear=14, mass=10_300.0, road=0.7):
    for _ in range(int(seconds / 0.01)):
        clk.t += 0.01
        t.update_accel(pedal, accel, 27.8, 0.0, 0.0, gear, mass, False, road_load_ms2=road)


def test_a_light_truck_holding_speed_does_not_read_as_weak(clock):
    """2026-10-20: 0.1 m/s2 of authority at 100 km/h took the anchor 2.56 -> 0.59 in one burst."""
    t = _light_truck()
    _hold_speed(t, clock, 0.45, 0.12, seconds=30.0)
    assert t._accel_anchor_gain_ms2 > 0.9 * 2.4


def test_one_sample_moves_the_anchor_a_bounded_step(clock):
    t = _light_truck()
    _hold_speed(t, clock, 0.6, 0.0, seconds=1.0)  # fill the settle windows
    before = t._accel_anchor_gain_ms2
    clock.t += 0.01
    t.update_accel(0.6, 0.35, 27.8, 0.0, 0.0, 14, 10_300.0, False, road_load_ms2=0.7)
    step = abs(t._accel_anchor_gain_ms2 / before - 1.0)
    assert step <= pc._ACCEL_BASE_ALPHA * pc._UNDERPERFORM_MULT * pc._ACCEL_RESIDUAL_MAX * 1.05


def test_a_real_pull_still_teaches_a_weaker_truck(clock):
    """The floor must not starve learning: a firm pull with clear authority moves the anchor."""
    t = _light_truck(anchor=4.0)
    _hold_speed(t, clock, 0.9, 0.9, seconds=20.0, gear=8, mass=20_000.0, road=0.3)
    assert t._accel_anchor_gain_ms2 < 3.6
