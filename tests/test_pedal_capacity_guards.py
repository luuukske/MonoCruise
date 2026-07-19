"""Guards on the brake capacity estimate (pedal_capacity.py).

The estimate is AEB's engagement denominator: contamination that inflates it
silently disables emergency braking (crash clips ddc0cdf7 / 0fe85c88,
2026-07-10, estimate at 16.8-17.9 m/s2 vs a real ~7.8). These tests pin the
three guards: implausible-candidate rejection, the 1.3x baseline ceiling, and
the startup clamp on a poisoned persisted value.
"""

from __future__ import annotations

import pytest

import core.sending_thread.pedal_capacity as pc

BASE = 8.74


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


def _fresh() -> pc.PedalCapacityTracker:
    t = pc.PedalCapacityTracker()
    t._max_brake_ms2 = BASE
    return t


def _feed(t, clk, pedal, decel, ticks=60, dt=0.033, speed=20.0, aeb=False):
    for _ in range(ticks):
        clk.t += dt
        t.update_brake(pedal, decel, speed, 0.0, BASE, road_load_ms2=0.0,
                       aeb_active=aeb)


def test_contaminated_sample_rejected(clock):
    """Light pedal + retarder-inflated decel implies an impossible candidate."""
    t = _fresh()
    _feed(t, clock, pedal=0.3, decel=5.4)   # candidate = 18 m/s2
    assert t.max_brake_ms2 == pytest.approx(BASE)


def test_legit_strong_brake_still_learns(clock):
    t = _fresh()
    _feed(t, clock, pedal=0.9, decel=8.5)   # candidate = 9.44 m/s2
    assert BASE < t.max_brake_ms2 <= 9.45


def test_gentle_press_drifts_estimate_slowly(clock):
    """Soft presses still teach, but slowly: a 2 s gentle stop must barely
    move the estimate (candidate 4.0 vs BASE 8.74), so a stretch of abnormal
    braking cannot poison it (estimate fell 9 -> 4 m/s2 in routine driving,
    2026-07-19)."""
    t = _fresh()
    _feed(t, clock, pedal=0.3, decel=1.2, ticks=60)   # candidate = 4.0
    assert t.max_brake_ms2 == pytest.approx(BASE, rel=0.04)
    assert t.max_brake_ms2 < BASE  # direction still correct


def test_aeb_braking_learns_fast(clock):
    """The same underperforming stream during an AEB event re-teaches the
    estimate quickly (deep, honest presses)."""
    slow = _fresh()
    fast = _fresh()
    _feed(slow, clock, pedal=1.0, decel=4.0, ticks=60, aeb=False)
    _feed(fast, clock, pedal=1.0, decel=4.0, ticks=60, aeb=True)
    assert fast.max_brake_ms2 < slow.max_brake_ms2
    assert fast.max_brake_ms2 == pytest.approx(4.0, rel=0.15)


def test_estimate_ceiling(clock):
    t = _fresh()
    _feed(t, clock, pedal=1.0, decel=11.7, ticks=600)
    assert t.max_brake_ms2 <= BASE * pc._ESTIMATE_UPPER_BOUND + 1e-6


def test_settle_window_blocks_early_samples(clock):
    t = _fresh()
    clock.t += 1.0
    t.update_brake(0.0, 0.0, 20.0, 0.0, BASE)
    clock.t += 0.02
    t.update_brake(0.9, 8.5, 20.0, 0.0, BASE)   # pedal step
    _feed(t, clock, pedal=0.9, decel=8.5, ticks=13)   # 0.43 s span
    assert t.max_brake_ms2 == pytest.approx(BASE)
    _feed(t, clock, pedal=0.9, decel=8.5, ticks=10)   # crosses 0.5 s
    assert t.max_brake_ms2 > BASE


def test_load_persisted_clamps_poisoned_value(monkeypatch):
    class _FakeSettings:
        pedal_capacity_max_brake_ms2 = 17.9
        pedal_capacity_max_accel_ms2 = 2.0
        pedal_capacity_accel_anchor_gain_ms2 = 0.0
        pedal_capacity_accel_ratio_step = 0.0

    monkeypatch.setattr(pc, "Settings", _FakeSettings)
    t = pc.PedalCapacityTracker()
    t.load_persisted(BASE, 2.0)
    assert t.max_brake_ms2 <= BASE * pc._ESTIMATE_UPPER_BOUND + 1e-6

    _FakeSettings.pedal_capacity_max_brake_ms2 = 0.0
    t2 = pc.PedalCapacityTracker()
    t2.load_persisted(BASE, 2.0)
    assert t2.max_brake_ms2 == pytest.approx(BASE)
