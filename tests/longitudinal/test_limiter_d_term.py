"""SpeedLimiter D-term on low-passed speed. See core/longitudinal/README.md.

2026-10-21: telemetry speed steps at the game's ~20 Hz physics rate while the limiter
ticks at 100 Hz, so a tick-to-tick derivative spiked the bid on every physics step.
"""
from __future__ import annotations

import pytest

import core.longitudinal.limiter as limiter_mod
from core.longitudinal.limiter import SpeedLimiter
from tests.longitudinal.harness import make_ctx

DT = 0.01
STEP_TICKS = 5          # one physics step every 5 ticks: 20 Hz against 100 Hz
ACCEL = 0.25


class _DOnlySettings:
    limiter_kp = 0.0
    limiter_ki = 0.0
    limiter_kd = 0.25
    limiter_integral_clamp = 2.0
    limiter_accel_min_ms2 = -0.8


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    monkeypatch.setattr(limiter_mod, "Settings", _DOnlySettings)


def _staircase_bids(seconds: float = 3.0) -> list[float]:
    """Speed rising at ACCEL m/s2 in 20 Hz steps, well below a 90 km/h cap."""
    lim = SpeedLimiter()
    lim.set_target_kmh(90.0)
    lim.enable()
    speed, bids = 60.0 / 3.6, []
    for i in range(int(seconds / DT)):
        if i % STEP_TICKS == 0:
            speed += ACCEL * DT * STEP_TICKS
        bids.append(lim.step(make_ctx(speed, dt=DT, now=i * DT)).wanted_ms2)
    return bids[len(bids) // 2:]


def test_physics_steps_do_not_spike_the_bid(monkeypatch):
    smooth = _staircase_bids()
    monkeypatch.setattr(limiter_mod, "_KD_SPEED_SMOOTH_TAU_S", 0.0)
    raw = _staircase_bids()
    ripple = max(smooth) - min(smooth)
    raw_ripple = max(raw) - min(raw)
    assert raw_ripple > 0.2, "the staircase no longer reproduces the spike"
    assert ripple < 0.3 * raw_ripple


def test_the_damping_survives_the_low_pass(monkeypatch):
    """Same mean D-term: the filter removes the chatter, not the damping."""
    smooth = _staircase_bids()
    monkeypatch.setattr(limiter_mod, "_KD_SPEED_SMOOTH_TAU_S", 0.0)
    raw = _staircase_bids()
    assert sum(smooth) / len(smooth) < 0.0
    assert sum(smooth) / len(smooth) == pytest.approx(sum(raw) / len(raw), rel=0.10)
