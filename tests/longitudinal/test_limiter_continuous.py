"""Invariant: SpeedLimiter is a continuous tracker, not an over-limit reactor.

An "only when over the limit" gate on the limiter bid was tried twice and both
times caused overshoot or fight-with-cruise at the boundary. These tests exist
so a third attempt fails here instead of in the truck.
"""
from __future__ import annotations

import pytest

import core.longitudinal.limiter as limiter_mod
from core.longitudinal.limiter import SpeedLimiter
from tests.longitudinal.harness import make_ctx

TARGET_KMH = 90.0
ACCEL_MIN = -2.0


class _FakeSettings:
    limiter_kp = 0.5
    limiter_ki = 0.05
    limiter_kd = 0.0
    limiter_integral_clamp = 2.0
    limiter_accel_min_ms2 = ACCEL_MIN


@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    monkeypatch.setattr(limiter_mod, "Settings", _FakeSettings)


def _limiter_at(target_kmh: float) -> SpeedLimiter:
    lim = SpeedLimiter()
    lim.set_target_kmh(target_kmh)
    lim.enable()
    return lim


@pytest.fixture
def limiter():
    return _limiter_at(TARGET_KMH)


def _run(lim, speed_kmh: float, *, ticks: int = 50, dt: float = 0.02):
    """Hold a steady speed for `ticks` and return the final output."""
    out = None
    for i in range(ticks):
        out = lim.step(make_ctx(speed_kmh / 3.6, dt=dt, now=i * dt))
    return out


def test_active_every_tick_below_the_cap(limiter):
    """active=True unconditionally while enabled, even 30 km/h under the cap."""
    for i in range(50):
        out = limiter.step(make_ctx(60.0 / 3.6, now=i * 0.02))
        assert out.active is True
        assert out.wanted_ms2 is not None


def test_bids_positive_accel_below_the_cap(limiter):
    """The tracker bids, it does not sit at zero waiting for an overshoot."""
    out = _run(limiter, 60.0)
    assert out.wanted_ms2 > 0.0


def test_gas_cap_tightens_as_ego_approaches_the_cap(limiter):
    """Progressive tightening is the whole point of the continuous tracker."""
    far = _run(_limiter_at(TARGET_KMH), 60.0).wanted_ms2
    near = _run(_limiter_at(TARGET_KMH), 88.0).wanted_ms2
    assert near < far


def test_inactive_only_when_disabled_or_targetless():
    lim = SpeedLimiter()
    lim.set_target_kmh(TARGET_KMH)
    out = lim.step(make_ctx(60.0 / 3.6))
    assert out.active is False and out.wanted_ms2 is None

    lim.enable()
    assert lim.step(make_ctx(60.0 / 3.6)).active is True

    lim.disable()
    out = lim.step(make_ctx(60.0 / 3.6))
    assert out.active is False and out.wanted_ms2 is None


def test_asymmetric_clamp_leaves_the_positive_side_open(limiter):
    """max(accel_min, wanted) only: a positive bid must pass through unbounded."""
    out = _run(limiter, 30.0)
    assert out.wanted_ms2 > abs(ACCEL_MIN), "positive bid was clamped like the lower side"


def test_lower_clamp_bounds_the_decel_bid(limiter, monkeypatch):
    """With overshoot protection off, accel_min is a hard floor."""
    monkeypatch.setattr(limiter_mod, "_OVERSHOOT_CUBIC_K", 0.0)
    out = _run(limiter, TARGET_KMH + 20.0)
    assert out.wanted_ms2 == pytest.approx(ACCEL_MIN)


def test_overshoot_protection_at_most_doubles_the_floor(limiter):
    """The cubic is capped at |accel_min|, so it can never exceed 2x the floor."""
    out = _run(limiter, TARGET_KMH + 30.0, ticks=300)
    assert out.wanted_ms2 >= 2.0 * ACCEL_MIN - 1e-9
