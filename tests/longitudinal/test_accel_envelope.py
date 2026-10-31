"""The CC accel ceiling is a speed-scheduled comfort shape, and Sport has none.

A flat ceiling (the old cc_accel_max_ms2 = 1.0) is wrong at both ends: weak off
the line, a shove at highway speed. The shape fixes that. There is no capability
term: the estimate under-read low gears about 2x and held town pulls at 1.3 m/s2
on a truck doing that at a third of its pedal (mapper log, 2026-10-19).
"""
from __future__ import annotations

import math

import pytest

import core.longitudinal.cc as cc_mod
from core.longitudinal.accel_envelope import (
    EFFICIENCY,
    NORMAL,
    PROFILES,
    SPORT,
    resolve_profile,
    rise_limited_ms2,
    shape_ceiling_ms2,
)
from core.longitudinal.cc import CruiseController
from core.settings import Settings
from core.thread_management.registry import registry
from tests.longitudinal.harness import FakeThread, make_ctx, sending_data

SPEEDS_MS = [kmh / 3.6 for kmh in range(0, 135, 5)]


class _FakeSettings:
    global_speed_limit_kmh = None
    cc_kp = 0.5
    cc_ki = 0.0
    cc_kd = 0.2
    cc_integral_clamp = 3.0
    cc_accel_min_ms2 = -1.0
    cc_accel_max_ms2 = 4.0
    cc_accel_profile = "Normal"


@pytest.fixture
def fake_settings(monkeypatch):
    _FakeSettings.cc_accel_profile = "Normal"
    _FakeSettings.cc_accel_max_ms2 = 4.0
    monkeypatch.setattr(cc_mod, "Settings", _FakeSettings)
    return _FakeSettings


@pytest.fixture
def sending():
    """A registered sending_thread publishing a capacity estimate."""
    thread = FakeThread("sending_thread", sending_data())
    registry.replace(thread)
    yield thread.data
    registry.unregister("sending_thread")


@pytest.fixture
def cc(fake_settings):
    ctrl = CruiseController()
    ctrl.enable()
    ctrl.set_target_kmh(130.0)
    return ctrl


def settle(ctrl, speed_ms, *, ticks=300, dt=0.02, **ctx_kw):
    """Run to steady state and return the final bid."""
    out = None
    for _ in range(ticks):
        out = ctrl.step(make_ctx(speed_ms, dt=dt, **ctx_kw))
    return out.wanted_ms2


# Shape function


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.label)
def test_shape_is_monotone_non_increasing_in_speed(profile):
    values = [shape_ceiling_ms2(v, profile) for v in SPEEDS_MS]
    for slower, faster in zip(values, values[1:]):
        assert faster <= slower + 1e-12


@pytest.mark.parametrize("profile", (EFFICIENCY, NORMAL), ids=lambda p: p.label)
def test_shape_has_no_kink_anywhere(profile):
    """A knee is a jerk step the driver feels; the slope must change gradually."""
    h = 0.01
    slopes = [
        (shape_ceiling_ms2(v + h, profile) - shape_ceiling_ms2(v, profile)) / h
        for v in (i * 0.05 for i in range(800))
    ]
    assert abs(slopes[0]) < 1e-3, "the launch must start flat"
    for a, b in zip(slopes, slopes[1:]):
        assert abs(b - a) < 2e-3


@pytest.mark.parametrize("profile", (EFFICIENCY, NORMAL), ids=lambda p: p.label)
def test_shape_stays_inside_its_own_bounds(profile):
    for v in SPEEDS_MS:
        value = shape_ceiling_ms2(v, profile)
        assert profile.cruise_ms2 - 1e-12 <= value <= profile.launch_ms2 + 1e-12


def test_profiles_are_strictly_ordered_at_every_speed():
    for v in SPEEDS_MS:
        eco = shape_ceiling_ms2(v, EFFICIENCY)
        normal = shape_ceiling_ms2(v, NORMAL)
        sport = shape_ceiling_ms2(v, SPORT)
        assert eco < normal < sport, f"ordering broken at {v * 3.6:.0f} km/h"


def test_default_profile_answers_both_halves_of_the_complaint():
    """Normal is what a driver who never opens the drawer gets."""
    assert shape_ceiling_ms2(20 / 3.6, NORMAL) > 1.0, "weak off the line"
    assert shape_ceiling_ms2(85 / 3.6, NORMAL) < 1.0, "a shove at highway speed"


@pytest.mark.parametrize("profile", (EFFICIENCY, NORMAL), ids=lambda p: p.label)
def test_the_pull_does_not_fall_away_through_town_speeds(profile):
    """2026-10-19: Efficiency asked 0.14 m/s2 at 50 km/h and felt broken.

    No 10 km/h step may cost more than a fifth of the pull.
    """
    for kmh in range(0, 130, 10):
        here = shape_ceiling_ms2(kmh / 3.6, profile)
        nxt = shape_ceiling_ms2((kmh + 10) / 3.6, profile)
        assert nxt > 0.8 * here, f"{profile.label} drops off a cliff at {kmh} km/h"


def test_normal_keeps_a_real_pull_until_well_past_town_speed():
    assert shape_ceiling_ms2(30 / 3.6, NORMAL) > 1.5
    assert shape_ceiling_ms2(50 / 3.6, NORMAL) > 1.1
    assert shape_ceiling_ms2(90 / 3.6, NORMAL) > 0.75, "a highway resume must not crawl"


def test_efficiency_is_a_softer_normal_at_every_speed():
    """Same curve family: the driver feels one character, gentler, not a new shape."""
    for v in SPEEDS_MS:
        ratio = shape_ceiling_ms2(v, EFFICIENCY) / shape_ceiling_ms2(v, NORMAL)
        assert 0.5 <= ratio <= 0.75, f"ratio {ratio:.2f} at {v * 3.6:.0f} km/h"
    assert shape_ceiling_ms2(90 / 3.6, EFFICIENCY) > 0.4, "a highway resume must not crawl"


def test_sport_has_no_acceleration_ceiling():
    """Full throttle at every speed: the shape is not a comfort curve."""
    for kmh in (0, 10, 40, 85, 130):
        assert shape_ceiling_ms2(kmh / 3.6, SPORT) > 20.0
    assert SPORT.uncapped


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.label)
def test_broken_speed_telemetry_bids_the_least_not_the_most(profile):
    for bad in (float("nan"), float("inf"), None, "fast"):
        assert shape_ceiling_ms2(bad, profile) <= min(profile.cruise_ms2, 0.5)


# Profile lookup


def test_profile_lookup_is_case_insensitive_and_falls_back():
    assert resolve_profile("sport") is SPORT
    assert resolve_profile("  EFFICIENCY ") is EFFICIENCY
    assert resolve_profile("Turbo") is NORMAL
    assert resolve_profile(None) is NORMAL
    assert resolve_profile(7) is NORMAL


def test_settings_normalizer_matches_the_profile_table():
    assert Settings._normalize_accel_profile("sport") == "Sport"
    assert Settings._normalize_accel_profile("nonsense") == "Normal"
    assert Settings._normalize_accel_profile(None) == "Normal"


# Rise limit


def test_rise_limit_never_drags_a_bid_negative():
    """The regression guard: clamping against a raw negative prev reads as brake."""
    out = rise_limited_ms2(0.3, -0.5, NORMAL, 0.02)
    assert out >= 0.0
    assert out == pytest.approx(NORMAL.rise_jerk_ms3 * 0.02)


def test_falling_bids_are_never_rate_limited():
    assert rise_limited_ms2(-2.0, 1.0, EFFICIENCY, 0.02) == -2.0
    assert rise_limited_ms2(0.1, 1.0, EFFICIENCY, 0.02) == 0.1


def test_first_commanding_tick_ramps_from_zero():
    assert rise_limited_ms2(2.0, None, NORMAL, 0.02) == pytest.approx(NORMAL.rise_jerk_ms3 * 0.02)


def test_sport_does_not_ramp_the_bid():
    assert rise_limited_ms2(4.0, None, SPORT, 0.02) == 4.0


def test_rise_limit_respects_the_profile_ordering():
    assert (
        rise_limited_ms2(3.0, 0.0, EFFICIENCY, 0.1)
        < rise_limited_ms2(3.0, 0.0, NORMAL, 0.1)
        < rise_limited_ms2(3.0, 0.0, SPORT, 0.1)
    )


# CruiseController integration


def test_bid_ceiling_falls_as_speed_rises(cc):
    low = settle(cc, 20 / 3.6)
    cc.reset()
    high = settle(cc, 85 / 3.6)
    assert low > high
    assert low == pytest.approx(shape_ceiling_ms2(20 / 3.6, NORMAL), abs=1e-3)
    assert high == pytest.approx(shape_ceiling_ms2(85 / 3.6, NORMAL), abs=1e-3)


def test_selected_profile_changes_the_bid_ceiling(cc, fake_settings):
    seen = {}
    for label in ("Efficiency", "Normal", "Sport"):
        fake_settings.cc_accel_profile = label
        cc.reset()
        seen[label] = settle(cc, 50 / 3.6)
    assert seen["Efficiency"] < seen["Normal"] < seen["Sport"]


@pytest.mark.parametrize("estimate", [0.0, 0.5, 1.3, 10.0])
def test_the_capacity_estimate_never_holds_the_bid_down(cc, sending, estimate):
    """2026-10-19 log: 0.95 x an estimate of 1.36 held a town pull at 1.29 m/s2.

    The truck did that on 20 to 50 % gas; a truck that really is weaker rails the pedal.
    """
    sending.set(mapper_est_max_accel_ms2=estimate)
    assert settle(cc, 20 / 3.6) == pytest.approx(shape_ceiling_ms2(20 / 3.6, NORMAL), abs=1e-3)


def test_a_missing_sending_thread_does_not_raise(cc):
    registry.unregister("sending_thread")
    assert settle(cc, 50 / 3.6) == pytest.approx(shape_ceiling_ms2(50 / 3.6, NORMAL), abs=1e-3)


def test_the_safety_rail_caps_the_calmer_profiles(cc, fake_settings):
    fake_settings.cc_accel_profile = "Efficiency"
    fake_settings.cc_accel_max_ms2 = 0.6
    assert settle(cc, 20 / 3.6) == pytest.approx(0.6, abs=1e-3)


def test_sport_ignores_the_safety_rail(cc, fake_settings):
    fake_settings.cc_accel_profile = "Sport"
    fake_settings.cc_accel_max_ms2 = 0.6
    assert settle(cc, 20 / 3.6) > 2.0


def test_sport_bids_the_speed_error_from_the_first_tick(cc, fake_settings):
    """No ramp and no ceiling: the pedal is meant to reach the floor at once."""
    fake_settings.cc_accel_profile = "Sport"
    assert cc.step(make_ctx(40 / 3.6, dt=0.02)).wanted_ms2 > 5.0


def test_braking_authority_is_untouched_by_the_envelope(cc):
    """Overspeed: the bid must reach the unchanged negative clamp."""
    cc.set_target_kmh(50.0)
    assert settle(cc, 110 / 3.6) == pytest.approx(_FakeSettings.cc_accel_min_ms2, abs=1e-3)


@pytest.mark.parametrize("label", ["Efficiency", "Normal", "Sport"])
def test_the_bid_is_always_finite(cc, fake_settings, label):
    fake_settings.cc_accel_profile = label
    for kmh in (0, 20, 60, 130):
        cc.reset()
        assert math.isfinite(settle(cc, kmh / 3.6, ticks=50))


def test_the_rise_limit_is_bypassed_in_neutral(cc):
    """Auto-neutral shifts back to drive off this bid crossing 0.25 m/s2.

    Ramping from zero in neutral buys no comfort and delays every launch.
    """
    first = cc.step(make_ctx(2.0 / 3.6, dt=0.02, gear_dashboard=0)).wanted_ms2
    assert first > 0.25

    cc.reset()
    in_gear = cc.step(make_ctx(2.0 / 3.6, dt=0.02, gear_dashboard=1)).wanted_ms2
    assert in_gear == pytest.approx(NORMAL.rise_jerk_ms3 * 0.02)
