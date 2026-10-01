"""The CC accel ceiling is speed-scheduled and bounded by truck capability.

A flat ceiling (the old cc_accel_max_ms2 = 1.0) is wrong at both ends: weak off
the line, a shove at highway speed. The shape function fixes that. The capability
term keeps the request honest, but only as a headroom guard: a per-profile share
of capability would make a loaded truck slower than no envelope at all.
"""
from __future__ import annotations

import math

import pytest

import core.longitudinal.cc as cc_mod
from core.longitudinal.accel_envelope import (
    EFFICIENCY,
    HEADROOM_FRAC,
    NORMAL,
    PROFILES,
    SPORT,
    envelope_ms2,
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
    cc_accel_max_ms2 = 2.5
    cc_accel_profile = "Normal"


@pytest.fixture
def fake_settings(monkeypatch):
    _FakeSettings.cc_accel_profile = "Normal"
    _FakeSettings.cc_accel_max_ms2 = 2.5
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


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.label)
def test_shape_is_continuous_at_the_knee(profile):
    below = shape_ceiling_ms2(profile.knee_ms - 1e-6, profile)
    at = shape_ceiling_ms2(profile.knee_ms, profile)
    above = shape_ceiling_ms2(profile.knee_ms + 1e-6, profile)
    assert below == pytest.approx(at, abs=1e-6)
    assert above == pytest.approx(at, abs=1e-6)


@pytest.mark.parametrize("profile", PROFILES, ids=lambda p: p.label)
def test_shape_stays_inside_its_own_bounds(profile):
    for v in SPEEDS_MS:
        value = shape_ceiling_ms2(v, profile)
        assert profile.floor_ms2 - 1e-12 <= value <= profile.launch_ms2 + 1e-12


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


def test_normal_raised_the_launch_without_touching_the_highway_tail():
    """The tail is launch_ms2 * knee_ms when taper_power is 1.0, and it is pinned."""
    assert NORMAL.taper_power == 1.0
    assert NORMAL.launch_ms2 * NORMAL.knee_ms == pytest.approx(1.50 * (35 / 3.6))
    for kmh in (40, 50, 70, 85, 110):
        v = kmh / 3.6
        assert shape_ceiling_ms2(v, NORMAL) == pytest.approx((1.50 * (35 / 3.6)) / v)


def test_efficiency_eases_off_right_after_launch():
    """The old flat plateau out to 30 km/h asked for more pedal than the speed warranted."""
    assert shape_ceiling_ms2(20 / 3.6, EFFICIENCY) < 1.0
    assert EFFICIENCY.knee_ms < 15 / 3.6, "taper has to start just above the launch"
    # The highway tail is deliberately left near where it already was.
    assert shape_ceiling_ms2(85 / 3.6, EFFICIENCY) == pytest.approx(0.40, abs=0.05)


def test_sport_is_high_enough_that_capability_is_the_real_limit():
    """Sport should be met only where the truck runs out of engine.

    Capability at gas=1.0 tops out near 1.4 m/s2 even on a light rig above
    40 km/h, so a shape above that is never the binding term on flat ground.
    """
    for kmh in (40, 60, 85, 110, 130):
        assert shape_ceiling_ms2(kmh / 3.6, SPORT) > 1.4


def test_broken_speed_telemetry_bids_the_least_not_the_most():
    for bad in (float("nan"), float("inf"), None, "fast"):
        assert shape_ceiling_ms2(bad, SPORT) == SPORT.floor_ms2


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


# Envelope


def test_abundant_capability_leaves_the_comfort_shape_in_charge():
    v = 20 / 3.6
    assert envelope_ms2(v, NORMAL, 10.0) == pytest.approx(shape_ceiling_ms2(v, NORMAL))


def test_scarce_capability_binds_with_headroom_left():
    v = 20 / 3.6
    assert envelope_ms2(v, NORMAL, 0.5) == pytest.approx(HEADROOM_FRAC * 0.5)


@pytest.mark.parametrize("unknown", [None, 0.0, -1.0, float("nan"), float("inf"), "n/a"])
def test_unknown_capability_falls_back_to_the_shape(unknown):
    v = 60 / 3.6
    assert envelope_ms2(v, NORMAL, unknown) == pytest.approx(shape_ceiling_ms2(v, NORMAL))


def test_capability_guard_never_asks_for_more_than_the_truck_has():
    for cap in (0.2, 0.4, 0.8, 1.6):
        for v in SPEEDS_MS:
            assert envelope_ms2(v, SPORT, cap) <= cap


def test_profiles_converge_when_capability_is_the_binding_term():
    """A loaded truck at highway speed has no choice left to make."""
    v, cap = 70 / 3.6, 0.40
    values = {envelope_ms2(v, p, cap) for p in PROFILES}
    assert len(values) == 1


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
    assert rise_limited_ms2(2.0, None, SPORT, 0.02) == pytest.approx(SPORT.rise_jerk_ms3 * 0.02)


def test_rise_limit_respects_the_profile_ordering():
    assert (
        rise_limited_ms2(3.0, 0.0, EFFICIENCY, 0.1)
        < rise_limited_ms2(3.0, 0.0, NORMAL, 0.1)
        < rise_limited_ms2(3.0, 0.0, SPORT, 0.1)
    )


# CruiseController integration


def test_bid_ceiling_falls_as_speed_rises(cc, sending):
    sending.set(mapper_est_max_accel_ms2=10.0)
    low = settle(cc, 20 / 3.6)
    cc.reset()
    high = settle(cc, 85 / 3.6)
    assert low > high
    assert low == pytest.approx(shape_ceiling_ms2(20 / 3.6, NORMAL), abs=1e-3)
    assert high == pytest.approx(shape_ceiling_ms2(85 / 3.6, NORMAL), abs=1e-3)


def test_selected_profile_changes_the_bid_ceiling(cc, sending, fake_settings):
    sending.set(mapper_est_max_accel_ms2=10.0)
    seen = {}
    for label in ("Efficiency", "Normal", "Sport"):
        fake_settings.cc_accel_profile = label
        cc.reset()
        seen[label] = settle(cc, 50 / 3.6)
    assert seen["Efficiency"] < seen["Normal"] < seen["Sport"]


def test_capacity_estimate_caps_the_bid(cc, sending):
    sending.set(mapper_est_max_accel_ms2=0.5)
    assert settle(cc, 20 / 3.6) == pytest.approx(HEADROOM_FRAC * 0.5, abs=1e-3)


def test_a_missing_sending_thread_does_not_raise(cc):
    registry.unregister("sending_thread")
    assert settle(cc, 50 / 3.6) == pytest.approx(shape_ceiling_ms2(50 / 3.6, NORMAL), abs=1e-3)


def test_a_gear_change_does_not_step_the_ceiling(cc, sending):
    """Published capacity is per-gear and jumps at each shift; the EMA absorbs it."""
    sending.set(mapper_est_max_accel_ms2=0.60)
    before = settle(cc, 50 / 3.6)
    sending.set(mapper_est_max_accel_ms2=0.60 / 1.27)
    after = cc.step(make_ctx(50 / 3.6, dt=0.02)).wanted_ms2
    assert after < before
    assert after > before * 0.97, "one tick moved the ceiling like a raw gear step"


def test_the_safety_rail_still_caps_every_profile(cc, sending, fake_settings):
    fake_settings.cc_accel_profile = "Sport"
    fake_settings.cc_accel_max_ms2 = 0.6
    sending.set(mapper_est_max_accel_ms2=10.0)
    assert settle(cc, 20 / 3.6) == pytest.approx(0.6, abs=1e-3)


def test_braking_authority_is_untouched_by_the_envelope(cc, sending):
    """Overspeed: the bid must reach the unchanged negative clamp."""
    sending.set(mapper_est_max_accel_ms2=0.3)
    cc.set_target_kmh(50.0)
    assert settle(cc, 110 / 3.6) == pytest.approx(_FakeSettings.cc_accel_min_ms2, abs=1e-3)


def test_the_bid_is_always_finite(cc, sending):
    for cap in (0.0, 0.3, 5.0):
        sending.set(mapper_est_max_accel_ms2=cap)
        cc.reset()
        assert math.isfinite(settle(cc, 60 / 3.6, ticks=50))


def test_the_rise_limit_is_bypassed_in_neutral(cc, sending):
    """Auto-neutral shifts back to drive off this bid crossing 0.25 m/s2.

    Ramping from zero in neutral buys no comfort (no torque path to the wheels)
    and delays every launch by up to half a second.
    """
    sending.set(mapper_est_max_accel_ms2=10.0)
    first = cc.step(make_ctx(2.0 / 3.6, dt=0.02, gear_dashboard=0)).wanted_ms2
    assert first > 0.25

    cc.reset()
    in_gear = cc.step(make_ctx(2.0 / 3.6, dt=0.02, gear_dashboard=1)).wanted_ms2
    assert in_gear == pytest.approx(NORMAL.rise_jerk_ms3 * 0.02)
