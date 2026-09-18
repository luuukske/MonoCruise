"""Brake release after a hard ACC brake.

The at-clamp slam bypasses the jerk limiter on entry but not on exit, so letting
go of -6.55 m/s^2 at the 2.5 m/s^3 onset rate took 2.6 s whatever the law asked.
Clip c5a0a74e: the truck braked to 8 km/h behind a lead holding 31 km/h. See
core/acc/ACC_ARCHITECTURE.md §13.1."""
from __future__ import annotations

import math
import random

import pytest

from core.cruise_control_thread import acc_controller
from core.cruise_control_thread.acc_controller import AdaptiveCruiseController, _LeadSnapshot
from core.cruise_control_thread.blinker_arbitration import BlinkerState
from core.cruise_control_thread.idm_cah import jerk_step
from core.settings import Settings

DT = 1.0 / 100.0


def _plain(prev: float, target: float, dt: float, j_max: float) -> float:
    """The limiter before §13.1: one symmetric rate."""
    delta = target - prev
    if delta > j_max * dt:
        return prev + j_max * dt
    if delta < -j_max * dt:
        return prev - j_max * dt
    return target


def _cases(n: int = 20000):
    rng = random.Random(1789)
    for _ in range(n):
        yield rng.uniform(-8.0, 2.0), rng.uniform(-8.0, 2.0), rng.uniform(1e-3, 0.2)


def _cfg():
    return AdaptiveCruiseController().config


def test_zero_release_tau_is_the_old_limiter_bit_for_bit():
    j = _cfg().j_max_ms3
    for prev, target, dt in _cases():
        assert jerk_step(prev, target, dt, j, 0.0) == _plain(prev, target, dt, j)


def test_brake_onset_and_the_gas_side_are_untouched():
    cfg = _cfg()
    for prev, target, dt in _cases():
        if target < prev or prev >= 0.0:
            assert jerk_step(prev, target, dt, cfg.j_max_ms3, cfg.j_release_tau_s) == \
                _plain(prev, target, dt, cfg.j_max_ms3)


def test_small_releases_keep_the_plain_limit():
    """Jitter lives well inside j_max * tau, so it still sees one symmetric rate."""
    cfg = _cfg()
    band = cfg.j_max_ms3 * cfg.j_release_tau_s
    rng = random.Random(7)
    for _ in range(20000):
        prev = rng.uniform(-8.0, -0.01)
        target = prev + rng.uniform(0.0, band)
        dt = rng.uniform(1e-3, 0.2)
        assert jerk_step(prev, target, dt, cfg.j_max_ms3, cfg.j_release_tau_s) == \
            _plain(prev, target, dt, cfg.j_max_ms3)


def test_a_release_is_never_slower_never_overshoots_and_stops_at_zero():
    cfg = _cfg()
    for prev, target, dt in _cases():
        if not (prev < 0.0 < target - prev):
            continue
        out = jerk_step(prev, target, dt, cfg.j_max_ms3, cfg.j_release_tau_s)
        plain = _plain(prev, target, dt, cfg.j_max_ms3)
        assert plain <= out <= target
        assert out <= max(0.0, plain), "the boost must not carry the command past zero"


def _snap(dist_m: float, v_lead: float, a_lead: float = 0.0):
    raw = _LeadSnapshot(vid=1, dist_m=dist_m, v_lead_ms=v_lead, a_lead_ms2=a_lead, score=6.0)
    smooth = _LeadSnapshot(vid=1, dist_m=dist_m, v_lead_ms=v_lead, a_lead_ms2=a_lead,
                           score=6.0, conf=1.0, a_lead_ff_ms2=a_lead)
    return [raw], [smooth]


def _tick(ctrl, dist_m, v_ego, v_lead, a_lead=0.0):
    raw, smooth = _snap(dist_m, v_lead, a_lead)
    a_raw, bypass = ctrl._compute_command(raw, smooth, v_ego, DT)
    return a_raw, ctrl._output_filter(ctrl._jerk_limit(a_raw, DT, bypass), DT, bypass), bypass


def _seconds_to_let_go(release_tau: float) -> float:
    """Slam on a close braking lead, then the lead is clear: time until the cap is >= 0."""
    ctrl = AdaptiveCruiseController()
    ctrl.config.j_release_tau_s = release_tau
    _, cap, bypass = _tick(ctrl, 8.0, 15.0, 8.0, -6.0)
    assert bypass and cap == pytest.approx(ctrl.config.max_decel_ms2)
    t = 0.0
    while cap < 0.0 and t < 10.0:
        law, cap, _ = _tick(ctrl, 30.0, 15.0, 20.0)
        assert law > 0.0, "the law has already let go; only the limiter holds the brake"
        t += DT
    return t


def test_a_slam_lets_go_once_the_law_does():
    assert _seconds_to_let_go(_cfg().j_release_tau_s) < 0.8
    assert _seconds_to_let_go(0.0) > 2.5, "the fixture must reproduce the old hangover"


def test_a_lost_lead_releases_at_the_plain_rate(monkeypatch):
    """No lead is missing information, not a law asking to let go."""
    ctrl = AdaptiveCruiseController()
    ctrl._prev_cmd_ms2 = ctrl._output_ema = -4.0
    ctrl._prev_mono = 100.0
    monkeypatch.setattr(ctrl, "_read_acc_snapshot", lambda: ([], None, BlinkerState()))
    monkeypatch.setattr(acc_controller.time, "monotonic", lambda: 100.0 + DT)
    ctrl.accel_cap_ms2(15.0)
    assert ctrl._prev_cmd_ms2 == pytest.approx(-4.0 + ctrl.config.j_max_ms3 * DT)


def test_the_standstill_hold_is_eased_into_at_the_plain_rate():
    """§10.1: from a braking cap the hold eases in; a fast drop would dip the brake."""
    ctrl = AdaptiveCruiseController()
    j = ctrl.config.j_max_ms3
    ctrl._prev_cmd_ms2 = ctrl._output_ema = -2.0
    law, _, bypass = _tick(ctrl, 3.5, 0.3, 0.0)
    assert law == 0.0 and not bypass and ctrl._standstill.held
    assert ctrl._prev_cmd_ms2 == pytest.approx(-2.0 + j * DT)
    # Same starting command, law asking to go: the release is not rate-capped.
    moving = AdaptiveCruiseController()
    moving._prev_cmd_ms2 = moving._output_ema = -2.0
    _tick(moving, 30.0, 15.0, 20.0)
    assert moving._prev_cmd_ms2 > -2.0 + 1.5 * j * DT


def _follow_a_braking_lead(release_tau: float, level: int = 1, v0_kmh: float = 50.0,
                           decel: float = 4.0, dv_kmh: float = 25.0):
    """Closed loop: lead brakes, then holds its new speed. Returns (undershoot km/h, hang s).

    Perfect lead kinematics, so this isolates the limiter from the radar chain."""
    previous = Settings.instance().acc_gap_level
    Settings.instance().acc_gap_level = level
    try:
        ctrl = AdaptiveCruiseController()
        ctrl.config.j_release_tau_s = release_tau
        v = v_lead = v0_kmh / 3.6
        v_end = v_lead - dv_kmh / 3.6
        gap = ctrl.config.s0_m + v * acc_controller.T_HEADWAY_BY_LEVEL_S[level]
        a_truck, t, v_min, lead_done, law_up, cap_up = 0.0, 0.0, v, None, None, None
        while t < 14.0:
            braking = t >= 2.0 and v_lead > v_end
            v_lead = max(v_end, v_lead - decel * DT) if braking else v_lead
            if lead_done is None and t >= 2.0 and not braking:
                lead_done = t
            law, cap, _ = _tick(ctrl, gap, v, v_lead, -decel if braking else 0.0)
            if lead_done is not None:
                law_up = law_up if law_up is not None or law < 0.0 else t
                cap_up = cap_up if cap_up is not None or cap < 0.0 else t
            a_truck += (min(cap, 1.5) - a_truck) * (1.0 - math.exp(-DT / 0.15))
            v = max(0.0, v + a_truck * DT)
            gap += (v_lead - v) * DT
            v_min = min(v_min, v)
            t += DT
        return (v_end - v_min) * 3.6, cap_up - law_up
    finally:
        Settings.instance().acc_gap_level = previous


def test_ego_does_not_fall_far_below_a_lead_that_stopped_braking():
    under, hang = _follow_a_braking_lead(_cfg().j_release_tau_s)
    old_under, old_hang = _follow_a_braking_lead(0.0)
    assert hang < 0.7 and under < 3.0
    assert old_hang > 1.2 and old_under > 8.0, "the fixture must reproduce the old hangover"
