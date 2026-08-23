"""Clearance-model required decel: equivalences, crossing behaviour, cost.

The equivalence tests are the load-bearing ones. The clearance solver replaced
`_required_decel_two_frame` and `_codir_required_cap` for every target class, so
the classes that must not move (rear-end, stationary obstacle) are pinned here
against the closed forms those functions computed.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace

import pytest

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.clearance import (
    clearance_required, min_decel_to_clear, occupancy_profile, required_at,
    sample_times,
)
from core.radar.traffic import build_arc, capsule_extents

_EGO_OFFSET = (CAL.arc_start_pctg - 0.5) * (2.0 * CAL.ego_half_length)
_CAP_FWD, _CAP_BACK = capsule_extents(
    CAL.ego_half_length, CAL.ego_half_length, _EGO_OFFSET,
)
_FRONT_TO_SURFACE = _CAP_FWD + CAL.ego_half_width

# The clear margin only fires when occupancy ends, so the co-directional
# equivalences are unaffected by it. Zeroed anyway so they read as pure algebra.
_NO_MARGIN = replace(CAL, clearance_clear_margin_s=0.0)


def _ego(v_ms: float, kappa: float = 0.0, horizon: float = 3.0):
    return build_arc(
        0.0, 0.0, 0.0, v_ms, kappa, CAL.ego_half_width, horizon,
        fwd_len=_CAP_FWD, back_len=_CAP_BACK,
        parallel_margin_scale=CAL.capsule_parallel_margin_scale,
    )


def _lead(gap_m: float, v_ms: float, decel: float = 0.0,
          half_len: float = 3.0, half_w: float = 1.1, horizon: float = 3.0):
    """Co-directional body whose rear surface sits `gap_m` from ego's arc origin."""
    return build_arc(
        0.0, -(gap_m + half_len), 0.0, v_ms, 0.0, half_w, horizon, decel=decel,
        fwd_len=half_len, back_len=half_len,
        parallel_margin_scale=CAL.capsule_parallel_margin_scale,
    )


def _crosser(x_m: float, z_m: float, v_ms: float, yaw_deg: float = 270.0,
             half_len: float = 2.5, half_w: float = 0.9):
    return build_arc(
        x_m, z_m, math.radians(yaw_deg), v_ms, 0.0, half_w, 3.0,
        fwd_len=half_len, back_len=half_len,
        parallel_margin_scale=CAL.capsule_parallel_margin_scale,
    )


def _solve(ego_arc, arcs, v_ms, cal=_NO_MARGIN, lag_s=0.0, pad_m=0.0):
    return clearance_required(
        ego_arc, arcs, v_ms, cal, lag_s=lag_s, pad_m=pad_m,
        front_to_surface=_FRONT_TO_SURFACE, near_horizon_s=3.0,
    )


_STEADY_CASES = [
    (v0, vl, gap)
    for v0 in (10.0, 15.0, 20.0, 25.0, 30.0, 33.0)
    for vl in (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
    for gap in (15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0)
    if vl < v0
]


@pytest.mark.parametrize("lag", [0.0, CAL.stop_buffer_response_s,
                                 CAL.stop_buffer_response_trailer_s])
def test_a_steady_lead_reproduces_the_relative_frame_formula(lag):
    """The co-directional limit of the clearance model is `dv^2 / (2 * d_rel)`.

    `s_near(t)` is linear for a steady lead, so the demand peaks at
    `t* = 2 * gap / dv` and evaluates to exactly what the relative frame
    computed in closed form. This is the guarantee that rear-end timing, and
    with it the whole stop-distance envelope, did not move.
    """
    for v0, vl, gap in _STEADY_CASES:
        res = _solve(_ego(v0), [_lead(gap, vl)], v0, lag_s=lag)
        assert res is not None
        d_rel = gap - _FRONT_TO_SURFACE - CAL.stop_buffer - (v0 - vl) * lag
        classic = ((v0 - vl) ** 2 / (2.0 * d_rel)) if d_rel > 0.0 else math.inf
        if classic > 100.0:
            # Ten g. Both models mean "unavoidable" and the command clips at
            # effective_max either way, so the magnitude carries no decision.
            assert res.required_ms2 > 100.0
            continue
        assert res.required_ms2 == pytest.approx(classic, rel=0.02), (
            f"v0={v0} vl={vl} gap={gap} lag={lag}"
        )


def test_a_stationary_obstacle_reproduces_the_ego_frame_formula():
    """A parked body never clears, so the answer is stop-short: `v^2 / (2 * d)`."""
    for gap, v0 in ((40.0, 25.0), (30.0, 20.0), (20.0, 15.0), (60.0, 30.0)):
        res = _solve(_ego(v0), [_lead(gap, 0.0)], v0, lag_s=0.30)
        d = gap - _FRONT_TO_SURFACE - CAL.stop_buffer - v0 * 0.30
        assert res.required_ms2 == pytest.approx(v0 * v0 / (2.0 * d), rel=0.02)
        assert not res.clears


def test_a_braking_lead_is_at_least_the_hand_rolled_stop_branch():
    """`_codir_required_cap` sampled two points of a curve; the solver maximises it.

    So the clearance demand must never fall below the old `r_stop`, and where the
    binding moment sits between the lead's start and its stop it reads higher.
    That is a correction, not a regression: the old pair could under-read a
    hard-braking lead by about 30%.
    """
    for v0, vl, a_t, gap in ((25.0, 20.0, 4.0, 30.0), (25.0, 15.0, 6.0, 25.0),
                             (30.0, 25.0, 5.0, 40.0), (25.0, 10.0, 3.0, 20.0)):
        res = _solve(_ego(v0), [_lead(gap, vl, decel=a_t)], v0, lag_s=0.30)
        s_stop = vl * vl / (2.0 * a_t)
        d = gap - _FRONT_TO_SURFACE - CAL.stop_buffer + s_stop - v0 * 0.30
        r_stop = v0 * v0 / (2.0 * d)
        assert res.required_ms2 >= r_stop * 0.98


def test_a_crosser_that_clears_demands_far_less_than_stopping_for_it():
    """The whole point: ego arrives behind the crosser instead of stopping short."""
    v0 = 80.0 / 3.6
    v_t = 50.0 / 3.6
    v_closing = math.hypot(v0, v_t)
    for z in (45.0, 50.0, 60.0):
        res = _solve(_ego(v0), [_crosser(-18.0, -z, v_t)], v0,
                     cal=CAL, lag_s=0.30)
        assert res is not None and res.clears
        d = z - _FRONT_TO_SURFACE - CAL.stop_buffer - v_closing * 0.30
        old = v_closing * v_closing / (2.0 * d)
        assert res.required_ms2 < old * 0.5


def test_crosser_demand_rises_monotonically_as_ego_closes():
    """Only trigger when it becomes a real danger, so demand must be monotone."""
    v0 = 80.0 / 3.6
    v_t = 50.0 / 3.6
    prev = -1.0
    for z in (60.0, 50.0, 45.0, 40.0, 35.0, 30.0, 25.0):
        res = _solve(_ego(v0), [_crosser(-18.0, -z, v_t)], v0,
                     cal=CAL, lag_s=0.30)
        assert res.required_ms2 >= prev, f"demand fell closing to {z} m"
        prev = res.required_ms2


def test_a_crosser_that_stops_in_the_lane_is_stop_short():
    """No clearing means the last sample binds and the answer is the parked one."""
    v0 = 80.0 / 3.6
    blocked = _crosser(0.0, -45.0, 0.0)
    res = _solve(_ego(v0), [blocked], v0, cal=CAL, lag_s=0.30)
    assert res is not None and not res.clears
    d = 45.0 - _FRONT_TO_SURFACE - CAL.stop_buffer - v0 * 0.30
    assert res.required_ms2 == pytest.approx(v0 * v0 / (2.0 * d), rel=0.05)


def test_an_unavoidable_geometry_never_reads_as_a_low_demand():
    """Fail closed: no decel avoids it, so the answer is infinite, not small."""
    v0 = 25.0
    res = _solve(_ego(v0), [_lead(4.0, 0.0)], v0, lag_s=0.30)
    assert res.required_ms2 == float("inf")


def test_the_clear_margin_never_fires_for_a_co_directional_lead():
    """A lead that never leaves the corridor must not collect the crossing pad."""
    v0, vl, gap = 25.0, 15.0, 30.0
    with_margin = _solve(_ego(v0), [_lead(gap, vl)], v0, cal=CAL, lag_s=0.30)
    without = _solve(_ego(v0), [_lead(gap, vl)], v0, cal=_NO_MARGIN, lag_s=0.30)
    assert not with_margin.clears
    assert with_margin.required_ms2 == pytest.approx(without.required_ms2, rel=1e-9)


def test_v_pass_is_the_speed_that_actually_clears_the_conflict():
    """`v_pass` answers what ego can go through the intersection at."""
    v0 = 80.0 / 3.6
    res = _solve(_ego(v0), [_crosser(-18.0, -40.0, 50.0 / 3.6)], v0,
                 cal=CAL, lag_s=0.30)
    assert 0.0 < res.v_pass_ms < v0
    brake_dist = res.s_bind_m - _FRONT_TO_SURFACE - CAL.stop_buffer - v0 * 0.30
    expected = math.sqrt(max(v0 * v0 - 2.0 * res.required_ms2 * brake_dist, 0.0))
    assert res.v_pass_ms == pytest.approx(expected, rel=0.02)


def test_required_at_is_continuous_across_the_roll_stop_branch():
    """The two branches meet at `d = v0 * (t + lag) / 2`; a step there would chatter."""
    v0, lag = 22.0, 0.30
    for t in (0.8, 1.5, 2.4, 4.0):
        d_switch = v0 * (t + lag) / 2.0
        lo = required_at(v0, t, d_switch * (1.0 - 1e-6), lag)
        hi = required_at(v0, t, d_switch * (1.0 + 1e-6), lag)
        assert lo == pytest.approx(hi, rel=1e-4)


def test_no_occupancy_returns_none_so_the_caller_can_fall_back():
    v0 = 25.0
    far_aside = _crosser(60.0, -40.0, 0.0)
    assert _solve(_ego(v0), [far_aside], v0) is None
    assert _solve(_ego(v0), [], v0) is None


def test_the_solver_stays_cheap_enough_for_the_30_hz_loop():
    """Worst case is a body in the corridor at every sample, tractor plus trailer.

    Budget is 33.3 ms a tick and the solver only runs for targets that already
    produced a hit. The bar is here so a later "just project every body point"
    simplification cannot silently restore the 213 us path this replaced.
    """
    ego = _ego(80.0 / 3.6)
    arcs = [_lead(35.0, 60.0 / 3.6),
            _lead(43.0, 60.0 / 3.6, half_len=6.5, half_w=1.3)]
    _solve(ego, arcs, 80.0 / 3.6, cal=CAL, lag_s=0.30)
    reps = 400
    t0 = time.perf_counter()
    for _ in range(reps):
        _solve(ego, arcs, 80.0 / 3.6, cal=CAL, lag_s=0.30)
    per_call_us = (time.perf_counter() - t0) / reps * 1e6
    assert per_call_us < 400.0, f"{per_call_us:.0f} us per target"


def test_occupancy_of_a_crosser_is_a_bounded_window():
    """A crosser enters and leaves; the profile must reflect both edges."""
    ego = _ego(80.0 / 3.6)
    times = sample_times(3.0, CAL.clearance_horizon_s,
                         CAL.clearance_samples, CAL.clearance_far_samples)
    profile, clears = occupancy_profile(
        ego, [_crosser(-18.0, -45.0, 50.0 / 3.6)], times, _NO_MARGIN,
    )
    assert clears
    assert 0 < len(profile) < len(times)
    assert profile[-1][0] < times[-1]
