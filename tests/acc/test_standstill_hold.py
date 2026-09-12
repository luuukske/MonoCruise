"""ACC standstill hold: released by the gap law's wanted accel, never by lead speed.

The ACC chain latches a departing lead's speed at zero until it holds 0.6 m/s for
0.5 s, so a lead-speed release waited for ~1.35 m/s. See ACC_ARCHITECTURE.md §10."""
from __future__ import annotations

import pytest

from core.cruise_control_thread import idm_cah
from core.cruise_control_thread.acc_controller import (
    AdaptiveCruiseController, _headway_for_level, _LeadSnapshot,
)
from core.cruise_control_thread.standstill_hold import RELEASE_MARGIN_MS2, REST_SPEED_MS
from core.settings import Settings

DT = 1.0 / 100.0


def _snap(dist_m: float, v_lead: float = 0.0, a_lead: float = 0.0, vid: int = 1):
    raw = _LeadSnapshot(vid=vid, dist_m=dist_m, v_lead_ms=v_lead,
                        a_lead_ms2=a_lead, score=6.0)
    smooth = _LeadSnapshot(vid=vid, dist_m=dist_m, v_lead_ms=v_lead,
                           a_lead_ms2=a_lead, score=6.0, conf=1.0, a_lead_ff_ms2=a_lead)
    return [raw], [smooth]


def _law(ctrl: AdaptiveCruiseController, dist_m: float, v_ego: float,
         v_lead: float = 0.0, a_lead: float = 0.0) -> float:
    t_headway = _headway_for_level(int(Settings.acc_gap_level))
    return idm_cah.lead_law(ctrl.config, dist_m, v_ego, v_lead, a_lead, t_headway, a_lead)


def _tick(ctrl: AdaptiveCruiseController, dist_m: float, v_ego: float,
          v_lead: float = 0.0, a_lead: float = 0.0, vid: int = 1) -> tuple[float, float, bool]:
    """One controller tick. Returns (law command, published cap, smoothing bypassed)."""
    raw, smooth = _snap(dist_m, v_lead, a_lead, vid)
    a_raw, bypass = ctrl._compute_command(raw, smooth, v_ego, DT)
    cap = ctrl._output_filter(ctrl._jerk_limit(a_raw, DT, bypass), DT, bypass)
    return a_raw, cap, bypass


def test_a_departing_lead_releases_the_hold_while_its_speed_still_reads_zero():
    ctrl = AdaptiveCruiseController()
    launch = ctrl.config.standstill_launch_accel_ms2
    for _ in range(100):
        a, _, _ = _tick(ctrl, 4.6, 0.0)
        assert a == 0.0
    # Gap opened and the lead is pulling away, but the ACC chain still reports v = 0.
    assert _law(ctrl, 5.8, 0.0, a_lead=1.0) >= launch
    a, _, _ = _tick(ctrl, 5.8, 0.0, v_lead=0.0, a_lead=1.0)
    assert a >= launch


@pytest.mark.parametrize("dist_m", [4.0, 4.6, 5.0])
def test_a_stationary_lead_at_a_stop_gap_never_releases(dist_m):
    ctrl = AdaptiveCruiseController()
    for _ in range(1000):
        a, cap, _ = _tick(ctrl, dist_m, 0.0)
        assert a == 0.0
        assert cap <= 0.0


def test_ego_at_rest_never_sits_on_a_sub_launch_bid():
    """A small positive bid at rest winds up the mapper against the hold brake."""
    ctrl = AdaptiveCruiseController()
    law = _law(ctrl, 5.3, 0.0)
    assert 0.0 < law < ctrl.config.standstill_launch_accel_ms2
    a, _, _ = _tick(ctrl, 5.3, 0.0)
    assert a == 0.0


def test_a_crawl_the_law_still_wants_is_not_pinned():
    ctrl = AdaptiveCruiseController()
    v_ego = 0.3
    assert REST_SPEED_MS <= v_ego < ctrl.config.standstill_speed_ms
    law = _law(ctrl, 5.8, v_ego)
    assert 0.0 < law < ctrl.config.standstill_launch_accel_ms2
    a, _, bypass = _tick(ctrl, 5.8, v_ego)
    assert a > 0.0
    assert not bypass


def test_a_hold_engaged_just_under_the_launch_bid_needs_the_margin():
    """Otherwise a stop left at the hold FSM's own release re-launches on noise."""
    ctrl = AdaptiveCruiseController()
    launch = ctrl.config.standstill_launch_accel_ms2
    engaged, nudged, opened = 5.4, 5.6, 5.9
    base = _law(ctrl, engaged, 0.0)
    assert 0.0 < base < launch
    assert launch <= _law(ctrl, nudged, 0.0) < base + RELEASE_MARGIN_MS2
    assert _law(ctrl, opened, 0.0) >= base + RELEASE_MARGIN_MS2

    assert _tick(ctrl, engaged, 0.0)[0] == 0.0
    assert _tick(ctrl, nudged, 0.0)[0] == 0.0
    assert _tick(ctrl, opened, 0.0)[0] >= launch


def test_a_normal_stop_keeps_the_plain_launch_release():
    """Engaged while the law brakes, the margin must not delay a real departure."""
    ctrl = AdaptiveCruiseController()
    launch = ctrl.config.standstill_launch_accel_ms2
    assert _tick(ctrl, 4.6, 0.3)[0] == 0.0
    for _ in range(50):
        _tick(ctrl, 4.6, 0.0)
    dist = 5.45
    assert launch <= _law(ctrl, dist, 0.0, a_lead=0.3) < launch + RELEASE_MARGIN_MS2
    assert _tick(ctrl, dist, 0.0, a_lead=0.3)[0] >= launch


def test_a_positive_cap_snaps_to_zero_when_the_hold_engages():
    """A cap decaying toward zero from above never reads as a stop to the hold FSM."""
    ctrl = AdaptiveCruiseController()
    for _ in range(100):
        _, cap, _ = _tick(ctrl, 6.5, 0.2)
    assert cap > 0.0
    a, cap, bypass = _tick(ctrl, 4.0, 0.2, vid=2)
    assert (a, cap, bypass) == (0.0, 0.0, True)


def test_a_raw_gap_shorter_than_the_smoothed_one_holds_at_once():
    """Same id, gap jumped closer: the distance EMA must not buy a launch."""
    ctrl = AdaptiveCruiseController()
    raw, _ = _snap(4.0)
    _, smooth = _snap(6.5)
    assert _law(ctrl, 6.5, 0.0) >= ctrl.config.standstill_launch_accel_ms2
    a, _ = ctrl._compute_command(raw, smooth, 0.0, DT)
    assert a == 0.0


def test_a_braking_cap_eases_into_the_hold_through_the_jerk_limit():
    ctrl = AdaptiveCruiseController()
    for _ in range(100):
        _, cap, _ = _tick(ctrl, 3.5, 0.5)
    assert cap < 0.0
    a, cap_next, bypass = _tick(ctrl, 3.5, 0.3)
    assert a == 0.0
    assert not bypass
    assert cap_next < 0.0


def test_the_hold_lets_go_once_ego_is_moving():
    ctrl = AdaptiveCruiseController()
    assert _tick(ctrl, 4.6, 0.0)[0] == 0.0
    v_ego = ctrl.config.standstill_speed_ms
    a, _, _ = _tick(ctrl, 4.6, v_ego)
    assert a < 0.0
