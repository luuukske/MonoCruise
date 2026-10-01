"""Geometric lead failsafe: what it rescues, and everything it must not.

The rescue exists because the road model can lose a lead on a layout it fits
badly. Its whole value is that it runs on raw ego-arc geometry, so these tests
drive the tracker with a road model that has been poisoned into disagreement
and check the lead survives. See core/acc/README.md §10."""
from __future__ import annotations

import math

import pytest

from core.acc import failsafe
from core.acc.failsafe import FailsafeInputs, FailsafeState, failsafe_step
from core.acc.scoring import IN_PATH_THRESHOLD
from core.acc.tracker import ACCTracker, effective_score

from .harness import make_vehicle


_DT = 1.0 / 30.0
_EGO_YAW = 0.0


def _run(tracker, vehicles, frames, ego_speed=20.0, start_t=100.0, steer=0.0):
    for i in range(frames):
        tracker.update(
            now_mono=start_t + i * _DT, dt=_DT, vehicles=vehicles,
            ego_x=0.0, ego_z=0.0,
            ego_yaw_rad=_EGO_YAW,
            ego_speed_ms=ego_speed, ego_steer=steer,
            ego_history_kappa=0.0,
            blinker_left=False, blinker_right=False,
        )
    return tracker.tracks


def _ahead(vid, dist_m, speed, lat=0.0, **kw):
    return make_vehicle(vid, lat, -dist_m, speed, yaw_rad=_EGO_YAW, **kw)


def _inputs(**kw):
    base = dict(
        dist_m=25.0, body_lat_min=-1.2, body_lat_max=1.2,
        lat_uncertainty_m=0.0, yaw_diff_deg=0.0,
        ego_speed_ms=20.0, lead_speed_ms=20.0, desired_th_s=1.5,
        recently_led=False, suppressed=False,
    )
    base.update(kw)
    return FailsafeInputs(**base)


def _hold(state, inp, seconds):
    """Feed one input for ``seconds`` and return the last published floor."""
    floor = 0.0
    for _ in range(int(seconds / _DT)):
        floor = failsafe_step(state, inp, _DT)
    return floor


def test_dead_ahead_close_lead_earns_full_authority():
    state = FailsafeState()
    assert _hold(state, _inputs(), 2.0) == pytest.approx(failsafe.score_floor())
    assert state.reason == "close"


def test_authority_is_not_granted_before_the_confirm_window():
    """A single bad frame of geometry must never publish a lead."""
    state = FailsafeState()
    assert failsafe_step(state, _inputs(), _DT) == 0.0
    assert state.authority == 0.0


def test_entry_and_release_are_ramps_not_steps():
    state = FailsafeState()
    _hold(state, _inputs(), 2.0)
    first_drop = failsafe.failsafe_decay(state, _DT)
    assert 0.0 < first_drop < failsafe.score_floor()
    # Release is slower than entry: dropping a real lead is the costlier way
    # to be wrong.
    assert failsafe.RAMP_DOWN_S > failsafe.RAMP_UP_S


def test_adjacent_lane_vehicle_is_never_rescued():
    state = FailsafeState()
    beside = _inputs(body_lat_min=2.4, body_lat_max=4.8)
    assert _hold(state, beside, 2.0) == 0.0


def test_oncoming_vehicle_is_never_rescued():
    state = FailsafeState()
    assert _hold(state, _inputs(yaw_diff_deg=175.0), 2.0) == 0.0


def test_far_lead_that_nothing_depends_on_is_not_rescued():
    """Dead ahead, same speed, beyond the close band: no reason, no rescue."""
    state = FailsafeState()
    far = _inputs(dist_m=48.0, ego_speed_ms=20.0, lead_speed_ms=20.0)
    assert _hold(state, far, 2.0) == 0.0
    assert state.reason == ""


def test_the_same_far_lead_is_rescued_once_ego_starts_closing():
    state = FailsafeState()
    closing = _inputs(dist_m=48.0, ego_speed_ms=20.0, lead_speed_ms=15.0)
    assert _hold(state, closing, 2.0) == pytest.approx(failsafe.score_floor())


def test_range_cap_holds_where_no_estimator_resolves_a_lane():
    state = FailsafeState()
    beyond = _inputs(dist_m=failsafe.MAX_DIST_M + 5.0, lead_speed_ms=0.0)
    assert _hold(state, beyond, 2.0) == 0.0


def test_uncertainty_still_shrinks_the_gate():
    """The sigma asymmetry of the main gate is not dropped by the rescue."""
    state = FailsafeState()
    edge = _inputs(body_lat_min=1.9, body_lat_max=4.3, lat_uncertainty_m=0.9)
    assert _hold(state, edge, 2.0) == 0.0


def test_suppressed_targets_are_not_rescued():
    """Blinker candidates and the lane ego is leaving are the controller's."""
    state = FailsafeState()
    assert _hold(state, _inputs(suppressed=True), 2.0) == 0.0


def test_recently_scored_lead_confirms_faster_than_a_cold_one():
    cold = FailsafeState()
    warm = FailsafeState()
    span = failsafe.CONFIRM_S - 0.05
    _hold(cold, _inputs(), span)
    _hold(warm, _inputs(recently_led=True), span)
    assert cold.authority == 0.0
    assert warm.authority > 0.0


def _poison_road_model(tracker):
    """Force the tracker's road model to insist the lane is somewhere else.

    This is the failure the rescue exists for: a confident centreline several
    metres off the lane ego is actually in."""
    from core.acc.road_model import RoadModel

    tracker._road_smoother.step = lambda road, *a, **kw: RoadModel(
        c1=20.0, base_kappa=0.0, confidence=1.0, support_s_m=150.0,
    )


def test_lead_survives_a_road_model_that_puts_the_lane_elsewhere():
    tracker = ACCTracker()
    lead = _ahead(1, 22.0, 18.0)
    _run(tracker, [lead], frames=45)
    assert tracker.tracks[1].score > IN_PATH_THRESHOLD

    _poison_road_model(tracker)
    tracks = _run(tracker, [lead], frames=90, start_t=102.0)
    assert tracks[1].score <= IN_PATH_THRESHOLD, (
        "fixture no longer reproduces the tracking loss it was written for"
    )
    assert tracks[1].failsafe.authority > 0.0
    assert effective_score(tracks[1]) > IN_PATH_THRESHOLD


def test_rescued_lead_is_published_with_consumer_grade_confidence():
    """A rescue the controller gives no authority is the same as no rescue."""
    from core.cruise_control_thread.acc_controller import ANT_SCORE_MIN

    tracker = ACCTracker()
    lead = _ahead(2, 20.0, 15.0)
    _run(tracker, [lead], frames=45)
    _poison_road_model(tracker)
    _run(tracker, [lead], frames=90, start_t=102.0)

    leads = tracker._top_leads({2: lead}, [lead], 0.0, -1.0, 20.0)
    assert [item.vehicle.id for item in leads] == [2]
    assert leads[0].score > ANT_SCORE_MIN


def test_rescue_does_not_touch_the_scorers_own_score():
    """The floor is applied at publish time so release is never hooked."""
    tracker = ACCTracker()
    lead = _ahead(3, 20.0, 15.0)
    _run(tracker, [lead], frames=45)
    _poison_road_model(tracker)
    _run(tracker, [lead], frames=60, start_t=102.0)
    st = tracker.tracks[3]
    assert st.score < st.failsafe_score


def test_adjacent_lane_traffic_is_not_rescued_by_the_tracker():
    tracker = ACCTracker()
    beside = _ahead(4, 30.0, 22.0, lat=3.6)
    tracks = _run(tracker, [beside], frames=90)
    assert tracks[4].failsafe.authority == 0.0
    assert effective_score(tracks[4]) <= IN_PATH_THRESHOLD


def test_oncoming_traffic_is_not_rescued_by_the_tracker():
    tracker = ACCTracker()
    oncoming = make_vehicle(5, 0.0, -35.0, 22.0, yaw_rad=math.pi)
    tracks = _run(tracker, [oncoming], frames=90)
    assert tracks[5].failsafe.authority == 0.0


def test_rescue_releases_once_the_vehicle_leaves_the_corridor():
    tracker = ACCTracker()
    lead = _ahead(6, 20.0, 15.0)
    _run(tracker, [lead], frames=45)
    _poison_road_model(tracker)
    _run(tracker, [lead], frames=60, start_t=102.0)
    assert tracker.tracks[6].failsafe.authority > 0.0

    gone = _ahead(6, 20.0, 15.0, lat=4.5)
    tracks = _run(tracker, [gone], frames=60, start_t=104.0)
    assert tracks[6].failsafe.authority == 0.0
    assert tracks[6].failsafe_score == 0.0
