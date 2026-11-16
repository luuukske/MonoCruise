"""Lane track and lane clear: the map as a second opinion on a stage's verdict.

The lane frame is positional: metres ahead along ego's lane, metres right of ego's
own track. A body held on that track is kept where a stage would drop it; with no
map there is no sight and every stage decides as before.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from types import SimpleNamespace

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.calibration import ego_path_params
from core.aeb.clip_eval import _make_headless
from core.aeb.filters import lane_clear_held
from core.aeb.lane_track import (
    LaneSight, LaneTrackHold, clear_of_ego_track, lane_drivable, lane_sight, on_ego_track,
)
from core.aeb.road_paths import RoadPaths
from core.road_layout.ego_route import build_route
from core.road_layout.road_bend import fast_offset, lateral_offset, road_curve

from .harness import make_vehicle
from .test_road_bent_scenarios import _DT, _STRAIGHT, _Lane

_BAR = 2.5


def _bound(lane: _Lane, s: float, offset: float = 0.0, speed: float = 20.0) -> RoadPaths:
    """Road paths bound to ego at arc length ``s``, ``offset`` m right of the lane."""
    window = lane.window()
    x, z, tx, tz = lane.point(s)
    x += offset * (-tz)
    z += offset * tx
    route = build_route(window.lanes, window.index, 0, x, z, tx, tz)
    paths = RoadPaths()
    paths.frame(SimpleNamespace(route=route, window=window, target_routes={}), CAL, 0.0)
    paths.bind_ego_lane(x, z, CAL, speed * tx, speed * tz)
    return paths


def _at(lane: _Lane, s: float, offset: float) -> tuple[float, float]:
    x, z, tx, tz = lane.point(s)
    return x + offset * (-tz), z + offset * tx


def test_without_a_map_there_is_no_lane_frame():
    paths = RoadPaths()
    paths.frame(None, CAL, 0.0)
    assert not paths.bind_ego_lane(0.0, 0.0, CAL)
    assert paths.lane_offset(0.0, -50.0) is None


def test_the_lane_frame_follows_the_bend():
    lane = _Lane(250.0)
    paths = _bound(lane, 100.0)
    for offset in (0.0, 4.5, -4.5):
        ahead, right, _tx, _tz = paths.lane_offset(*_at(lane, _STRAIGHT + 60.0, offset))
        assert abs(ahead - 110.0) < 3.0
        assert abs(right - offset) < 0.4, offset


def test_the_frame_is_measured_from_egos_own_track():
    lane = _Lane(250.0)
    paths = _bound(lane, 100.0, offset=1.0)
    _ahead, right, _tx, _tz = paths.lane_offset(*_at(lane, 140.0, 0.0))
    assert abs(right + 1.0) < 0.2


def test_ego_off_its_lane_binds_no_frame():
    lane = _Lane(250.0)
    paths = _bound(lane, 100.0, offset=3.0)
    assert paths.lane_offset(*_at(lane, 140.0, 0.0)) is None


def test_behind_ego_is_out_of_the_frame():
    lane = _Lane(250.0)
    paths = _bound(lane, 100.0)
    assert paths.lane_offset(*_at(lane, 80.0, 0.0)) is None


def test_the_fast_lane_lookup_gives_the_full_search_answer():
    for radius in (60.0, 150.0, 250.0):
        lane = _Lane(radius)
        curve = road_curve(_bound(lane, 100.0)._snap.route.pts)
        rng = random.Random(7)
        for _ in range(400):
            x, z = _at(lane, rng.uniform(60.0, 330.0), rng.uniform(-12.0, 12.0))
            slow, fast = lateral_offset(curve, x, z), fast_offset(curve, x, z)
            assert abs(slow[0] - fast[0]) < 1e-9 and abs(slow[1] - fast[1]) < 1e-9


def test_a_bend_too_tight_for_egos_speed_is_not_a_lane_ego_will_keep():
    paths = _bound(_Lane(60.0), 100.0)
    # R 60 m ahead: fine at 50 km/h, beyond a truck at 90.
    assert lane_drivable(paths, 120.0, 14.0, CAL)
    assert not lane_drivable(paths, 120.0, 25.0, CAL)
    # The straight before the bend is drivable at any speed.
    assert lane_drivable(paths, 20.0, 25.0, CAL)
    assert lane_drivable(_bound(_Lane(250.0), 100.0), 120.0, 25.0, CAL)


def _bound_with(lane: _Lane, divergence) -> RoadPaths:
    """Ego at 100 m on ``lane`` with the route's branch ``divergence`` set by hand."""
    window = lane.window()
    x, z, tx, tz = lane.point(100.0)
    route = replace(build_route(window.lanes, window.index, 0, x, z, tx, tz), divergence=divergence)
    paths = RoadPaths()
    paths.frame(SimpleNamespace(route=route, window=window, target_routes={}), CAL, 0.0)
    paths.bind_ego_lane(x, z, CAL, 20.0 * tx, 20.0 * tz)
    return paths


def test_a_branch_ends_the_frame_but_a_lane_starting_ahead_of_ego_does_not():
    lane = _Lane(250.0)
    beside = _at(lane, 140.0, 4.5)
    fork = _bound_with(lane, ((0.0, 0.0), (5.0, 0.2), (10.0, 1.7), (15.0, 3.0)))
    assert fork.lane_offset(*beside) is None
    # Apart at the first sample only, together again 5 m on: no branch.
    start = _bound_with(lane, ((0.0, 1.7), (5.0, 0.01), (10.0, 0.0), (15.0, 0.0)))
    assert start.lane_offset(*beside) is not None
    # Apart from the start and staying apart is a branch.
    apart = _bound_with(lane, ((0.0, 1.7), (5.0, 1.6), (10.0, 2.0)))
    assert apart.lane_offset(*beside) is None
    assert _bound_with(lane, ((0.0, 1.7),)).lane_offset(*beside) is None


def test_a_body_is_on_egos_track_only_with_real_overlap():
    assert on_ego_track(LaneSight(40.0, 1.0, 1.0, 1.0), False, _BAR, CAL)
    # Passing with a hand's width to spare is not on the track.
    assert not on_ego_track(LaneSight(40.0, 2.4, 2.4, 1.0), False, _BAR, CAL)
    assert not on_ego_track(None, False, _BAR, CAL)
    assert not on_ego_track(LaneSight(CAL.lane_track_range_m + 1.0, 0.0, 0.0, 1.0), False, _BAR, CAL)


def test_a_mover_is_on_the_track_only_along_the_lane():
    assert on_ego_track(LaneSight(40.0, 0.5, 0.5, -0.97), True, _BAR, CAL)
    assert not on_ego_track(LaneSight(40.0, 0.5, 0.5, 0.2), True, _BAR, CAL)
    # Parked across the lane still blocks it.
    assert on_ego_track(LaneSight(40.0, 0.5, 0.5, 0.2), False, _BAR, CAL)


def test_clear_of_the_track_needs_room_and_ego_calm_in_its_lane():
    beside = LaneSight(50.0, 4.8, 4.8, 1.0)
    assert clear_of_ego_track(beside, _BAR, 0.2, 0.0, 25.0, CAL)
    # Moving across its lane at 1 m/s, either way, ego is not keeping it: no verdict.
    assert not clear_of_ego_track(beside, _BAR, 0.2, 1.0, 25.0, CAL)
    assert not clear_of_ego_track(beside, _BAR, 0.2, -1.0, 25.0, CAL)
    calm = CAL.lane_clear_max_drift_ms - 0.05
    assert clear_of_ego_track(beside, _BAR, 0.2, calm, 25.0, CAL)
    assert clear_of_ego_track(beside, _BAR, 0.2, -calm, 25.0, CAL)
    # The same for a body on the left.
    left = LaneSight(50.0, -4.8, 4.8, 1.0)
    assert not clear_of_ego_track(left, _BAR, 0.2, -1.0, 25.0, CAL)
    assert clear_of_ego_track(left, _BAR, -0.2, -calm, 25.0, CAL)
    # With a looser calm bar the drift toward it must still leave room on arrival.
    loose = replace(CAL, lane_clear_max_drift_ms=2.0)
    assert not clear_of_ego_track(beside, _BAR, 0.2, 1.5, 25.0, loose)
    assert clear_of_ego_track(beside, _BAR, 0.2, -1.5, 25.0, loose)
    assert not clear_of_ego_track(LaneSight(50.0, 3.5, 3.5, 1.0), _BAR, 0.2, 0.0, 25.0, CAL)
    assert not clear_of_ego_track(beside, _BAR, CAL.lane_clear_max_offset_m + 0.1, 0.0, 25.0, CAL)
    assert not clear_of_ego_track(None, _BAR, 0.0, 0.0, 25.0, CAL)


def test_lane_clear_preserves_latched_and_crash_confirmed_bodies():
    cal = replace(CAL, lane_clear_stationary=True)
    vehicle = SimpleNamespace(id=7, crash_confirmed=False)
    ctx = SimpleNamespace(v=vehicle, lane_clear_s=cal.lane_clear_hold_s, latched_threat_ids=set())
    assert lane_clear_held(ctx, cal)
    ctx.latched_threat_ids.add(vehicle.id)
    assert not lane_clear_held(ctx, cal)
    ctx.latched_threat_ids.clear()
    vehicle.crash_confirmed = True
    assert not lane_clear_held(ctx, cal)


def test_the_hold_counts_unbroken_time_only():
    hold = LaneTrackHold()
    assert hold.step(1, True, 10.0) == 0.0
    assert abs(hold.step(1, True, 10.1) - 0.1) < 1e-9
    assert hold.step(1, False, 10.2) == 0.0
    assert hold.step(1, True, 10.3) == 0.0
    # Not judged for a while (out of range): the hold starts over.
    assert abs(hold.step(1, True, 10.4) - 0.1) < 1e-9
    assert hold.step(1, True, 11.0) == 0.0
    hold.prune(set())
    assert hold.step(1, True, 11.1) == 0.0


def test_lane_sight_reads_a_long_body_by_its_nearest_end():
    lane = _Lane(250.0)
    paths = _bound(lane, 100.0)
    x, z = _at(lane, 140.0, 3.0)
    # Angled 25 degrees across the lane: the nose reaches toward ego's track.
    rig = make_vehicle(7, x, z, 25.0, 0.0, width=2.5, length=14.0)
    sight = lane_sight(paths, rig, math.radians(25.0), 1.0)
    assert sight is not None
    assert abs(sight.gap - 3.0) < 0.3
    assert sight.reach < 0.5


def _run_oncoming(offset: float, with_map: bool, cal=CAL):
    """Ego at 20 m/s on the straight, a car driving at it ``offset`` m right of the lane."""
    lane = _Lane(250.0)
    t = _make_headless(cal)
    t._read_vehicle_key = lambda: None
    t._read_ego_geometry = lambda: None
    t._read_max_brake_ms2 = lambda: 7.0
    t._read_user_braking = lambda: False
    t._read_addressing_brake = lambda: False
    window = lane.window()
    first_brake = None
    for k in range(int(3.5 / _DT)):
        now = 100.0 + k * _DT
        s_ego = 20.0 * k * _DT
        s_car = 140.0 - 15.0 * k * _DT
        if s_car - s_ego < 6.0:
            break
        car = make_vehicle(7, offset, -s_car, 180.0, 15.0, width=2.5, length=6.0)
        steer = 0.0 / ego_path_params(cal).gain_prior
        snap = ([car], 0.0, 0.0, -s_ego, 0.0, 20.0, 0.0, steer, False, None, False, False,
                now, frozenset(), now, frozenset())
        t._read_radar_snapshot = lambda snap=snap: snap
        t._now = lambda now=now: now
        if with_map:
            route = build_route(window.lanes, window.index, 0, 0.0, -s_ego, 0.0, -1.0)
            map_snap = SimpleNamespace(route=route, window=window, target_routes={})
            t._read_map_snapshot = lambda m=map_snap: m
        t.loop()
        if first_brake is None and t.data.AEB_brake:
            first_brake = k * _DT
    return first_brake


def test_a_wrong_way_car_on_egos_track_is_never_braked_for_later_with_the_map():
    base = _run_oncoming(0.0, with_map=False)
    mapped = _run_oncoming(0.0, with_map=True)
    assert mapped is not None
    assert base is None or mapped <= base


def test_a_half_overlapping_head_on_is_braked_for_with_the_map():
    # 2 m off ego's line the bodies still overlap by half a metre; the pose alone reads "own lane".
    assert _run_oncoming(-2.0, with_map=False) is None
    assert _run_oncoming(-2.0, with_map=True) is not None


def test_an_oncoming_car_in_the_next_lane_is_left_exactly_as_today():
    for offset in (-3.9, -4.5):
        assert _run_oncoming(offset, with_map=True) == _run_oncoming(offset, with_map=False)


def test_the_map_is_a_second_opinion_only_while_enabled():

    off = replace(CAL, lane_track_enabled=False)
    assert _run_oncoming(0.0, with_map=True, cal=off) == _run_oncoming(0.0, with_map=False, cal=off)
