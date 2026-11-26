"""Lane change clear: ego moving across toward a neighbour lane, a parked body clear of both lanes.

A three-lane straight road running north, lanes 4.5 m apart at x = 0, 4.5 and 9.0. Ego is on
the middle or left lane; the rule only ever judges a stopped body, and only while ego can still
settle in the lane it moves to (core/aeb/README.md, "Lane change clear").
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.filters import OutOfLaneParallelFilter
from core.aeb.lane_track import lane_change_clear
from core.aeb.road_paths import RoadPaths
from core.road_layout.ego_lane import LineIndex
from core.road_layout.ego_route import build_route
from core.road_layout.game_map import LaneWindow, pack_lane
from core.road_layout.lanes import LaneLine

from .harness import make_vehicle

_LANES_M = (0.0, 4.5, 9.0)
_SPEED = 25.0
_EGO_HW = 1.235


def _window() -> LaneWindow:
    lines = tuple(
        LaneLine("road", f"lane{k}", tuple((x, 0.0, -float(s)) for s in range(-100, 500, 5)),
                 None, 1, k, len(_LANES_M), x)
        for k, x in enumerate(_LANES_M))
    lanes = tuple(pack_lane(line) for line in lines)
    return LaneWindow(lines, (), (0, 0, 0, 0), lanes, LineIndex(lanes))


def _paths(lane: int, offset: float, drift: float, accel: float = 0.0, cal=CAL) -> RoadPaths:
    """Ego 50 m up the road on ``lane``, ``offset`` m right of it, moving right at ``drift``."""
    window = _window()
    x, z = _LANES_M[lane] + offset, -50.0
    route = build_route(window.lanes, window.index, lane, x, z, 0.0, -1.0)
    paths = RoadPaths()
    paths.frame(SimpleNamespace(route=route, window=window, target_routes={}), cal, 0.0)
    # The keep frame reads ego's lateral accel off ego's curvature against the lane's (left positive).
    assert paths.bind_ego_lane(x, z, cal, drift, -_SPEED, -accel / (_SPEED * _SPEED))
    paths.bind_lane_change(x, z, cal)
    return paths


def _parked(x: float, ahead: float = 60.0):
    return make_vehicle(9, x, -50.0 - ahead, 0.0, 0.0, 0.0, 2.5, 10.0)


def _clear(paths: RoadPaths, body, cal=CAL) -> bool:
    return lane_change_clear(paths, body, 0.0, _EGO_HW, _SPEED, cal)


def test_a_car_parked_beyond_the_lane_ego_moves_to_is_clear():
    paths = _paths(0, 0.5, 1.5)
    assert _clear(paths, _parked(9.0))


def test_a_car_parked_in_the_lane_ego_moves_to_is_kept():
    paths = _paths(0, 0.5, 1.5)
    assert not _clear(paths, _parked(4.5))
    # Straddling the line into it is no better.
    assert not _clear(paths, _parked(6.3))


def test_a_calm_ego_is_left_to_lane_clear():
    paths = _paths(0, 0.2, 0.1)
    assert not _clear(paths, _parked(9.0))


def test_moving_toward_a_side_with_no_lane_has_no_verdict():
    # Rightmost lane, moving further right: no lane to settle in, so cars parked there stay a threat.
    paths = _paths(2, 0.5, 1.5)
    assert not _clear(paths, _parked(13.0))


def test_too_fast_across_to_settle_in_the_next_lane_has_no_verdict():
    # 3 m/s across with 3 m to go needs 1.3 m/s^2 to stop by the overshoot: past the tipping point.
    paths = _paths(0, 1.5, 3.0)
    assert not _clear(paths, _parked(9.0))
    assert _clear(_paths(0, 1.5, 3.0, cal=replace(CAL, lane_change_settle_ms2=2.0)),
                  _parked(9.0), replace(CAL, lane_change_settle_ms2=2.0))


def test_settling_back_in_its_own_lane_needs_ego_slowing_across():
    # Left of its centre, moving right: back to its own centre only while it already slows.
    settling = _paths(1, -1.5, 1.0, accel=-1.0)
    assert _clear(settling, _parked(9.0))
    # Not slowing (ac7bb3ab crossed its centre toward parked cars): it heads for the next lane,
    # where the car is parked.
    drifting = _paths(1, -1.5, 1.0, accel=0.0)
    assert not _clear(drifting, _parked(9.0))


def _tick(paths: RoadPaths, lane: int, offset: float, drift: float, accel: float, now: float,
          cal=CAL, carry_only: bool = False) -> RoadPaths:
    """One more AEB frame on the same ``RoadPaths``, ego's frame bound to ``lane``."""
    window = _window()
    x, z = _LANES_M[lane] + offset, -50.0
    route = build_route(window.lanes, window.index, lane, x, z, 0.0, -1.0)
    paths.frame(SimpleNamespace(route=route, window=window, target_routes={}), cal, now)
    assert paths.bind_ego_lane(x, z, cal, drift, -_SPEED, -accel / (_SPEED * _SPEED))
    paths.bind_lane_change(x, z, cal, carry_only=carry_only)
    return paths


def test_the_lane_ego_moves_onto_carries_over_the_frame_handover():
    # 870f17e3: 2 m right of its lane heading for the next, then the frame hands over to that lane
    # with ego 2.5 m short of its centre, not yet slowing across. Still the same lane change.
    paths = _tick(RoadPaths(), 0, 2.0, 0.8, 0.0, 10.0)
    assert _clear(paths, _parked(9.0))
    _tick(paths, 1, -2.5, 0.8, 0.6, 10.05)
    assert _clear(paths, _parked(9.0))
    # It keeps carrying while ego stays on that lane, and the tipping point still binds.
    _tick(paths, 1, -2.2, 0.8, 0.3, 10.10)
    assert _clear(paths, _parked(9.0))
    _tick(paths, 1, -0.6, 2.0, 0.0, 10.15)
    assert not _clear(paths, _parked(9.0))
    # Without the carry the frame alone reads ego heading on for the car's lane.
    off = replace(CAL, lane_change_handover_carry=False)
    paths = _tick(RoadPaths(), 0, 2.0, 0.8, 0.0, 10.0, off)
    _tick(paths, 1, -2.5, 0.8, 0.6, 10.05, off)
    assert not _clear(paths, _parked(9.0), off)


def test_with_the_route_frame_lagging_only_the_lane_change_under_way_carries():
    # 870f17e3 at 5.02 s: the route still holds ego on the old lane, 2.5 m off, so only the keep
    # frame is bound. The lane change already bound carries on; nothing new binds.
    paths = _tick(RoadPaths(), 0, 2.0, 0.8, 0.0, 10.0)
    _tick(paths, 1, -2.5, 0.8, 0.6, 10.05)
    assert paths.settling
    _tick(paths, 1, -2.0, 1.2, 1.0, 10.10, carry_only=True)
    assert _clear(paths, _parked(9.0))
    # A lane change that was not under way does not start there.
    fresh = _tick(RoadPaths(), 0, 2.0, 0.8, 0.0, 10.0, carry_only=True)
    assert not fresh.settling and not _clear(fresh, _parked(9.0))
    # Nor does the old lane's neighbour bind once ego is past settling in its new lane.
    _tick(paths, 1, -0.6, 2.0, 0.0, 10.15, carry_only=True)
    assert paths.lane_change_free(_parked(9.0), 0.0, _EGO_HW) is None


def test_no_carry_from_a_stale_frame_another_lane_or_the_other_way():
    # Late: the last lane change bound 0.5 s ago.
    paths = _tick(RoadPaths(), 0, 2.0, 0.8, 0.0, 10.0)
    _tick(paths, 1, -2.5, 0.8, 0.6, 10.5)
    assert not _clear(paths, _parked(9.0))
    # Ego's frame now on a lane other than the one it was settling in.
    paths = _tick(RoadPaths(), 0, 2.0, 0.8, 0.0, 10.0)
    _tick(paths, 2, -2.5, 0.8, 0.6, 10.05)
    assert not _clear(paths, _parked(13.5))
    # Ego turned back: moving the other way is a new lane change, never the old one carried.
    paths = _tick(RoadPaths(), 0, 2.0, 0.8, 0.0, 10.0)
    last = paths._settle
    paths.frame(paths._snap, CAL, 10.05)
    assert paths._same_settle_lane(last, 1.0, 2.0, -50.0, -2.5)
    assert not paths._same_settle_lane(last, -1.0, 2.0, -50.0, -2.5)


def test_a_swerve_back_toward_a_car_on_the_side_ego_leaves_counts():
    # Moving left away from a car parked in the right lane: clear, until ego swerves back (1d7eb731).
    leaving = _paths(1, 0.3, -0.9)
    assert _clear(leaving, _parked(9.0, ahead=40.0))
    back = _paths(1, 0.3, -0.9, accel=5.0)
    assert not _clear(back, _parked(9.0, ahead=40.0))


def test_switched_off_there_is_no_lane_change_frame():
    off = replace(CAL, lane_change_clear=False)
    window = _window()
    route = build_route(window.lanes, window.index, 0, 0.5, -50.0, 0.0, -1.0)
    paths = RoadPaths()
    paths.frame(SimpleNamespace(route=route, window=window, target_routes={}), off, 0.0)
    paths.bind_ego_lane(0.5, -50.0, off, 1.5, -_SPEED)
    # The thread binds the lane change only while enabled; unbound, there is never a verdict.
    assert paths.lane_change_free(_parked(9.0), 0.0, _EGO_HW) is None


def _oolp_ctx(follow: bool, latched: bool, cal=CAL):
    v = SimpleNamespace(id=9, crash_confirmed=False)
    return SimpleNamespace(
        v=v, head_on=False, abs_v_speed=0.0, co_directional=True, cal=cal,
        follow_threat_ids={9} if follow else set(), latched_threat_ids={9} if latched else set(),
        lane_clear_s=cal.lane_clear_hold_s + 0.1)


def test_map_lane_clear_outranks_the_follow_flag_for_a_stopped_body_but_never_a_latch():
    stage = OutOfLaneParallelFilter(CAL)
    assert stage.apply(_oolp_ctx(follow=True, latched=False)).suppressed
    assert not stage.apply(_oolp_ctx(follow=True, latched=True)).suppressed
    old = replace(CAL, lane_clear_over_follow=False)
    assert not OutOfLaneParallelFilter(old).apply(_oolp_ctx(True, False, old)).suppressed
