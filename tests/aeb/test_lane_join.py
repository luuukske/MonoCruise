"""Lane join: a mover heading for the lane beside ego's, able to stop on its centre, is let go.

A three-lane straight road running north, lanes 4.5 m apart at x = 0, 4.5 and 9.0, as in
``test_lane_change_clear.py`` (core/aeb/README.md, "Lane join").
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.lane_join import (
    LaneJoinFilter, LaneJoinTracker, lane_join_frame, lane_join_sight, settles,
)
from core.aeb.road_paths import RoadPaths
from core.road_layout.ego_lane import LineIndex
from core.road_layout.ego_route import build_route
from core.road_layout.game_map import LaneWindow, pack_lane
from core.road_layout.lanes import LaneLine

from .harness import make_vehicle

_LANES_M = (0.0, 4.5, 9.0)
_SPEED = 25.0


def _window() -> LaneWindow:
    lines = tuple(
        LaneLine("road", f"lane{k}", tuple((x, 0.0, -float(s)) for s in range(-100, 500, 5)),
                 None, 1, k, len(_LANES_M), x)
        for k, x in enumerate(_LANES_M))
    lanes = tuple(pack_lane(line) for line in lines)
    return LaneWindow(lines, (), (0, 0, 0, 0), lanes, LineIndex(lanes))


def _paths(lane: int = 0, offset: float = 0.0, drift: float = 0.0, now: float = 0.0,
           paths: RoadPaths | None = None) -> tuple[RoadPaths, bool]:
    """Ego 50 m up the road on ``lane``, ``offset`` m right of it, moving right at ``drift``."""
    window = _window()
    x, z = _LANES_M[lane] + offset, -50.0
    route = build_route(window.lanes, window.index, lane, x, z, 0.0, -1.0)
    paths = paths or RoadPaths()
    paths.frame(SimpleNamespace(route=route, window=window, target_routes={}), CAL, now)
    assert paths.bind_ego_lane(x, z, CAL, drift, -_SPEED)
    paths.bind_lane_change(x, z, CAL)
    return paths, lane_join_frame(paths, x, z, CAL)


def _car(x: float, ahead: float = 30.0):
    return make_vehicle(9, x, -50.0 - ahead, 0.0, 20.0)


def test_c01e50a2_settles_and_a_late_or_fast_one_does_not():
    # 2.5 m out at 1.75 m/s needs 0.6 m/s^2 to stop on the next lane's centre: it settles.
    assert settles(2.5, 1.75, CAL)
    # 0.5 m out at the same speed needs 3.1 m/s^2: a cut-in.
    assert not settles(0.5, 1.75, CAL)
    # Twice its lateral speed is the bypass, whatever the room.
    assert not settles(8.0, 3.6, CAL)
    # Moving away or still, it settles; on or past the centre toward ego it never does.
    assert settles(2.0, -1.0, CAL) and settles(2.0, 0.0, CAL)
    assert not settles(0.0, 0.0, CAL) and not settles(-0.3, 0.0, CAL)


def test_the_next_lane_is_the_one_beside_ego_on_the_body_side():
    paths, frame = _paths()
    assert frame

    def sight(car, travel=1.0):
        return lane_join_sight(paths, frame, car, 0.0, travel, 1.25, CAL)
    side, beyond, ahead, along, spacing, changing, stop = sight(_car(9.0))
    assert side == 1.0 and abs(beyond - 4.5) < 0.1 and abs(ahead - 30.0) < 0.5 and along > 0.99
    assert abs(spacing - 4.5) < 0.1 and not changing
    # Ego centred: the car may come 4.5 - 1.25 - 1.25 - 1.0 = 1.0 m past the next lane's centre.
    assert abs(stop - 1.0) < 0.1
    # Straddling the next lane's far line, 0.5 m beyond its centre.
    assert abs(sight(_car(5.0))[1] - 0.5) < 0.1
    # In the next lane, ego side of its centre: nothing beyond.
    assert sight(_car(4.0))[1] < 0.0
    # Oncoming reads against the lane.
    assert sight(_car(9.0), -1.0)[3] < -0.99


def test_where_the_next_lane_has_merged_into_egos_a_body_there_is_in_egos_lane():
    # 6457d079: the right lane runs into ego's 50 to 110 m ahead of ego; a car 0.3 m right of
    # ego's lane there read 0.3 m beyond the next lane's centre and was let go.
    def x_right(s: float) -> float:
        return 4.5 * min(1.0, max(0.0, (160.0 - s) / 60.0))
    lines = (
        LaneLine("road", "lane0", tuple((0.0, 0.0, -float(s)) for s in range(-100, 500, 5)),
                 None, 1, 0, 2, 0.0),
        LaneLine("road", "lane1", tuple((x_right(s), 0.0, -float(s)) for s in range(-100, 500, 5)),
                 None, 1, 1, 2, 4.5),
    )
    lanes = tuple(pack_lane(line) for line in lines)
    window = LaneWindow(lines, (), (0, 0, 0, 0), lanes, LineIndex(lanes))
    route = build_route(window.lanes, window.index, 0, 0.0, -50.0, 0.0, -1.0)
    paths = RoadPaths()
    paths.frame(SimpleNamespace(route=route, window=window, target_routes={}), CAL, 0.0)
    assert paths.bind_ego_lane(0.0, -50.0, CAL, 0.0, -_SPEED)
    frame = lane_join_frame(paths, 0.0, -50.0, CAL)
    assert frame
    assert lane_join_sight(paths, frame, make_vehicle(9, 0.3, -180.0, 0.0, 20.0), 0.0, 1.0, 1.25, CAL) is None
    # Before the merge the same lane is judged as usual.
    assert lane_join_sight(paths, frame, make_vehicle(9, 9.0, -80.0, 0.0, 20.0), 0.0, 1.0, 1.25, CAL)


def test_ego_moving_across_with_no_lane_change_has_no_lane_and_a_lane_change_moves_it():
    # Drifting off centre with nowhere to settle (rightmost lane, moving right).
    assert _paths(2, 0.5, 1.0)[1] is None
    # Off centre toward the body by more than lane keep's bar, calm: no lane either.
    assert _paths(0, 1.0, 0.0)[1] is None
    # Changing lanes right from the left lane: ego's lane is the middle one, so a car centred
    # in the right lane sits on the next lane's centre.
    paths, frame = _paths(0, 1.0, 1.0)
    assert frame
    assert abs(lane_join_sight(paths, frame, _car(9.0), 0.0, 1.0, 1.25, CAL)[1]) < 0.1


def _feed(tracker: LaneJoinTracker, start: float, rate: float, seconds: float, t0: float = 0.0,
          speed: float = 20.0, accel: float = 0.0) -> float:
    """The car's distance beyond the next lane's centre shrinking at ``rate``, 30 Hz."""
    held, t = 0.0, t0
    while t <= t0 + seconds + 1e-9:
        beyond = start - rate * (t - t0)
        held = tracker.step(9, (1.0, beyond, 30.0, 1.0, 4.5, False, 0.0), speed, t, CAL, accel=accel)
        t += 1.0 / 30.0
    return held


def test_the_tracker_reads_the_rate_off_positions_and_holds_before_it_lets_go():
    # c01e50a2: from 4.5 m beyond at 1.75 m/s. Half the fit window plus the hold.
    tracker = LaneJoinTracker()
    assert _feed(tracker, 4.5, 1.75, 0.3) < CAL.lane_join_hold_s
    assert _feed(tracker, 4.5 - 1.75 * 0.3, 1.75, 0.3, t0=0.3) >= CAL.lane_join_hold_s
    # Once close enough that settling needs more than the bar, it is a cut-in at once.
    assert _feed(tracker, 0.9, 1.75, 0.05, t0=0.6) == 0.0


def test_the_tracker_restarts_on_a_jump_a_gap_or_a_slow_body():
    tracker = LaneJoinTracker()
    _feed(tracker, 4.5, 1.0, 0.6)
    # The lane read jumped across by more than a body can move in a frame: a new track.
    assert tracker.step(9, (1.0, 2.0, 30.0, 1.0, 4.5, False, 0.0), 20.0, 0.64, CAL) == 0.0
    tracker = LaneJoinTracker()
    assert _feed(tracker, 4.5, 1.0, 0.6, speed=3.0) == 0.0


def test_only_a_body_that_came_from_beyond_the_next_lanes_far_edge_is_let_go():
    # bca96716: a truck already in the next lane, 1.3 m beyond its centre, starting across.
    assert _feed(LaneJoinTracker(), 1.3, 0.5, 1.0) == 0.0
    # The same reading for a body that crossed the far edge (2.25 m) on its way in.
    tracker = LaneJoinTracker()
    _feed(tracker, 2.6, 1.0, 0.4)
    assert _feed(tracker, 2.2, 0.5, 1.0, t0=0.4) >= CAL.lane_join_hold_s
    # Where it came from counts for 3 s, then it is a body in the next lane like any other.
    tracker = LaneJoinTracker()
    _feed(tracker, 2.6, 1.0, 0.4)
    assert _feed(tracker, 2.2, 0.0, 3.5, t0=0.4) == 0.0


def test_where_it_came_from_is_forgotten_when_the_lane_read_jumps_across_a_gap():
    # ffb1688a: two lanes over, then no sight while ego's lane moved over by one; the truck now
    # reads 0.4 m beyond the next lane's centre. That is a truck in the next lane, not a join.
    tracker = LaneJoinTracker()
    _feed(tracker, 4.9, 0.0, 0.5)
    tracker.step(9, None, 20.0, 0.55, CAL)
    assert _feed(tracker, 0.4, 0.0, 1.0, t0=0.6) == 0.0


def test_a_swerve_reads_its_rate_now_and_is_a_cut_in():
    # 02e3a612: 3.2 m out, then 0 to 4 m/s across in 0.3 s. The window's mean rate still read 2 m/s
    # and let it go; the rate now passes the 3.5 m/s bypass before AEB brakes.
    tracker = LaneJoinTracker()
    assert _feed(tracker, 3.2, 0.0, 0.6) >= CAL.lane_join_hold_s
    held, t = None, 0.6
    while t <= 0.95:
        u = t - 0.6
        held = tracker.step(9, (1.0, 3.2 - 6.5 * u * u, 30.0, 1.0, 4.5, False, 0.0), 20.0, t, CAL)
        t += 1.0 / 30.0
    assert held == 0.0


def test_ego_crossing_fast_still_has_its_lane_if_it_can_settle_a_little_past_the_centre():
    # ffb1688a: 1.7 m short of its new lane's centre at 3 m/s across. Settling at 3 m/s^2 within
    # 1 m past the centre needs 1.5 m of the 2.7 m it has.
    paths, frame = _paths(1, -1.7, 3.0)
    assert frame and frame[3] == 1.0
    # 0.8 m short at 4 m/s it would need 2.7 m of 1.8: past settling, no lane.
    assert _paths(1, -0.8, 4.0)[1] is None


def test_a_body_crossing_steeply_or_braking_hard_is_not_let_go():
    # From 4.5 m out at 2 m/s across it settles at road speed.
    assert _feed(LaneJoinTracker(), 4.5, 2.0, 0.6) >= CAL.lane_join_hold_s
    # 39d96205: the same 2 m/s across at 6 m/s along is a crosser, not a lane change.
    assert _feed(LaneJoinTracker(), 4.5, 2.0, 0.6, speed=6.0) == 0.0
    # e0638fb2: braking at 4 m/s^2 beside ego's lane.
    assert _feed(LaneJoinTracker(), 4.5, 1.0, 0.6, accel=-4.0) == 0.0
    assert _feed(LaneJoinTracker(), 4.5, 1.0, 0.6, accel=-2.0) >= CAL.lane_join_hold_s


def test_ego_still_pushing_across_near_the_centre_it_settles_in_has_no_lane():
    def frame(offset: float, accel: float):
        paths, _ = _paths(1, offset, 1.5)
        keep = paths.keep_lane
        fake = SimpleNamespace(keep_lane=keep[:6] + (accel,), active=True, neighbour_lane=paths.neighbour_lane)
        return lane_join_frame(fake, _LANES_M[1] + offset, -50.0, CAL)
    # 7f002208: 0.5 m short of its lane's centre, drift still growing at 3 m/s^2: it swerves on.
    assert frame(-0.5, 0.0) and frame(-0.5, -2.0)
    assert frame(-0.5, 3.0) is None
    # ffb1688a: pushing as hard 1.7 m short is the first half of a lane change.
    assert frame(-1.7, 3.0)


def _ctx(held: float, latched: bool = False, head_on: bool = False):
    return SimpleNamespace(v=SimpleNamespace(id=9, crash_confirmed=False), lane_join_s=held,
                           latched_threat_ids={9} if latched else set(), head_on=head_on)


def test_the_stage_drops_only_after_the_hold_and_never_a_latched_or_oncoming_body():
    stage = LaneJoinFilter(CAL)
    assert stage.apply(_ctx(0.3)).suppressed
    assert not stage.apply(_ctx(0.2)).suppressed
    assert not stage.apply(_ctx(0.3, latched=True)).suppressed
    assert not stage.apply(_ctx(0.3, head_on=True)).suppressed
    assert not LaneJoinFilter(replace(CAL, lane_join_enabled=False)).apply(_ctx(1.0)).suppressed
