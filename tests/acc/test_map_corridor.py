"""ACC corridor with the map path: a shoulder car at a bend leaves it, a lane car stays.

Ego at the origin heading north (-Z), right is +X. The lane runs straight to
5 m ahead, then bends right with R = 300 m. No map files, no clips.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from core.acc.tracker import ACCTracker
from core.road_layout.ego_lane import LineIndex
from core.road_layout.ego_route import build_route
from core.road_layout.game_map import LaneWindow, pack_lane
from core.road_layout.lanes import LaneLine

from .harness import make_vehicle

_DT = 1.0 / 30.0
_R = 300.0
_BEND_AT = 5.0


def _lane_point(s):
    if s <= _BEND_AT:
        return 0.0, -s
    a = (s - _BEND_AT) / _R
    return _R * (1.0 - math.cos(a)), -_BEND_AT - _R * math.sin(a)


def _map():
    pts = [_lane_point(float(s)) for s in range(-100, 300, 5)]
    line = LaneLine("road", "t", tuple((x, 0.0, z) for x, z in pts), None, 1, 0, 1, 0.0)
    lanes = (pack_lane(line),)
    window = LaneWindow((line,), (), (0, 0, 0, 0), lanes, LineIndex(lanes))
    route = build_route(lanes, window.index, 0, 0.0, 0.0, 0.0, -1.0)
    return SimpleNamespace(route=route, window=window)


def _track(vehicle, map_snapshot, frames=45):
    tracker = ACCTracker()
    for i in range(frames):
        tracker.update(
            now_mono=100.0 + i * _DT, dt=_DT, vehicles=[vehicle],
            ego_x=0.0, ego_z=0.0, ego_yaw_rad=0.0,
            ego_speed_ms=22.0, ego_steer=0.0, ego_history_kappa=0.0,
            blinker_left=False, blinker_right=False, map_snapshot=map_snapshot,
        )
    return tracker.tracks[vehicle.id]


def test_shoulder_car_straight_ahead_of_a_bend_leaves_the_corridor_with_the_map():
    car = make_vehicle(3, 0.0, -50.0, 0.0, history_speed=0.0)
    assert _track(car, None).in_path_unshifted
    assert not _track(car, _map()).in_path_unshifted


def test_car_on_the_lane_in_the_bend_stays_in_the_corridor_with_the_map():
    x, z = _lane_point(50.0)
    yaw = -math.atan2(x - _lane_point(45.0)[0], -(z - _lane_point(45.0)[1]))
    car = make_vehicle(4, x, z, 15.0, yaw_rad=yaw)
    assert _track(car, _map()).in_path_unshifted


def test_no_map_is_bit_identical():
    car = make_vehicle(5, 1.0, -60.0, 12.0)
    a = _track(car, None)
    b = _track(make_vehicle(5, 1.0, -60.0, 12.0), None)
    assert a.score == b.score and a.last_road_lat == b.last_road_lat
