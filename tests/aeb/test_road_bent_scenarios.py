"""Road-bent trajectories, closed loop through the headless AEB thread with a synthetic map.

World X east, Z south. Ego drives north (-Z) along one lane: straight for 150 m,
then a right-hand bend. Ego follows the lane exactly, steering the bend's
curvature once it is in it, unless a scenario has it drive straight on. Each
scenario runs once without a map and once with one, and compares when AEB first
warns and brakes. The map may only add or advance braking for what the road puts
in ego's lane; everything else must come out exactly as today.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from types import SimpleNamespace

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.clip_eval import _make_headless
from core.aeb.calibration import ego_path_params
from core.road_layout.ego_lane import LineIndex
from core.road_layout.ego_route import build_route
from core.road_layout.game_map import LaneWindow, pack_lane
from core.road_layout.lanes import LaneLine

from .harness import make_vehicle

_DT = 1.0 / 30.0
_STRAIGHT = 150.0

logging.getLogger("ui.popup.popup_window").setLevel(logging.CRITICAL)


class _Lane:
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def point(self, s: float) -> tuple[float, float, float, float]:
        """``(x, z, tangent_x, tangent_z)`` at arc length ``s``."""
        if s <= _STRAIGHT:
            return 0.0, -s, 0.0, -1.0
        a = (s - _STRAIGHT) / self.radius
        r = self.radius
        return r - r * math.cos(a), -_STRAIGHT - r * math.sin(a), math.sin(a), -math.cos(a)

    def kappa(self, s: float) -> float:
        return 0.0 if s <= _STRAIGHT else -1.0 / self.radius

    def window(self) -> LaneWindow:
        pts = []
        s = -100.0
        while s <= _STRAIGHT + self.radius * 1.4:
            x, z, _tx, _tz = self.point(max(s, 0.0)) if s >= 0.0 else (0.0, -s, 0.0, -1.0)
            pts.append((x, 0.0, z))
            s += 5.0
        line = LaneLine("road", "t", tuple(pts), None, 1, 0, 1, 0.0)
        lanes = (pack_lane(line),)
        return LaneWindow((line,), (), (0, 0, 0, 0), lanes, LineIndex(lanes))


def _run(lane: _Lane, vehicle, speed: float, end_s: float, with_map: bool, cal=CAL,
         straight_on: bool = False):
    """First warn and first brake times (s), or None. ``straight_on`` ignores the bend."""
    t = _make_headless(cal)
    t._read_vehicle_key = lambda: None
    t._read_ego_geometry = lambda: None
    t._read_max_brake_ms2 = lambda: 7.0
    t._read_user_braking = lambda: False
    t._read_addressing_brake = lambda: False
    window = lane.window()
    first_warn = first_brake = None
    s = 0.0
    k = 0
    while s < end_s:
        x, z, tx, tz = (0.0, -s, 0.0, -1.0) if straight_on else lane.point(s)
        yaw = math.atan2(-tx, -tz)
        steer = 0.0 if straight_on else lane.kappa(s) / ego_path_params(cal).gain_prior
        now = 100.0 + k * _DT
        snap = ([vehicle], x, 0.0, z, yaw, speed, 0.0, steer, False, None, False, False,
                now, frozenset(), now, frozenset())
        t._read_radar_snapshot = lambda snap=snap: snap
        t._now = lambda now=now: now
        if with_map:
            route = build_route(window.lanes, window.index, 0, x, z, tx, tz)
            map_snap = SimpleNamespace(route=route, window=window, target_routes={})
            t._read_map_snapshot = lambda m=map_snap: m
        t.loop()
        tt = k * _DT
        if first_warn is None and t.data.AEB_warn:
            first_warn = tt
        if first_brake is None and t.data.AEB_brake:
            first_brake = tt
        s += speed * _DT
        k += 1
    return first_warn, first_brake


def _parked(lane: _Lane, s: float, offset: float, vid: int = 7):
    """A stopped car ``offset`` m right of the lane at arc length ``s``, along the lane."""
    x, z, tx, tz = lane.point(s)
    x += offset * (-tz)
    z += offset * tx
    return make_vehicle(vid, x, z, math.degrees(math.atan2(-tx, -tz)), 0.0, width=1.9, length=4.6)


def test_crashed_car_in_lane_behind_a_bend_brakes_sooner_with_the_map():
    for radius, speed in ((250.0, 22.0), (150.0, 17.0)):
        lane = _Lane(radius)
        for ahead in (35.0, 55.0, 80.0):
            car = _parked(lane, _STRAIGHT + ahead, 0.0)
            end = _STRAIGHT + ahead - 5.0
            base_warn, base_brake = _run(lane, car, speed, end, with_map=False)
            map_warn, map_brake = _run(lane, car, speed, end, with_map=True)
            assert map_warn is not None and map_brake is not None
            assert base_warn is None or map_warn <= base_warn - 0.8, (radius, ahead)
            assert base_brake is None or map_brake <= base_brake - 0.5, (radius, ahead)


def test_the_map_changes_nothing_for_parked_cars_beside_a_bend():
    for radius, speed in ((250.0, 22.0), (150.0, 17.0)):
        lane = _Lane(radius)
        for offset, ahead in ((-4.0, 25.0), (-4.5, 40.0), (-5.0, 60.0), (-4.5, 80.0), (4.5, 40.0)):
            car = _parked(lane, _STRAIGHT + ahead, offset)
            end = _STRAIGHT + ahead + 5.0
            base = _run(lane, car, speed, end, with_map=False)
            assert _run(lane, car, speed, end, with_map=True) == base, (radius, offset, ahead)


def test_a_driver_not_taking_the_bend_keeps_todays_braking():
    lane = _Lane(250.0)
    # Stopped on the straight line ego keeps to, off the road on the outside of the bend.
    car = make_vehicle(7, 0.0, -(_STRAIGHT + 60.0), 0.0, 0.0, width=1.9, length=4.6)
    base = _run(lane, car, 22.0, _STRAIGHT + 55.0, with_map=False, straight_on=True)
    mapped = _run(lane, car, 22.0, _STRAIGHT + 55.0, with_map=True, straight_on=True)
    assert base[1] is not None
    assert mapped == base


def test_oncoming_traffic_in_the_bend_is_left_alone():
    lane = _Lane(250.0)
    x, z, tx, tz = lane.point(_STRAIGHT + 70.0)
    x -= 3.75 * (-tz)
    z -= 3.75 * tx
    car = make_vehicle(7, x, z, math.degrees(math.atan2(tx, tz)), 20.0, width=2.5, length=10.0)
    base = _run(lane, car, 22.0, _STRAIGHT + 40.0, with_map=False)
    assert _run(lane, car, 22.0, _STRAIGHT + 40.0, with_map=True) == base


def test_today_brakes_for_some_of_those_parked_cars():
    lane = _Lane(250.0)
    car = _parked(lane, _STRAIGHT + 40.0, -4.5)
    assert _run(lane, car, 22.0, _STRAIGHT + 45.0, with_map=False)[1] is not None


def test_a_bend_too_tight_for_the_speed_keeps_the_roadside_threat():
    lane = _Lane(60.0)
    car = _parked(lane, _STRAIGHT + 18.0, -4.0)
    base = _run(lane, car, 25.0, _STRAIGHT + 10.0, with_map=False)
    mapped = _run(lane, car, 25.0, _STRAIGHT + 10.0, with_map=True)
    assert base[1] is not None and mapped[1] is not None
    assert abs(mapped[1] - base[1]) < 0.1
