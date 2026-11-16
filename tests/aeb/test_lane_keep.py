"""Lane keep: a mover following its own map lane, clear of ego's track, is not braked for.

Ego drives north (-Z) on a straight lane. A slower truck joins the next lane over an
S-shaped slip road: for a second its yaw rate points across ego's lane, though its
lane never comes closer than a lane width. Without a map every run is as today.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.aeb.calibration import DEFAULT, ego_path_params
from core.aeb.clip_eval import _make_headless
from core.aeb.lane_track import _excursion, keeps_own_lane, lane_keep_free
from core.aeb.road_paths import RoadPaths
from core.road_layout.ego_lane import LineIndex
from core.road_layout.ego_route import build_route
from core.road_layout.game_map import LaneWindow, pack_lane
from core.road_layout.lanes import LaneLine

from .harness import make_vehicle

_DT = 1.0 / 30.0
_LANE_M = 4.5
_SHIFT_M = 9.0
_RADIUS_M = 60.0
_S_START = 60.0
CAL = DEFAULT
OFF = replace(DEFAULT, lane_keep_enabled=False)

logging.getLogger("ui.popup.popup_window").setLevel(logging.CRITICAL)


def _arc() -> tuple[float, float]:
    theta = math.acos(1.0 - _SHIFT_M / (2.0 * _RADIUS_M))
    return theta, _RADIUS_M * theta


def _slip_kappa(s: float) -> float:
    """Curvature of the slip road at ``s``, positive to the left."""
    _theta, arc = _arc()
    if _S_START < s <= _S_START + arc:
        return 1.0 / _RADIUS_M
    if _S_START + arc < s <= _S_START + 2.0 * arc:
        return -1.0 / _RADIUS_M
    return 0.0


def _slip(s: float) -> tuple[float, float, float]:
    """``(x, z, heading left of north in rad)`` on the slip road at arc length ``s``."""
    theta, arc = _arc()
    x0 = _LANE_M + _SHIFT_M
    if s <= _S_START:
        return x0, -s, 0.0
    if s <= _S_START + arc:
        a = (s - _S_START) / _RADIUS_M
        return x0 - _RADIUS_M * (1.0 - math.cos(a)), -_S_START - _RADIUS_M * math.sin(a), a
    xm = x0 - _RADIUS_M * (1.0 - math.cos(theta))
    zm = -_S_START - _RADIUS_M * math.sin(theta)
    if s <= _S_START + 2.0 * arc:
        a = theta - (s - _S_START - arc) / _RADIUS_M
        return (xm - _RADIUS_M * (math.cos(a) - math.cos(theta)),
                zm - _RADIUS_M * (math.sin(theta) - math.sin(a)), a)
    x2, z2, _a = _slip(_S_START + 2.0 * arc)
    return x2, z2 - (s - _S_START - 2.0 * arc), 0.0


def _window() -> LaneWindow:
    ego = tuple((0.0, 0.0, -s) for s in range(-100, 500, 5))
    slip = tuple((p[0], 0.0, p[1]) for p in (_slip(float(s)) for s in range(-100, 500, 3)))
    lines = (LaneLine("road", "ego", ego, None, 1, 0, 2, 0.0),
             LaneLine("road", "slip", slip, None, 2, 0, 1, 0.0))
    lanes = tuple(pack_lane(line) for line in lines)
    return LaneWindow(lines, (), (0, 0, 0, 0), lanes, LineIndex(lanes))


def _run(with_map: bool, cal=CAL, swerve_at: float | None = None, ego_drift: float = 0.0):
    """First brake time (s) or None. ``swerve_at``: the truck leaves its lane for ego's there."""
    t = _make_headless(cal)
    t._read_vehicle_key = lambda: None
    t._read_ego_geometry = lambda: None
    t._read_max_brake_ms2 = lambda: 7.0
    t._read_user_braking = lambda: False
    t._read_addressing_brake = lambda: False
    window = _window()
    first_brake = None
    cut = 0.0
    for k in range(int(5.0 / _DT)):
        now = 100.0 + k * _DT
        tt = k * _DT
        s_ego = 27.0 * tt
        s_truck = 50.0 + 15.0 * tt
        x, z, a = _slip(s_truck)
        if swerve_at is not None and s_truck >= swerve_at:
            # Heading 12 degrees further left: across the line and into ego's lane.
            cut += 15.0 * _DT * math.sin(math.radians(12.0))
            x -= cut
            a += math.radians(12.0)
        if s_truck - s_ego < 8.0:
            break
        truck = make_vehicle(7, x, z, math.degrees(a), 15.0, _slip_kappa(s_truck), 2.5, 7.0)
        ego_x = ego_drift * tt
        yaw = math.atan2(-ego_drift, 27.0)
        steer = 0.0 / ego_path_params(cal).gain_prior
        snap = ([truck], ego_x, 0.0, -s_ego, yaw, 27.0, 0.0, steer, False, None, False, False,
                now, frozenset(), now, frozenset())
        t._read_radar_snapshot = lambda snap=snap: snap
        t._now = lambda now=now: now
        if with_map:
            route = build_route(window.lanes, window.index, 0, ego_x, -s_ego, 0.0, -1.0)
            fx, fz = -math.sin(a), -math.cos(a)
            own = build_route(window.lanes, window.index, 1, x, z, fx, fz)
            map_snap = SimpleNamespace(route=route, window=window, target_routes={7: own})
            t._read_map_snapshot = lambda m=map_snap: m
        t.loop()
        if first_brake is None and t.data.AEB_brake:
            first_brake = tt
    return first_brake


def _free(s_truck: float, speed: float = 15.0, ego_x: float = 0.0, ego_vx: float = 0.0,
          kappa: float | None = None, offset: float = 0.0, ego_kappa: float = 0.0):
    """The lane-keep margin for the truck at ``s_truck`` on the slip road, ego at the origin."""
    window = _window()
    x, z, a = _slip(s_truck)
    x -= offset
    fx, fz = -math.sin(a), -math.cos(a)
    snap = SimpleNamespace(
        route=build_route(window.lanes, window.index, 0, ego_x, 0.0, 0.0, -1.0), window=window,
        target_routes={7: build_route(window.lanes, window.index, 1, x, z, fx, fz)})
    paths = RoadPaths()
    paths.frame(snap, CAL, 0.0)
    paths.bind_ego_lane(ego_x, 0.0, CAL, ego_vx, -27.0, ego_kappa)
    truck = make_vehicle(7, x, z, math.degrees(a), speed, width=2.5, length=7.0)
    own = _slip_kappa(s_truck) if kappa is None else kappa
    return lane_keep_free(paths, truck, a, 1.0, own, 27.0, 2.5, CAL)


def test_the_slip_road_never_comes_closer_than_a_lane():
    assert min(_slip(float(s))[0] for s in range(0, 400)) > _LANE_M - 0.05


def test_a_mover_on_the_next_lane_has_a_lane_less_two_half_widths_to_spare():
    free = _free(150.0)
    assert free is not None and abs(free - (_LANE_M - 2.5)) < 0.3
    assert free >= CAL.lane_keep_min_free_m


def test_following_the_bend_of_its_own_lane_is_no_swerve():
    # Mid way through the first arc the truck turns toward ego's lane exactly as its lane does.
    assert _free(_S_START + 8.0) >= CAL.lane_keep_min_free_m
    # The same turn on the straight part of its lane is a swerve toward ego.
    swerve = _free(150.0, kappa=1.0 / _RADIUS_M)
    assert swerve is not None and swerve < CAL.lane_keep_min_free_m


def test_a_mover_off_its_lane_toward_ego_gives_the_room_away():
    assert _free(150.0, offset=0.5) < _free(150.0) - 0.4
    # A metre over, its body no longer leaves the room the verdict needs.
    assert _free(150.0, offset=1.0) is None


def test_a_crawling_vehicle_is_never_a_lane_keeper():
    assert _free(150.0, speed=CAL.lane_keep_min_speed_ms - 1.0) is None


def test_ego_off_the_middle_of_its_lane_gets_no_verdict():
    assert _free(150.0, ego_x=CAL.lane_keep_max_ego_offset_m + 0.2) is None
    assert _free(150.0, ego_x=-CAL.lane_keep_max_ego_offset_m - 0.2) is None


def test_ego_moving_or_swerving_toward_it_uses_the_room_up():
    assert _free(150.0, ego_vx=1.0) < _free(150.0) - 1.0
    assert _free(150.0, ego_vx=-1.0) == _free(150.0)
    # A small steer toward it without any drift yet is a transient; a hard one is a swerve.
    gentle = -2.0 / 27.0 ** 2
    hard = -(CAL.lane_keep_swerve_ms2 + 1.0) / 27.0 ** 2
    assert _free(150.0, ego_kappa=gentle) == _free(150.0)
    assert _free(150.0, ego_kappa=hard) < CAL.lane_keep_min_free_m


def test_without_a_map_the_joining_truck_is_braked_for_as_today():
    assert _run(with_map=False) is not None
    assert _run(with_map=False) == _run(with_map=False, cal=OFF)


def test_a_truck_joining_the_next_lane_on_its_own_lane_is_not_braked_for():
    assert _run(with_map=True) is None


def test_switched_off_the_map_changes_nothing():
    assert _run(with_map=True, cal=OFF) == _run(with_map=False, cal=OFF)


def test_a_truck_that_leaves_its_lane_for_egos_is_braked_for_as_it_leaves():
    # It turns out of its lane at 3.0 s; today's brake came earlier, on the slip road's bend.
    assert _run(with_map=False, swerve_at=95.0) is not None
    mapped = _run(with_map=True, swerve_at=95.0)
    assert mapped is not None and 3.0 <= mapped <= 3.3


def test_ego_drifting_toward_the_joining_truck_gets_no_benefit_of_the_doubt():
    assert _run(with_map=True, ego_drift=1.5) == _run(with_map=False, ego_drift=1.5)


def test_the_excursion_is_the_furthest_point_of_the_move_across():
    assert _excursion(-1.0, 0.0, 1.5) == 0.0
    assert _excursion(2.0, 0.0, 1.5) == 3.0
    # 2 m/s across, turning back at 4 m/s2: stops after half a second, half a metre out.
    assert abs(_excursion(2.0, -4.0, 1.5) - 0.5) < 1e-9
    # Standing in its lane but turning in at 2 m/s2.
    assert abs(_excursion(0.0, 2.0, 1.0) - 1.0) < 1e-9


def _returning(rate, kappa=-0.027, gap=4.5, with_map=True, cal=CAL):
    """Lane keep for a truck 0.6 m over toward ego, heading 4 degrees in, turning at ``kappa``."""
    paths = SimpleNamespace(
        keep_frame=(0.0, 0.0, 0.0, 0.0) if with_map else None,
        lane_offset=lambda *args: (34.0, 4.2, 0.0, -1.0),
        target_lane=lambda *args: (0.6, 0.07, -0.017, 0.0, 1.0),
        route_gap=lambda *args: gap,
        lane_bend=lambda *args: 0.0,
    )
    truck = make_vehicle(7, 4.2, -34.0, 4.0, 10.5, width=2.5, length=6.0)
    return keeps_own_lane(paths, truck, math.radians(4.0), 1.0, kappa, 27.8, 2.53, cal, rate)


@pytest.mark.parametrize("rate", [None, float("nan"), float("inf"), -0.01, -1.0])
def test_turning_back_without_a_measured_opening_miss_earns_nothing(rate):
    assert not _returning(rate)


@pytest.mark.parametrize("rate", [0.0, 0.1, 2.0])
def test_turning_back_with_the_miss_opening_counts_the_turn_away(rate):
    assert _returning(rate)
    assert not _returning(rate, cal=replace(CAL, lane_return_enabled=False))


def test_an_opening_miss_alone_never_earns_the_return():
    assert not _returning(0.1, kappa=0.027)
    assert not _returning(0.1, kappa=0.0)
    assert not _returning(0.1, with_map=False)
    assert not _returning(0.1, gap=3.9)
