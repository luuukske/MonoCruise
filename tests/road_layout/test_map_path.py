"""Fused ego path and the lane router, on synthetic lanes. No map files.

World X east, Z south. Ego heads east, so ego-frame right is world +Z.
"""

from __future__ import annotations

import math

from core.road_layout.ego_lane import LineIndex
from core.road_layout.ego_route import EgoRouter, LaneRoute, build_route
from core.road_layout.game_map import LaneWindow, pack_lane
from core.road_layout.lanes import LaneLine
from core.road_layout.map_path import PathParams, build_map_path


def _line(pts, kind="road", uid=1, lane_index=0, prefab_id=None):
    return LaneLine(kind, "test", tuple((x, 0.0, z) for x, z in pts), prefab_id, uid, lane_index, 2, 0.0)


def _window(lines):
    lanes = tuple(pack_lane(line) for line in lines)
    return LaneWindow(tuple(lines), (), (0, 0, 0, 0), lanes, LineIndex(lanes))


def _straight(z, x0=-100.0, x1=400.0, uid=1, lane_index=0):
    return _line([(float(x), z) for x in range(int(x0), int(x1) + 1, 10)], uid=uid, lane_index=lane_index)


def _bend_right(radius=300.0, start=60.0):
    """Straight east to ``start``, then a right turn (towards +Z) of 90 degrees."""
    pts = [(float(x), 0.0) for x in range(-100, int(start) + 1, 10)]
    for k in range(1, 61):
        a = (math.pi / 2.0) * k / 60.0
        pts.append((start + radius * math.sin(a), radius * (1.0 - math.cos(a))))
    return _line(pts)


def _route(lines, x=0.0, z=0.0):
    window = _window(lines)
    return build_route(window.lanes, window.index, 0, x, z, 1.0, 0.0)


def test_route_starts_at_the_foot_and_reads_the_offset():
    route = _route([_straight(0.0)], z=1.0)
    assert route is not None
    assert abs(route.pts[0][0]) < 1e-6 and abs(route.pts[0][1]) < 1e-6
    assert abs(route.offset_m - 1.0) < 1e-6
    assert route.reach_m > 200.0


def test_offset_is_held_along_a_straight_lane_not_steered_back():
    route = _route([_straight(0.0)], z=1.0)
    path = build_map_path(route, 0.0, 1.0, 1.0, 0.0, 0.0, 25.0)
    assert path is not None
    for s in (20.0, 80.0, 150.0):
        assert abs(path.point_at(s)[1]) < 1e-6


def test_the_path_leaves_ego_along_the_steering_arc():
    route = _route([_straight(0.0)])
    kappa = 0.004
    path = build_map_path(route, 0.0, 0.0, 1.0, 0.0, kappa, 25.0)
    x, y = path.point_at(5.0)
    arc_y = -(1.0 - math.cos(kappa * 5.0)) / kappa
    assert abs(y - arc_y) < 0.02
    assert abs(path.heading[0]) < 1e-9
    assert abs(path.point_at(path.blend_m + 10.0)[1]) < 1e-6


def test_far_field_follows_the_bend_while_ego_still_drives_straight():
    route = _route([_bend_right()])
    path = build_map_path(route, 0.0, 0.0, 1.0, 0.0, 0.0, 25.0)
    s = 60.0 + 300.0 * (math.pi / 6.0)
    x, y = path.point_at(s)
    want_x = 60.0 + 300.0 * math.sin(math.pi / 6.0)
    want_y = 300.0 * (1.0 - math.cos(math.pi / 6.0))
    assert math.hypot(x - want_x, y - want_y) < 0.6
    assert y > 35.0


def test_trust_falls_with_offset_and_heading_error():
    near = _route([_straight(0.0)], z=0.5)
    far = _route([_straight(0.0)], z=2.2)
    assert build_map_path(near, 0.0, 0.5, 1.0, 0.0, 0.0, 25.0).trust_at(50.0) > 0.99
    assert build_map_path(far, 0.0, 2.2, 1.0, 0.0, 0.0, 25.0).trust_at(50.0) < 0.2
    yawed = LaneRoute(near.pts, 0.0, math.radians(20.0), 0, "road", 2, (), near.reach_m, None)
    assert build_map_path(yawed, 0.0, 0.0, 1.0, 0.0, 0.0, 25.0).trust_at(50.0) < 0.01


def test_trust_ends_where_a_branch_parts():
    base = _route([_straight(0.0)])
    div = tuple((s, max(0.0, (s - 100.0) * 0.05)) for s in range(0, 250, 5))
    route = LaneRoute(base.pts, 0.0, 0.0, 0, "road", 2, div, base.reach_m, 100.0)
    path = build_map_path(route, 0.0, 0.0, 1.0, 0.0, 0.0, 25.0)
    assert path.trust_at(90.0) > 0.99
    assert path.trust_at(140.0) < 0.01


def test_coords_measure_from_the_path_right_positive():
    route = _route([_straight(0.0)])
    path = build_map_path(route, 0.0, 0.0, 1.0, 0.0, 0.0, 25.0)
    s, d = path.coords(80.0, 3.0)
    assert abs(s - 80.0) < 1e-6 and abs(d - 3.0) < 1e-6
    s, d = path.coords(40.0, -2.0)
    assert abs(d + 2.0) < 1e-6
    assert path.coords(-5.0, 0.0) is None


def test_router_holds_its_lane_until_the_other_is_clearly_better():
    window = _window([_straight(0.0, uid=1, lane_index=0), _straight(4.5, uid=1, lane_index=1)])
    router = EgoRouter()
    route = None
    for x in range(0, 60, 2):
        route = router.step(window, float(x), 0.0, 1.0, 1.0, 0.0)
    assert route is not None and abs(route.offset_m - 1.0) < 1e-6
    route = router.step(window, 62.0, 0.0, 2.5, 1.0, 0.0)
    assert abs(route.offset_m - 2.5) < 1e-6
    route = router.step(window, 64.0, 0.0, 3.6, 1.0, 0.0)
    assert abs(route.offset_m + 0.9) < 1e-6


def test_router_rejects_a_lane_on_another_level():
    low = _straight(0.0)
    bridge = LaneLine("road", "test", tuple((float(x), 9.0, 0.3) for x in range(-100, 401, 10)), None, 2, 0, 1, 0.0)
    window = _window([bridge, low])
    route = EgoRouter().step(window, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert route is not None and window.lanes[route.lane_index].uid == 1


def test_a_junction_split_is_reported_as_divergence():
    approach = _line([(float(x), 0.0) for x in range(-100, 101, 10)], uid=1)
    straight = _line([(float(x), 0.0) for x in range(100, 301, 10)], kind="prefab", uid=9, prefab_id=9)
    exit_pts = [(100.0 + 80.0 * math.sin(a / 20.0 * math.pi / 2), 80.0 * (1 - math.cos(a / 20.0 * math.pi / 2)))
                for a in range(0, 21)]
    turn = _line(exit_pts, kind="prefab", uid=9, lane_index=1, prefab_id=9)
    route = _route([approach, straight, turn])
    assert route.branch_s is not None and 95.0 <= route.branch_s <= 130.0
    gaps = dict(route.divergence)
    assert gaps[50.0] == 0.0
    assert gaps[160.0] > 5.0
    path = build_map_path(route, 0.0, 0.0, 1.0, 0.0, 0.0, 25.0, PathParams())
    assert path.trust_at(60.0) > 0.99 and path.trust_at(170.0) < 0.01


def test_overlapping_curves_that_fan_out_end_trust_but_parallel_lanes_do_not():
    straight = _line([(float(x), 0.0) for x in range(0, 201, 10)], kind="prefab", uid=5, prefab_id=5)
    fan = _line([(float(x), max(0.0, (x - 40.0) * 0.12)) for x in range(0, 201, 10)],
                kind="prefab", uid=5, lane_index=1, prefab_id=5)
    window = _window([straight, fan])
    route = EgoRouter().step(window, 5.0, 0.0, 0.2, 1.0, 0.0)
    path = build_map_path(route, 5.0, 0.2, 1.0, 0.0, 0.0, 20.0)
    assert path.trust_at(20.0) > 0.99
    assert path.trust_at(120.0) < 0.01
    parallel = _window([_straight(0.0, uid=1, lane_index=0), _straight(4.5, uid=1, lane_index=1)])
    route = EgoRouter().step(parallel, 0.0, 0.0, 2.2, 1.0, 0.0)
    assert route is not None and route.branch_s is None


def _merging_pair():
    """Lane 0 east along z = 0; lane 1 runs 4.5 m left of it and joins it at x = 100."""
    left = [(float(x), -4.5) for x in range(-100, 61, 10)] + [(80.0, -2.5), (100.0, 0.0)]
    return [_straight(0.0), _line(left, uid=2)]


def test_two_lanes_becoming_one_mark_the_merge():
    window = _window(_merging_pair())
    route = build_route(window.lanes, window.index, 0, 0.0, -2.25, 1.0, 0.0, siblings=[1])
    assert route is not None and route.merge_s is not None
    assert 85.0 < route.merge_s < 101.0
    assert route.merge_index is not None
    assert abs(route.merge_offset_m - 2.25) < 0.1


def test_a_lane_without_a_merge_keeps_its_offset():
    window = _window([_straight(0.0), _straight(-4.5, uid=2)])
    route = build_route(window.lanes, window.index, 0, 0.0, -2.25, 1.0, 0.0, siblings=[1])
    assert route is not None and route.merge_s is None


def test_the_map_path_joins_the_merged_lane_instead_of_holding_the_offset():
    window = _window(_merging_pair())
    route = build_route(window.lanes, window.index, 0, 0.0, -2.25, 1.0, 0.0, siblings=[1])
    path = build_map_path(route, 0.0, -2.25, 1.0, 0.0, 0.0, 15.0)
    # Ego-frame y is right of ego; the merged lane sits 2.25 m right of the start.
    assert abs(path.point_at(150.0)[1] - 2.25) < 0.3
