"""Lane choice weighs distance first. The highlight runs forward up to 200 m."""

from __future__ import annotations

import math

from core.road_layout.ego_lane import (
    EgoLaneTrack,
    choose_lane,
    extend_lane,
    path_curvature,
    travel_direction,
)


def test_sitting_on_a_line_beats_a_sharper_heading_match():
    straight = _north(0.0)
    sharp = _ray(3.2, 50.0, math.radians(35.0))
    lines = [sharp, straight]
    choice = choose_lane(lines, 3.2, 50.0, 0.0, 1.0)
    assert choice is not None
    assert choice.index == 0


def test_a_few_metres_of_gap_beats_a_matching_heading():
    straight = _north(0.0)
    branch = _ray(0.0, 40.0, math.radians(30.0))
    heading = (math.sin(math.radians(30.0)), math.cos(math.radians(30.0)))
    lines = [straight, branch]
    choice = choose_lane(lines, 0.4, 48.0, heading[0], heading[1])
    assert choice is not None
    assert choice.index == 0
    assert _closest_index(lines, 0.4, 48.0) == 0


def test_distance_beats_a_modest_angle():
    aligned = _north(0.0)
    angled = _ray(3.4, 50.0, math.radians(10.0))
    lines = [angled, aligned]
    choice = choose_lane(lines, 3.5, 50.0, 0.0, 1.0)
    assert choice is not None
    assert choice.index == 0
    assert _closest_index(lines, 3.5, 50.0) == 0


def test_a_sharp_angle_still_loses_when_both_lines_are_close():
    aligned = _north(0.0)
    sharp = _ray(0.9, 50.0, math.radians(40.0))
    lines = [sharp, aligned]
    choice = choose_lane(lines, 0.5, 50.0, 0.0, 1.0)
    assert choice is not None
    assert choice.index == 1
    assert _closest_index(lines, 0.5, 50.0) == 0


def test_parallel_lanes_follow_the_closer_center():
    left = _north(0.0)
    right = _north(4.5)
    choice = choose_lane([left, right], 3.6, 40.0, 0.0, 1.0)
    assert choice is not None
    assert choice.index == 1


def test_a_shared_tangent_split_follows_the_line_that_stays_with_the_heading():
    trunk = [(0.0, float(z)) for z in range(0, 21, 4)]
    branch = list(trunk)
    ang = math.radians(34.0)
    for step in range(4, 49, 4):
        branch.append((math.sin(ang) * step, 20.0 + math.cos(ang) * step))
    straight = trunk + [(0.0, float(z)) for z in range(24, 81, 4)]
    # Branch is listed first, so a pure distance tie would take it.
    choice = choose_lane([branch, straight], 0.0, 12.0, 0.0, 1.0)
    assert choice is not None
    assert choice.index == 1
    assert choice.angle_rad < math.radians(5.0)


def test_a_path_that_is_turning_stays_with_the_bending_line():
    kappa = -0.0327
    curve = _arc(kappa, 48.0)
    straight = _north(0.0)
    choice = choose_lane([straight, curve], 0.0, 0.0, 0.0, 1.0, curvature=kappa)
    assert choice is not None
    assert choice.index == 1


def test_nothing_near_the_truck_selects_no_lane():
    assert choose_lane([_north(0.0)], 30.0, 20.0, 0.0, 1.0) is None


def test_direction_lines_the_recent_path_up_with_the_heading():
    history = [(0.0, float(z)) for z in range(0, 9, 2)]
    dx, dz = travel_direction(history, 0.0, 1.0)
    assert abs(dx) < 1e-6
    assert dz > 0.9

    yaw = math.radians(20.0)
    dx, dz = travel_direction(history, math.sin(yaw), math.cos(yaw))
    lined = math.degrees(math.atan2(dx, dz))
    assert 8.0 < lined < 12.0

    dx, dz = travel_direction(history, 0.0, -1.0)
    assert abs(dx) < 1e-6
    assert dz > 0.9


def test_curvature_follows_the_recent_bend():
    kappa = -0.03
    pts = _arc(kappa, 12.0, step=2.0)
    got = path_curvature(pts)
    assert abs(got - kappa) < 0.01


def test_a_position_jump_drops_the_old_path():
    track = EgoLaneTrack()
    for z in (0.0, 2.0, 4.0, 6.0):
        track.note(0.0, z)
    track.note(0.0, 500.0)
    dx, dz = travel_direction(track.points(), 1.0, 0.0)
    assert dx > 0.9
    assert abs(dz) < 0.1


def test_track_picks_the_line_under_the_truck():
    track = EgoLaneTrack()
    for z in (0.0, 2.0, 4.0, 6.0, 8.0):
        track.note(2.5, z)
    straight = _north(0.0)
    sharp = _ray(2.5, 8.0, math.radians(32.0))
    choice = track.choose([sharp, straight], 2.5, 8.0, 0.0, 1.0)
    assert choice is not None
    assert choice.index == 0


def test_highlight_runs_through_the_next_junction_and_stops_at_the_one_after():
    road = [(0.0, float(z)) for z in range(0, 41, 4)]
    junction = [(0.0, float(z)) for z in range(40, 81, 4)]
    after = [(0.0, float(z)) for z in range(80, 121, 4)]
    later = [(0.0, float(z)) for z in range(120, 161, 4)]
    lines = [
        _piece(road),
        _piece(_turn(40.0, math.radians(50.0)), 1),
        _piece(junction, 1),
        _piece(after),
        _piece(later, 2),
    ]
    assert _ids(extend_lane(lines, 0, 0.0, 10.0, 0.0, 1.0)) == (0, 2, 3, 4)


def test_highlight_from_inside_a_junction_stops_at_the_next_one():
    junction = [(0.0, float(z)) for z in range(40, 81, 4)]
    after = [(0.0, float(z)) for z in range(80, 121, 4)]
    later = [(0.0, float(z)) for z in range(120, 161, 4)]
    lines = [_piece(junction, 1), _piece(after), _piece(later, 2)]
    assert _ids(extend_lane(lines, 0, 0.0, 50.0, 0.0, 1.0)) == (0, 1, 2)


def test_curves_of_one_junction_count_as_one_prefab():
    first = [(0.0, float(z)) for z in range(0, 21, 4)]
    second = [(0.0, float(z)) for z in range(20, 41, 4)]
    road = [(0.0, float(z)) for z in range(40, 81, 4)]
    later = [(0.0, float(z)) for z in range(80, 121, 4)]
    lines = [
        _piece(first, 1),
        _piece(second, 1),
        _piece(road),
        _piece(later, 2),
    ]
    assert _ids(extend_lane(lines, 0, 0.0, 4.0, 0.0, 1.0)) == (0, 1, 2, 3)


def test_highlight_does_not_walk_backward():
    behind = [(0.0, float(z)) for z in range(-40, 1, 4)]
    road = [(0.0, float(z)) for z in range(0, 41, 4)]
    lines = [_piece(behind), _piece(road)]
    assert _ids(extend_lane(lines, 1, 0.0, 10.0, 0.0, 1.0)) == (1,)


def test_a_lane_that_ends_beside_another_continues_on_it():
    dying = [(0.0, float(z)) for z in range(0, 41, 4)]
    dying += [(1.2, 44.0), (2.4, 48.0)]
    through = [(4.5, float(z)) for z in range(0, 101, 4)]
    after = [(4.5, float(z)) for z in range(100, 141, 4)]
    lines = [_piece(dying), _piece(through), _piece(after)]
    spans = extend_lane(lines, 0, 0.0, 10.0, 0.0, 1.0)
    assert _ids(spans) == (0, 1, 2)
    assert spans[1].pts[0][1] > 40.0


def test_a_merge_junction_does_not_count_as_the_stop():
    road0 = [(0.0, float(z)) for z in range(0, 41, 4)]
    junction = [(0.0, float(z)) for z in range(40, 81, 4)]
    road1 = [(0.0, float(z)) for z in range(80, 121, 4)]
    merge_a = [(0.0, float(z)) for z in range(120, 161, 4)]
    merge_b = [(5.0, 120.0)] + [(0.0, float(z)) for z in range(124, 161, 4)]
    road2 = [(0.0, float(z)) for z in range(160, 201, 4)]
    stop = [(0.0, float(z)) for z in range(200, 241, 4)]
    lines = [
        _piece(road0),
        _piece(junction, 1),
        _piece(road1),
        _piece(merge_a, 2),
        _piece(merge_b, 2),
        _piece(road2),
        _piece(stop, 3),
    ]
    assert _ids(extend_lane(lines, 0, 0.0, 10.0, 0.0, 1.0)) == (0, 1, 2, 3, 5, 6)
    assert spans_end_z(extend_lane(lines, 0, 0.0, 10.0, 0.0, 1.0)) < 220.0


def test_highlight_stops_before_the_sixth_prefab():
    lines = []
    for i in range(6):
        z0 = i * 20.0
        lines.append(_piece([(0.0, z0), (0.0, z0 + 20.0)], i + 1))
    assert _ids(extend_lane(lines, 0, 0.0, 2.0, 0.0, 1.0)) == (0, 1, 2, 3, 4)


def test_highlight_stops_at_200_m():
    road = [(0.0, float(z)) for z in range(0, 401, 4)]
    spans = extend_lane([_piece(road)], 0, 0.0, 0.0, 0.0, 1.0)
    end = spans[-1].pts[-1][1]
    assert 190.0 < end < 210.0


def _ids(chain) -> tuple[int, ...]:
    return tuple(span.index for span in chain)


def spans_end_z(spans) -> float:
    return spans[-1].pts[-1][1]


def _piece(pts, prefab_id=None):
    return type("Piece", (), {"pts": tuple(pts), "prefab_id": prefab_id})()


def _turn(z0: float, angle: float):
    pts = []
    step = 0.0
    while step <= 40.0:
        pts.append((math.sin(angle) * step, z0 + math.cos(angle) * step))
        step += 4.0
    return pts


def _north(x: float) -> list[tuple[float, float]]:
    return [(x, float(z)) for z in range(0, 81, 4)]


def _ray(x: float, z: float, angle: float) -> list[tuple[float, float]]:
    pts = []
    step = -20.0
    while step <= 50.0:
        pts.append((x + math.sin(angle) * step, z + math.cos(angle) * step))
        step += 4.0
    return pts


def _arc(kappa: float, length: float, step: float = 4.0) -> list[tuple[float, float]]:
    pts = []
    s = 0.0
    while s <= length + 1e-6:
        pts.append(((math.cos(kappa * s) - 1.0) / kappa, math.sin(kappa * s) / kappa))
        s += step
    return pts


def _closest_index(lines: list[list[tuple[float, float]]], x: float, z: float) -> int:
    best_i = 0
    best = None
    for index, pts in enumerate(lines):
        for i in range(len(pts) - 1):
            dist = _segment(x, z, pts[i], pts[i + 1])
            if best is None or dist < best:
                best = dist
                best_i = index
    return best_i


def _segment(x: float, z: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = b[0] - a[0]
    dz = b[1] - a[1]
    length2 = dx * dx + dz * dz
    if length2 < 1e-12:
        return math.hypot(x - a[0], z - a[1])
    t = max(0.0, min(1.0, ((x - a[0]) * dx + (z - a[1]) * dz) / length2))
    return math.hypot(x - (a[0] + t * dx), z - (a[1] + t * dz))
