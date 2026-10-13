"""Bent predicted paths and the road-bend profile. No map files, no clips.

World X east, Z south. Yaw 0 heads north (-Z); positive curvature turns left.
"""

from __future__ import annotations

import math

from core.radar.bent_path import BentPath, bend_arc, offset_curvature
from core.radar.traffic import arc_arc_collision, build_arc
from core.road_layout.road_bend import BendParams, bend_profile, choose_route, road_curve


def _bent(kappa_s, kappa_k, speed=20.0, yaw=0.0, horizon=4.0, **kw):
    return BentPath(start_x=0.0, start_z=0.0, yaw_rad=yaw, speed=speed, curvature=kappa_k[0],
                    half_width=1.2, horizon=horizon, kappa_s=kappa_s, kappa_k=kappa_k, **kw).build()


def test_constant_profile_matches_the_plain_arc():
    for k in (0.0, 0.01, -0.02):
        arc = build_arc(0.0, 0.0, 0.3, 20.0, k, 1.2, 4.0)
        bent = _bent((0.0, 100.0), (k, k), yaw=0.3)
        for d in (0.0, 13.0, 47.0, 79.0):
            ax, az = arc.position_at_dist(d)
            bx, bz = bent.position_at_dist(d)
            assert math.hypot(ax - bx, az - bz) < 0.02
            assert abs(arc.heading_at_dist(d) - bent.heading_at_dist(d)) < 1e-3


def test_bent_heading_keeps_the_arc_convention():
    """A bend must not change the body-heading convention AEB is tuned on (README §8)."""
    arc = build_arc(0.0, 0.0, 0.0, 10.0, 0.02, 1.0, 3.0)
    bent = _bent((0.0, 30.0), (0.02, 0.02), speed=10.0, horizon=3.0)
    for d in (5.0, 20.0):
        assert abs(arc.heading_at_dist(d) - bent.heading_at_dist(d)) < 1e-3


def test_flat_profile_returns_the_arc_itself():
    arc = build_arc(0.0, 0.0, 0.0, 20.0, 0.004, 1.2, 4.0)
    assert bend_arc(arc, (0.0, 50.0), (0.004, 0.004)) is arc
    assert bend_arc(arc, (), ()) is arc
    assert isinstance(bend_arc(arc, (0.0, 50.0), (0.004, 0.004), keep_flat=True), BentPath)


def test_project_reads_offset_right_positive_and_runs_past_the_end():
    bent = _bent((0.0, 20.0, 80.0), (0.0, 0.0, 0.01))
    s, side = bent.project(1.5, -10.0)
    assert abs(s - 10.0) < 0.05 and abs(side - 1.5) < 0.05
    s_far, _ = bent.project(0.0, -500.0)
    assert s_far > bent.arc_length


def test_offset_keeps_the_bend():
    bent = _bent((0.0, 20.0, 80.0), (0.0, 0.0, 0.01))
    left = offset_curvature(bent, 0.002)
    assert isinstance(left, BentPath)
    assert abs(left.kappa_at(50.0) - (bent.kappa_at(50.0) + 0.002)) < 1e-9


def test_bent_paths_collide_where_the_bend_goes():
    lead_x, lead_z = _bent((0.0, 10.0, 60.0), (0.0, 0.0, -0.01)).position_at_dist(60.0)
    ego = _bent((0.0, 10.0, 60.0), (0.0, 0.0, -0.01), fwd_len=3.0, back_len=3.0)
    parked = build_arc(lead_x, lead_z, 0.0, 0.0, 0.0, 1.0, 4.0, fwd_len=2.0, back_len=2.0)
    straight_ego = build_arc(0.0, 0.0, 0.0, 20.0, 0.0, 1.2, 4.0, fwd_len=3.0, back_len=3.0)
    assert arc_arc_collision(ego, parked, 0.3, 24) is not None
    assert arc_arc_collision(straight_ego, parked, 0.3, 24) is None


def _circle_route(radius, sign=-1.0, straight=40.0, sweep=1.2):
    """Heading north from the origin, then a turn (sign -1 right, +1 left)."""
    pts = [(0.0, -float(d)) for d in range(0, int(straight) + 1, 5)]
    for k in range(1, 60):
        a = sweep * k / 60.0
        pts.append((-sign * radius * (1.0 - math.cos(a)), -straight - radius * math.sin(a)))
    return pts


def test_road_curve_signs_and_magnitude():
    right = road_curve(_circle_route(200.0, -1.0))
    left = road_curve(_circle_route(200.0, +1.0))
    assert abs(right.kappa_at(120.0) + 1 / 200.0) < 4e-4
    assert abs(left.kappa_at(120.0) - 1 / 200.0) < 4e-4
    assert abs(right.kappa_at(10.0)) < 1e-4


def test_least_force_route_follows_what_the_vehicle_is_doing():
    straight = [(0.0, -float(d)) for d in range(0, 201, 5)]
    turn = _circle_route(60.0, -1.0, straight=20.0, sweep=1.5)
    assert choose_route([turn, straight], 0.0, 0.0, 0.0, -1.0, 0.0, 15.0) == 1
    assert choose_route([straight, turn], 0.0, 0.0, 0.0, -1.0, -1 / 60.0, 15.0) == 1


_POSE = (0.0, 0.0, 0.0, -1.0)


def test_a_following_vehicle_leads_into_the_bend_and_ends_on_the_lane():
    curve = road_curve(_circle_route(300.0, -1.0, straight=20.0))
    ss, ks = bend_profile(curve, _POSE, 0.0, 20.0, 140.0, params=BendParams())
    assert abs(ks[0]) < 1e-9
    bent = _bent(ss, ks, horizon=8.0)
    want = (300.0 * (1.0 - math.cos(100.0 / 300.0)), -20.0 - 300.0 * math.sin(100.0 / 300.0))
    got = bent.position_at_dist(120.0)
    assert math.hypot(got[0] - want[0], got[1] - want[1]) < 1.5


def test_turning_in_early_still_counts_as_following():
    curve = road_curve(_circle_route(300.0, -1.0, straight=20.0))
    ss, _ks = bend_profile(curve, _POSE, -0.6 / 300.0, 20.0, 140.0, params=BendParams())
    assert ss


def test_a_vehicle_not_following_its_lane_is_not_bent():
    curve = road_curve(_circle_route(300.0, -1.0, straight=20.0))
    # Steering against the bend, off the lane, or turned across it.
    assert bend_profile(curve, _POSE, 0.01, 20.0, 140.0) == ((), ())
    assert bend_profile(curve, (4.0, 0.0, 0.0, -1.0), 0.0, 20.0, 140.0) == ((), ())
    turned = (0.0, 0.0, math.sin(math.radians(30.0)), -math.cos(math.radians(30.0)))
    assert bend_profile(curve, turned, 0.0, 20.0, 140.0) == ((), ())


def test_the_road_just_behind_keeps_a_straight_driver_following_at_the_bend_start():
    # Two metres before the bend the smoothed road curvature already leans into it.
    curve = road_curve(_circle_route(150.0, -1.0, straight=40.0))
    pose = (0.0, -38.0, 0.0, -1.0)
    assert bend_profile(curve, pose, 0.0, 15.0, 80.0, params=BendParams(follow_tol_frac=0.0)) == ((), ())
    ss, _ks = bend_profile(curve, pose, 0.0, 15.0, 80.0, params=BendParams(follow_tol_frac=0.0),
                           recent_kappa=(0.0,))
    assert ss


def test_a_corner_too_tight_for_the_speed_is_ignored_from_there():
    curve = road_curve(_circle_route(60.0, -1.0, straight=40.0))
    ss, ks = bend_profile(curve, _POSE, 0.0, 25.0, 120.0, params=BendParams(max_lat_ms2=4.0))
    assert ks[-1] == 0.0
    assert ss[-1] < 60.0
    slow_ss, slow_ks = bend_profile(curve, _POSE, 0.0, 12.0, 60.0, params=BendParams(max_lat_ms2=4.0))
    assert min(slow_ks) < -0.005


def test_the_default_limit_follows_corners_people_take_fast():
    # 90 km/h through R 60 is 10 m/s2 of map curvature: ETS2 drivers take it (README road-bent).
    curve = road_curve(_circle_route(60.0, -1.0, straight=40.0))
    ss, ks = bend_profile(curve, _POSE, 0.0, 25.0, 120.0)
    assert min(ks) < -1 / 60.0 + 0.003


def test_where_two_lanes_become_one_the_road_path_joins_the_lane():
    # Ego 2 m left of a straight lane that it merges into 60 m ahead.
    curve = road_curve([(0.0, -float(d)) for d in range(0, 301, 5)])
    pose = (-2.0, 0.0, 0.0, -1.0)
    held_ss, held_ks = bend_profile(curve, pose, 0.0, 15.0, 120.0)
    ss, ks = bend_profile(curve, pose, 0.0, 15.0, 120.0, merge_s=60.0)
    # The test paths start at the origin, so the lane sits 2 m to their right (+x).
    held = _bent(held_ss, held_ks, speed=15.0, horizon=8.0).position_at_dist(100.0)
    joined = _bent(ss, ks, speed=15.0, horizon=8.0).position_at_dist(100.0)
    assert abs(held[0]) < 0.3
    assert abs(joined[0] - 2.0) < 0.3
    assert abs(ks[0]) < 1e-9
