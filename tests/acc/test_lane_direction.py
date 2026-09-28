"""Lane direction into the road model. No clips, no map files.

Ego at the origin, forward +x, right +z. That is the ego frame the fit uses:
forward x, lateral y = world z.
"""

from __future__ import annotations

from core.acc.lane_direction import LaneDirection, centreline_world, lane_direction_samples
from core.acc.road_model import fit_road_model, from_curvature
from core.acc.tracker import ACCTracker


def _straight(z: float, x0: float = -30.0, x1: float = 180.0):
    return tuple((x, z) for x in range(int(x0), int(x1) + 1, 4))


def _yawed(z0: float, slope: float):
    return tuple((float(x), z0 + slope * max(float(x), 0.0)) for x in range(-30, 181, 4))


def test_an_offset_from_a_straight_lane_is_not_steered_back():
    lane = LaneDirection(_straight(-1.6), distance_m=1.6, cost=1.6)
    samples = lane_direction_samples([lane], 0.0, 0.0, 1.0, 0.0)
    assert samples
    model = fit_road_model([], [], lane_samples=samples)
    for x in (20.0, 60.0, 100.0):
        assert abs(model.lateral_at(x)) < 0.35


def test_near_points_stay_with_the_heading_and_far_points_follow_the_lane():
    lane = LaneDirection(_yawed(0.0, 0.12), distance_m=0.2, cost=0.2)
    samples = lane_direction_samples([lane], 0.0, 0.0, 1.0, 0.0)
    near = next(y for x, y in samples if abs(x - 10.0) < 1.0)
    far = next(y for x, y in samples if abs(x - 100.0) < 8.0)
    raw_near = 0.12 * 10.0
    raw_far = 0.12 * 100.0
    assert abs(near) < abs(raw_near) * 0.45
    assert abs(far - raw_far) < 1.5


def test_between_two_lanes_the_routes_are_averaged_by_distance():
    left = LaneDirection(_straight(-2.0), distance_m=2.0, cost=2.0)
    right = LaneDirection(_yawed(2.0, 0.10), distance_m=2.0, cost=2.05)
    samples = lane_direction_samples([left, right], 0.0, 0.0, 1.0, 0.0)
    far = next(y for x, y in samples if abs(x - 100.0) < 8.0)
    assert abs(far - 5.0) < 1.2


def test_on_a_centreline_the_other_lane_is_ignored():
    held = LaneDirection(_straight(-0.2), distance_m=0.2, cost=0.2)
    other = LaneDirection(_yawed(4.0, 0.12), distance_m=4.2, cost=0.4)
    samples = lane_direction_samples([held, other], 0.0, 0.0, 1.0, 0.0)
    far = next(y for x, y in samples if abs(x - 100.0) < 8.0)
    assert abs(far) < 0.8


def test_a_trail_that_disagrees_outranks_the_lane():
    lane = LaneDirection(_yawed(0.0, 0.12), distance_m=0.2, cost=0.2)
    samples = lane_direction_samples([lane], 0.0, 0.0, 1.0, 0.0)
    trail = [(1, float(x), 0.0, 3.0) for x in range(15, 120, 8)]
    model = fit_road_model([], trail, lane_samples=samples)
    lane_only = fit_road_model([], [], lane_samples=samples)
    assert abs(model.lateral_at(80.0)) < abs(lane_only.lateral_at(80.0)) * 0.45


def test_tracker_publishes_the_lane_into_the_fit():
    lane = LaneDirection(_yawed(0.0, 0.1), distance_m=0.3, cost=0.3)
    tracker = ACCTracker()
    tracker.set_lane_directions([lane])
    road = tracker._build_road_model([], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    bare = ACCTracker()._build_road_model([], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert abs(road.lateral_at(90.0)) > abs(bare.lateral_at(90.0)) + 2.0


def test_centreline_world_follows_the_heading_axis():
    pts = centreline_world(from_curvature(0.0), 10.0, 20.0, 1.0, 0.0, reach_m=20.0, step_m=10.0)
    assert pts[0] == (10.0, 20.0)
    assert abs(pts[-1][0] - 30.0) < 1e-6
    assert abs(pts[-1][1] - 20.0) < 1e-6


def test_clearing_the_lane_restores_the_old_fit():
    tracker = ACCTracker()
    tracker.set_lane_directions([
        LaneDirection(_yawed(0.0, 0.1), distance_m=0.3, cost=0.3),
    ])
    tracker.set_lane_directions(None)
    road = tracker._build_road_model([], 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert abs(road.lateral_at(90.0)) < 0.2
