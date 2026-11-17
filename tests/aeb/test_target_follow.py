"""Target follow: a mover on its own map lane is judged on the road path's prediction of it.

Reuses lane keep's two-lane road with a slip road (``test_lane_keep``), with lane keep
switched off so only the new stage decides. Without a map it never acts.
"""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

from core.aeb.calibration import DEFAULT
from core.aeb.lane_frame import Lane
from core.aeb.road_paths import _follow_path, _lane_drift
from core.aeb.target_follow import TargetFollowFilter
from core.radar.traffic import build_arc
from core.road_layout.road_bend import lateral_offset, road_curve

from .test_lane_keep import _run

ON = replace(DEFAULT, lane_keep_enabled=False, target_follow_enabled=True)
OFF = replace(DEFAULT, lane_keep_enabled=False, target_follow_enabled=False)


def test_on_by_default_and_silent_without_a_map():
    assert DEFAULT.target_follow_enabled is True
    assert _run(with_map=False, cal=ON) == _run(with_map=False, cal=OFF)


def test_a_truck_joining_the_next_lane_on_its_own_lane_is_not_braked_for():
    assert _run(with_map=False, cal=ON) is not None
    assert _run(with_map=True, cal=ON) is None


def test_switched_off_it_changes_nothing():
    assert _run(with_map=True, cal=OFF) == _run(with_map=False, cal=OFF)


def test_a_truck_that_leaves_its_lane_for_egos_is_braked_for_as_it_leaves():
    mapped = _run(with_map=True, swerve_at=95.0, cal=ON)
    assert mapped is not None and 3.0 <= mapped <= 3.3


def _turn_lane(radius: float = 10.0):
    """20 m north, a quarter turn right of ``radius``, then 30 m east (x right, -z north)."""
    pts = [(0.0, -float(s)) for s in range(0, 21)]
    for k in range(1, 31):
        a = k / 30.0 * math.pi / 2.0
        pts.append((radius - radius * math.cos(a), -20.0 - radius * math.sin(a)))
    pts.extend((radius + float(s), -20.0 - radius) for s in range(1, 31))
    return pts


def _turn_path(err_deg: float):
    """A car 40 degrees into the turn, half its curvature measured, ``err_deg`` short of the lane."""
    curve = road_curve(_turn_lane())
    a = math.radians(40.0)
    x, z = 10.0 - 10.0 * math.cos(a), -20.0 - 10.0 * math.sin(a)
    # Yaw convention: fwd = (-sin yaw, -cos yaw); turning right lowers it.
    arc = build_arc(x, z, math.radians(-40.0 + err_deg), 9.0, -0.05, 0.9, 3.0)
    return curve, arc, _follow_path(curve, arc, arc.arc_length + 10.0, 5.0)


def test_the_follow_path_stays_on_a_tight_lane_from_a_lagging_turn():
    # The plain arc runs metres wide of a 10 m radius turn; the follow path keeps to it.
    for err in (0.0, 5.0):
        curve, arc, path = _turn_path(err)
        for d in (2.0, 6.0, 10.0, 14.0, 20.0, 26.0):
            assert abs(lateral_offset(curve, *path.position_at_dist(d))[1]) < 0.3
        assert abs(lateral_offset(curve, *arc.position_at_dist(20.0))[1]) > 2.0


def test_bodies_on_the_follow_path_turn_with_the_motion():
    _curve, arc, path = _turn_path(0.0)
    assert abs(path.heading_at_dist(0.0) - arc.yaw_rad) < 1e-6
    # The turn ends heading east: yaw -90 degrees, not mirrored back past north.
    assert abs(path.heading_at_dist(20.0) - math.radians(-90.0)) < math.radians(5.0)


def _ctx(branches, follow_s=1.0, latched=False, hit=False, head_on=False, road=None):
    ego = build_arc(0.0, 0.0, 0.0, 20.0, 0.0, 1.25, 3.0)
    far = build_arc(0.0, -30.0, 0.0, 5.0, 0.0, 0.9, 3.0) if hit else build_arc(
        10.0, -30.0, 0.0, 20.0, 0.0, 0.9, 3.0)
    arcs = [[far] for _ in range(branches)]
    return SimpleNamespace(
        v=SimpleNamespace(id=7, crash_confirmed=False, is_tmp=False), ego_arc=ego, ego_speed=2.0,
        latched_threat_ids={7} if latched else set(),
        follow_fn=(lambda: arcs) if branches else None, follow_s=follow_s,
        head_on=head_on, ego_road_arc=road, ego_hw=1.25,
        lane=Lane.OPPOSITE_OR_OUTER, all_target_arcs=arcs[0] if arcs else [],
    )


def test_the_stage_drops_only_a_held_follower_whose_every_branch_stays_clear():
    stage = TargetFollowFilter(ON)
    assert stage.apply(_ctx(2)).suppressed
    assert not stage.apply(_ctx(2, hit=True)).suppressed
    assert not stage.apply(_ctx(2, follow_s=0.1)).suppressed
    assert not stage.apply(_ctx(2, latched=True)).suppressed
    assert not stage.apply(_ctx(2, head_on=True)).suppressed
    assert not stage.apply(_ctx(0)).suppressed
    assert not TargetFollowFilter(OFF).apply(_ctx(2)).suppressed


def test_a_body_on_ego_road_path_is_kept_even_when_the_plain_arc_misses():
    # Ego's road path runs 10 m right, along the body's line; the plain arc stays clear of it.
    road = build_arc(10.0, 0.0, 0.0, 35.0, 0.0, 1.25, 3.0)
    stage = TargetFollowFilter(ON)
    assert stage.apply(_ctx(1)).suppressed
    assert not stage.apply(_ctx(1, road=road)).suppressed


def _straight_follow(heading_deg: float, speed: float = 12.0):
    """A body on a straight lane 4.5 m right of the origin, heading ``heading_deg`` toward x = 0."""
    curve = road_curve([(4.5, -float(s)) for s in range(0, 120)])
    # Yaw convention: fwd = (-sin yaw, -cos yaw); a positive yaw leans left, toward x = 0.
    arc = build_arc(4.5, -10.0, math.radians(heading_deg), speed, 0.0, 1.25, 3.0)
    return curve, _follow_path(curve, arc, arc.arc_length + 10.0, 6.0,
                               drift=_lane_drift(curve, 10.0, arc, ON))


def test_a_body_sliding_across_its_lane_keeps_sliding():
    # 5 degrees across a straight lane, 4 beyond lane keep's slack: carried on for 1.5 s.
    curve, path = _straight_follow(5.0)
    side = lateral_offset(curve, *path.position_at_dist(18.0))[1]
    expect = -(math.sin(math.radians(5.0)) - math.sin(math.radians(1.0))) * 18.0
    assert abs(side - expect) < 0.3
    # Past the 1.5 s horizon it holds the offset it reached.
    late = lateral_offset(curve, *path.position_at_dist(30.0))[1]
    assert abs(late - side) < 0.3


def test_heading_inside_the_slack_is_no_slide():
    curve, path = _straight_follow(0.5)
    assert abs(lateral_offset(curve, *path.position_at_dist(25.0))[1]) < 0.15


def test_ego_curving_into_the_next_lane_is_seen_on_the_arc_itself():
    # Ego at 25 m/s curving left at 0.016 1/m reaches 4.8 m left 24 m on; a car keeping that
    # lane is in ego's way there, whatever the straight-ahead offset says.
    ego = build_arc(0.0, 0.0, 0.0, 25.0, 0.016, 1.25, 3.0)
    x, z = ego.position_at_dist(24.0)
    car = build_arc(x, z, 0.0, 10.0, 0.0, 0.9, 3.0)
    ctx = _ctx(1)
    ctx.ego_arc = ego
    ctx.follow_fn = lambda: [[car]]
    ctx.all_target_arcs = [build_arc(-4.8, -15.0, 0.0, 10.0, 0.0, 0.9, 3.0)]
    assert not TargetFollowFilter(ON).apply(ctx).suppressed


def test_a_crosser_just_past_ego_path_point_but_in_front_of_its_bumper_is_kept():
    # A slow ego's path point reaches 10 m; its body runs 5 m further. A car crossing 13 m
    # ahead crosses where ego's front will be (5118b3f6).
    ego = build_arc(0.0, 0.0, 0.0, 4.0, 0.0, 1.25, 2.5, fwd_len=5.0, back_len=1.0)
    crosser = build_arc(-20.0, -13.0, math.radians(-90.0), 14.0, 0.0, 0.9, 2.5)
    ctx = _ctx(1)
    ctx.ego_arc = ego
    ctx.follow_fn = lambda: [[crosser]]
    ctx.all_target_arcs = [build_arc(-20.0, -13.0, math.radians(-90.0), 14.0, 0.0, 0.9, 2.5)]
    assert ego.arc_length < 13.0 - 1.25 - 0.9
    assert not TargetFollowFilter(ON).apply(ctx).suppressed


