"""BendPoseFilter: a parked body turned with the bend, standing beside ego's lane on it.

README "BendPoseFilter". Moving traffic, or else the parked row, reads the bend; the body must be
turned with it and stand outside ego's lane on that bend by a lane or so.
"""
from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

from core.aeb.bend_pose import BendPoseFilter, fold_heading, outside_offset
from core.aeb.calibration import DEFAULT as CAL
from core.radar.traffic import Quaternion
from tests.aeb.harness import make_vehicle

_R = 400.0
_LANE = 3.9


def _on_bend(vid: int, s: float, outside: float, *, speed: float = 0.0, extra_deg: float = 0.0,
             trailer: bool = False):
    """Body at arc s of a left bend from ego (origin, heading -z), ``outside`` metres to its right."""
    a = s / _R
    x = -_R * (1.0 - math.cos(a)) + outside * math.cos(a)
    z = -_R * math.sin(a) - outside * math.sin(a)
    yaw = a + math.radians(extra_deg)
    v = make_vehicle(vid, x, z, math.degrees(yaw), speed, is_trailer=trailer)
    v.rotation = Quaternion(math.cos(yaw / 2.0), 0.0, math.sin(yaw / 2.0), 0.0)
    return v


def _ctx(target, frame, *, ego_speed: float = 25.0, ego_kappa: float = 1.0 / _R, ground=0.0,
         latched=()):
    return SimpleNamespace(
        v=target, frame_vehicles=frame, dx=target.position.x, dz=target.position.z,
        ego_x=0.0, ego_z=0.0, ego_fwd_x=0.0, ego_fwd_z=-1.0, ego_yaw_rad=0.0,
        v_yaw_rad=math.radians(target.rotation.euler()[1]), ego_speed=ego_speed,
        ego_curvature=ego_kappa, abs_v_speed=abs(target.speed), v_ground_meas=ground,
        latched_threat_ids=set(latched), follow_threat_ids=set(),
    )


def _dropped(ctx, cal=CAL) -> bool:
    return BendPoseFilter(cal).apply(ctx).suppressed


def test_a_facing_body_reads_the_same_bend():
    assert abs(fold_heading(math.pi - 0.1) + 0.1) < 1e-9
    assert abs(fold_heading(-math.pi + 0.1) - 0.1) < 1e-9


def test_ego_lane_on_the_bend_is_not_outside():
    a = 60.0 / _R
    assert abs(outside_offset(_R * math.sin(a), _R * (1.0 - math.cos(a)), a, 1.0 / _R)) < 0.2


def test_a_parked_car_a_lane_outside_the_bend_traffic_drives_is_dropped():
    target = _on_bend(1, 60.0, _LANE)
    mover = _on_bend(2, 70.0, 0.0, speed=20.0)
    assert _dropped(_ctx(target, [target, mover]))


def test_a_parked_row_along_the_bend_reads_it_without_traffic():
    target = _on_bend(1, 60.0, _LANE)
    row = _on_bend(2, 75.0, _LANE, trailer=True)
    assert _dropped(_ctx(target, [target, row]))


def test_a_body_in_ego_lane_on_the_bend_is_kept():
    target = _on_bend(1, 60.0, 0.0)
    mover = _on_bend(2, 70.0, 0.0, speed=20.0)
    assert not _dropped(_ctx(target, [target, mover]))


def test_a_lone_askew_body_is_kept():
    # Nothing else turned the same way: one crooked car on a straight road reads the same.
    target = _on_bend(1, 60.0, _LANE)
    straight = make_vehicle(2, 3.9, -80.0, 0.0, 20.0)
    assert not _dropped(_ctx(target, [target, straight]))


def test_a_body_turned_well_past_the_bend_traffic_drives_is_kept():
    # A crashed car askew in the lane: turned twice the road, it says nothing of where the road goes.
    target = _on_bend(1, 60.0, 0.0, extra_deg=10.0)
    mover = _on_bend(2, 70.0, 0.0, speed=20.0)
    assert not _dropped(_ctx(target, [target, mover]))


def test_a_moving_trailer_record_is_no_traffic():
    target = _on_bend(1, 60.0, _LANE)
    trailer = _on_bend(2, 70.0, 0.0, speed=20.0, trailer=True)
    assert not _dropped(_ctx(target, [target, trailer]))


def test_ego_turning_toward_the_body_keeps_it():
    # A lane change or swerve toward the body is not the road bending (1d7eb731).
    target = _on_bend(1, 60.0, _LANE)
    mover = _on_bend(2, 70.0, 0.0, speed=20.0)
    assert not _dropped(_ctx(target, [target, mover], ego_kappa=-0.004))


def test_a_moving_or_latched_body_is_kept():
    target = _on_bend(1, 60.0, _LANE)
    mover = _on_bend(2, 70.0, 0.0, speed=20.0)
    assert not _dropped(_ctx(target, [target, mover], ground=3.0))
    assert not _dropped(_ctx(target, [target, mover], latched=(1,)))


def test_a_body_only_just_outside_is_kept():
    target = _on_bend(1, 60.0, 2.5)
    mover = _on_bend(2, 70.0, 0.0, speed=20.0)
    assert not _dropped(_ctx(target, [target, mover]))


def test_off_switch_changes_nothing():
    target = _on_bend(1, 60.0, _LANE)
    mover = _on_bend(2, 70.0, 0.0, speed=20.0)
    assert not _dropped(_ctx(target, [target, mover]), replace(CAL, bend_pose_enabled=False))


def test_it_runs_bare_after_the_bend_chain():
    # Outside the avoidability gate by decision: at road speed the deadline has passed by first hit.
    from core.aeb.avoidability import AvoidabilityGate
    from core.aeb.filters import build_pipeline

    pipe = build_pipeline(CAL)
    names = [s.name for s in pipe]
    at = names.index("BendPoseFilter")
    assert names[at - 1] == "BendChainFilter" and names[at + 1] == "TightTurnFilter"
    assert not isinstance(pipe[at], AvoidabilityGate)


def test_the_map_holding_the_body_on_ego_track_outranks_the_pose():
    target = _on_bend(1, 60.0, _LANE)
    mover = _on_bend(2, 70.0, 0.0, speed=20.0)
    ctx = _ctx(target, [target, mover])
    assert _dropped(ctx)
    ctx.lane_track_s = CAL.lane_track_hold_s
    assert not _dropped(ctx)
