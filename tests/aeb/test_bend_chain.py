"""BendChainFilter: a stopped oncoming car whose queue bends its lane beside ego.

README "BendChainFilter". The neighbours fit one circle through their positions and
headings; the car is dropped only when ego is not in that lane.
"""
from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

from core.aeb.avoidability import AvoidabilityGate
from core.aeb.bend_chain import BendChainFilter, fit_chain
from core.aeb.calibration import DEFAULT as CAL
from core.aeb.filters import FilterContext, build_pipeline
from core.radar.traffic import Quaternion
from tests.aeb.harness import make_vehicle

_R = 150.0
_LANE = 3.9
_QUEUE = (6.0, 18.0, 30.0, 42.0, 54.0)


def _heading(v) -> tuple[float, float]:
    yaw = math.radians(v.rotation.euler()[1])
    return -math.sin(yaw), -math.cos(yaw)


def _car(vid: int, s: float, offset: float, *, facing: bool = True, speed: float = 0.0,
         trailer: bool = False):
    """Car at arc s on a lane `offset` outside the ego lane (ego at the origin heading -z).

    The ego lane bends toward +x with radius _R. A facing car points back along the road.
    """
    a = s / _R
    x = _R * (1.0 - math.cos(a)) - offset * math.cos(a)
    z = -_R * math.sin(a) - offset * math.sin(a)
    sign = -1.0 if facing else 1.0
    hx, hz = sign * math.sin(a), -sign * math.cos(a)
    yaw = math.atan2(-hx, -hz)
    v = make_vehicle(vid, x, z, math.degrees(yaw), speed, is_trailer=trailer)
    v.rotation = Quaternion(math.cos(yaw / 2.0), 0.0, math.sin(yaw / 2.0), 0.0)
    return v


def _queue(offset: float = _LANE, positions=_QUEUE, **kw):
    return [_car(i + 1, s, offset, **kw) for i, s in enumerate(positions)]


def _ctx(frame, target, *, speed: float = 0.0, ground: float | None = 0.0, latched=()):
    fx, fz = _heading(target)
    return SimpleNamespace(
        v=target, frame_vehicles=frame, ego_x=0.0, ego_z=0.0, ego_fwd_x=0.0, ego_fwd_z=-1.0,
        veh_fwd_x=fx, veh_fwd_z=fz, abs_v_speed=speed, v_ground_meas=ground,
        latched_threat_ids=set(latched), follow_threat_ids=set(),
    )


def _suppressed(ctx, cal=CAL) -> bool:
    return BendChainFilter(cal).apply(ctx).suppressed


def test_the_cars_face_ego_along_the_bend():
    frame = _queue()
    fx, fz = _heading(frame[0])
    assert fz > 0.99 and abs(fx) < 0.1


def test_the_queue_fits_one_circle_through_the_lane():
    frame = _queue()
    target = frame[2]
    chain = fit_chain(target, [v for v in frame if v is not target], CAL)
    assert chain is not None
    k, members = chain
    assert len(members) == 5
    assert abs(abs(k) - 1.0 / (_R + _LANE)) < 0.1 / (_R + _LANE)


def test_a_queue_beside_ego_on_the_bend_is_dropped():
    frame = _queue()
    assert _suppressed(_ctx(frame, frame[2]))


def test_a_queue_on_the_inside_of_the_bend_is_dropped():
    frame = _queue(offset=-_LANE)
    assert _suppressed(_ctx(frame, frame[2]))


def test_the_same_queue_in_ego_own_lane_is_kept():
    frame = _queue(offset=0.0)
    assert not _suppressed(_ctx(frame, frame[2]))


def test_a_one_sided_queue_is_kept():
    frame = _queue(positions=(30.0, 42.0, 54.0, 66.0, 78.0))
    assert not _suppressed(_ctx(frame, frame[0]))
    frame = _queue(positions=(6.0, 18.0, 30.0, 42.0, 54.0))
    assert _suppressed(_ctx(frame, frame[2]))


def test_too_few_mates_are_kept():
    frame = _queue(positions=(18.0, 30.0, 42.0))
    assert not _suppressed(_ctx(frame, frame[1]))
    assert not _suppressed(_ctx([frame[1]], frame[1]))
    assert not _suppressed(_ctx([], frame[1]))
    assert _suppressed(_ctx(frame, frame[1]), replace(CAL, bend_chain_min_mates=2))


def test_trailer_records_are_not_mates():
    frame = _queue()
    for idx in (0, 4):
        frame[idx].is_trailer = True
    assert not _suppressed(_ctx(frame, frame[2]))
    frame = _queue()
    assert _suppressed(_ctx(frame, frame[2]))


def test_a_trailer_target_follows_the_switch():
    frame = _queue()
    frame[2].is_trailer = True
    assert _suppressed(_ctx(frame, frame[2]))
    assert not _suppressed(_ctx(frame, frame[2]), replace(CAL, bend_chain_trailer_targets=False))


def test_a_crawling_queue_is_dropped_only_until_the_braking_deadline():
    """The gate covers parked bodies only, so the stage holds a crawler to the deadline itself."""
    frame = _queue()
    target = frame[2]
    assert _suppressed(_ctx(frame, target, speed=2.0, ground=2.0))
    assert not _suppressed(_ctx(frame, target, speed=2.0, ground=2.0),
                           replace(CAL, bend_chain_max_speed_ms=0.0))
    ctx = _ctx(frame, target, speed=2.0, ground=2.0)
    ctx.deadline_passed, ctx.d_miss, ctx.ego_hw = True, 0.0, CAL.ego_half_width
    ctx.dx, ctx.dz = target.position.x, target.position.z
    ctx.v_yaw_rad = math.radians(target.rotation.euler()[1])
    assert not _suppressed(ctx)


def test_a_queue_driving_ego_way_is_kept():
    frame = _queue(facing=False)
    ctx = _ctx(frame, frame[2])
    assert not _suppressed(ctx)
    # Only the facing gate keeps it: the lane geometry alone would drop it.
    assert _suppressed(ctx, replace(CAL, bend_chain_min_facing=-1.0))


def test_moving_unknown_and_latched_targets_are_kept():
    frame = _queue()
    assert not _suppressed(_ctx(frame, frame[2], speed=5.0, ground=5.0))
    assert not _suppressed(_ctx(frame, frame[2], speed=0.0, ground=5.0))
    assert not _suppressed(_ctx(frame, frame[2], ground=None))
    assert not _suppressed(_ctx(frame, frame[2], latched={frame[2].id}))
    ctx = _ctx(frame, frame[2])
    ctx.follow_threat_ids = {frame[2].id}
    assert not _suppressed(ctx)


def test_switch_off_restores_the_pipeline():
    frame = _queue()
    off = replace(CAL, bend_chain_enabled=False)
    assert not _suppressed(_ctx(frame, frame[2]), off)


def test_the_gated_stage_follows_the_mirrored_corner_entry():
    pipe = build_pipeline(CAL)
    names = [s.name for s in pipe]
    at = names.index("CornerEntryStationaryFilterMirrored")
    assert names[at + 1] == "BendChainFilter"
    assert isinstance(pipe[at], AvoidabilityGate)
    assert isinstance(pipe[at + 1], AvoidabilityGate)
    assert isinstance(pipe[at + 1]._inner, BendChainFilter)
    assert names[at + 2] == "TightTurnFilter"
    assert names[at + 3] == "EgoEvasionFilter"


def test_the_context_carries_the_frame_vehicles():
    assert "frame_vehicles" in FilterContext.__dataclass_fields__
