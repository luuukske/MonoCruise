"""Reversing targets: regimes follow travel, and reversing is never oncoming.

Clip d80936f9 (TMP truck backing across ego's lane) and clip 66874532 (rig backing
into it). See core/aeb/README.md, travel frame.
"""
from __future__ import annotations

import math

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.filters import (
    FilterContext, LaneClassifier, TmpRelSpeedFilter,
    _body_centreline_d_abs, _build_vehicle_collision_data, travel_sign,
)
from core.radar.traffic import build_arc, capsule_extents
from tests.aeb.harness import make_vehicle

_EGO_SPEED = 11.0


def _ego_arc():
    offset = (CAL.arc_start_pctg - 0.5) * (2.0 * CAL.ego_half_length)
    fwd_len, back_len = capsule_extents(CAL.ego_half_length, CAL.ego_half_length, offset)
    # Ego at the origin heading +Z (yaw pi), straight.
    return build_arc(0.0, offset, math.pi, _EGO_SPEED, 0.0, CAL.ego_half_width,
                     CAL.arc_horizon_max, fwd_len=fwd_len, back_len=back_len)


def _ctx(v, tmp: bool = False) -> FilterContext:
    ego_arc = _ego_arc()
    arcs, pad, cross, v_yaw, abs_v, vfx, vfz, v_curv = _build_vehicle_collision_data(
        v, CAL.arc_horizon_max, math.pi, ego_arc.fwd_x, ego_arc.fwd_z, CAL,
    )
    return FilterContext(
        v=v, ego_arc=ego_arc, ego_braked_arc=ego_arc,
        ego_evasion_left=None, ego_evasion_right=None,
        ego_x=0.0, ego_y=0.0, ego_z=0.0, ego_yaw_rad=math.pi,
        ego_speed=_EGO_SPEED, ego_pitch_rad=0.0, ego_curvature=0.0,
        ego_fwd_x=ego_arc.fwd_x, ego_fwd_z=ego_arc.fwd_z, ego_hw=CAL.ego_half_width,
        dynamic_horizon=CAL.arc_horizon_max, tmp_traffic_session=tmp,
        ref_kmh_for_filter=_EGO_SPEED * 3.6, cal=CAL,
        dx=v.position.x, dz=v.position.z,
        dist_sq=v.position.x ** 2 + v.position.z ** 2,
        dist=math.hypot(v.position.x, v.position.z),
        v_yaw_rad=v_yaw, abs_v_speed=abs_v, veh_fwd_x=vfx, veh_fwd_z=vfz,
        v_curvature=v_curv, all_target_arcs=arcs,
        precomputed_cross_arcs=cross, cross_padding=pad,
    )


def _classified(v, tmp: bool = False) -> FilterContext:
    ctx = _ctx(v, tmp)
    LaneClassifier(CAL).apply(ctx)
    return ctx


def test_travel_sign_needs_a_real_reverse():
    """Near-standstill jitter keeps the heading frame; only a measured reverse flips it."""
    assert travel_sign(0.0, CAL) == 1.0
    assert travel_sign(-0.5 * CAL.reversing_speed_ms, CAL) == 1.0
    assert travel_sign(-2.0 * CAL.reversing_speed_ms, CAL) == -1.0


def test_forward_oncoming_is_unchanged():
    ctx = _classified(make_vehicle(vid=1, x=-4.0, z=30.0, yaw_deg=0.0, speed=10.0))
    assert ctx.head_on and ctx.near_head_on and not ctx.co_directional
    assert ctx.fwd_dot < -0.99
    assert ctx.lateral_gap == CAL.lane_separation


def test_reversing_toward_ego_is_not_oncoming():
    """Heading ego's way, backing at ego: travel is head-on, the oncoming class is not."""
    ctx = _classified(make_vehicle(vid=1, x=0.0, z=30.0, yaw_deg=180.0, speed=-3.0))
    assert ctx.fwd_dot < -0.99
    assert not ctx.head_on and not ctx.near_head_on and not ctx.co_directional
    # The oncoming lateral-gap exemption assumes a driver keeping to its own lane.
    assert ctx.lateral_gap == 0.0


def test_reversing_away_from_ego_is_co_directional():
    """Facing ego, backing away: it travels ego's way, whatever the heading says."""
    ctx = _classified(make_vehicle(vid=1, x=0.0, z=30.0, yaw_deg=0.0, speed=-3.0))
    assert ctx.co_directional and not ctx.head_on
    assert ctx.v_travel_speed == 3.0


def test_angled_reverse_across_the_lane_reads_crossing():
    """Clip d80936f9 pose: heading dot -0.67 read as oncoming in its own lane."""
    ctx = _classified(make_vehicle(vid=1, x=-4.5, z=12.0, yaw_deg=48.0, speed=-2.9))
    assert 0.6 < ctx.fwd_dot < 0.75
    assert not ctx.head_on and not ctx.near_head_on and not ctx.co_directional


def test_reversing_arc_does_not_assume_the_driver_brakes():
    """The full-brake arc is for an oncoming driver who can see ego; a reversing one cannot."""
    fwd = _build_vehicle_collision_data(
        make_vehicle(vid=1, x=0.0, z=30.0, yaw_deg=0.0, speed=5.0),
        CAL.arc_horizon_max, math.pi, 0.0, 1.0, CAL,
    )[0][0]
    rev = _build_vehicle_collision_data(
        make_vehicle(vid=1, x=0.0, z=30.0, yaw_deg=180.0, speed=-5.0),
        CAL.arc_horizon_max, math.pi, 0.0, 1.0, CAL,
    )[0][0]
    assert fwd.decel == CAL.full_brake_decel
    assert rev.decel == 0.0


def test_tmp_rel_speed_floor_still_owns_slow_reversing_traffic():
    """d80936f9 at 40 km/h stays silent by design: exempting reversing costs TMP FPs."""
    f = TmpRelSpeedFilter()
    truck = make_vehicle(vid=1, x=-4.5, z=12.0, yaw_deg=48.0, speed=-2.9, is_tmp=True)
    slow = _ctx(truck, tmp=True)
    assert slow.ref_kmh_for_filter <= CAL.tmp_filter_split_kmh
    assert f.apply(slow).suppressed
    fast = _ctx(truck, tmp=True)
    fast.ego_speed, fast.ref_kmh_for_filter = 20.0, 72.0
    assert not f.apply(fast).suppressed


def test_body_samples_stay_on_the_body_when_reversing():
    """Capsule extents are heading-relative; travel fwd used to mirror them off the body."""
    def samples(speed: float) -> list[float]:
        ctx = _ctx(make_vehicle(vid=1, x=-3.0, z=15.0, yaw_deg=48.0, speed=speed, length=6.0))
        return sorted(_body_centreline_d_abs(ctx.ego_arc, ctx.all_target_arcs))

    fwd, rev = samples(2.9), samples(-2.9)
    # Same pose, same body: the samples may not depend on which way it drives.
    assert all(abs(a - b) < 1e-6 for a, b in zip(fwd, rev))
