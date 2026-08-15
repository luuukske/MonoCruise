"""Engagement-entry vetoes: head-on LOS bar, extrapolation veto, lane confidence.

Corpus context: 22 of 27 labelled false-positive engagements came from geometry
the arc extrapolation cannot support, most of them oncoming traffic on a bend
too gentle for the steer-derived ego curvature to register. See
core/aeb/README.md (engagement-entry vetoes).

The scenario pair below is that exact mechanism: ego and an oncoming vehicle
both physically on a 1200 m radius bend, ego reporting zero steer, so the ego
arc extrapolates straight off the curve and onto the oncoming lane. Only the
oncoming lane offset differs between the two cases.
"""
from __future__ import annotations

import math
import struct
from dataclasses import replace

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.clip_eval import run_headless
from core.aeb.clip_schema import (
    AEBTickRecord, Clip, ClipMetadata, ConsumedContext, EgoTelemetry, LiveAEB,
    RadarFrameRecord,
)
from core.aeb.lane_frame import Lane
from core.aeb.thread import _extrapolation_veto, _los_veto_bar
from core.radar.reader import _BUF_SIZE, _TOTAL_FORMAT
from core.radar.elevation import BODY_DATUM_FRAC


# Traffic position.y is the body datum, not ground level: ego coordinateY is
# the road surface and a body sits BODY_DATUM_FRAC of its height above it.
_BODY_H: float = 3.0
_BODY_Y: float = BODY_DATUM_FRAC * _BODY_H

_EGO_MS = 25.0            # 90 km/h
_TGT_MS = 20.0            # 72 km/h oncoming
_ROAD_R = 1200.0          # bend radius: 4050/R = 3.4 m of lateral shift over 90 m
_START_GAP = 90.0
_HZ = 30.0
# Wide enough that the bodies pass; the arc model still calls it a collision.
_CLEAR_OFFSET = 3.0
# Behaviour before the head-on bar and the lane-confidence gate existed.
_OLD = replace(
    CAL,
    los_veto_headon_miss_dist_m=CAL.los_veto_miss_dist_m,
    los_veto_headon_min_range_m=30.0,
    lane_confidence_range_m=9999.0,
    extrap_veto_enabled=False,
    aeb_engage_confirm_oblique_s=0.20,
    los_veto_min_range_m=30.0,
)


def _traffic_buf(px: float, pz: float, yaw_rad: float, speed: float,
                 vid: int = 3) -> bytes:
    """One-slot traffic buffer. Quaternion (w, 0, sin(yaw/2), 0) yields euler yaw."""
    qw = math.cos(yaw_rad / 2.0)
    qy = math.sin(yaw_rad / 2.0)
    flat: list = [px, _BODY_Y, pz, qw, 0.0, qy, 0.0, 2.5, _BODY_H, 6.0, speed, 0.0]
    flat += [0, vid, 0, 0] + [0.0] * 30
    for _ in range(39):
        flat += [0.0] * 12 + [0, 0, 0, 0] + [0.0] * 30
    buf = struct.pack(_TOTAL_FORMAT, *flat)
    assert len(buf) == _BUF_SIZE
    return buf


def _bend_clip(lane_offset_m: float, n: int = 60) -> Clip:
    """Ego and an oncoming vehicle on a shared bend; ego reports no steer.

    ``lane_offset_m`` is how far outside ego's radius the oncoming lane sits,
    so it is also the centreline clearance the two will actually pass with.
    """
    dt = 1.0 / _HZ
    r_t = _ROAD_R + lane_offset_m
    phi_t0 = math.pi - _START_GAP / _ROAD_R
    frames, ticks = [], []
    for i in range(n):
        t = i * dt
        phi_e = math.pi - _EGO_MS * t / _ROAD_R
        phi_t = phi_t0 + _TGT_MS * t / r_t
        frames.append(RadarFrameRecord(
            t_wall=1000.0 + t, t_mono=t,
            ego=EgoTelemetry(
                coordinateX=_ROAD_R + _ROAD_R * math.cos(phi_e),
                coordinateZ=_ROAD_R * math.sin(phi_e),
                rotationX=(-phi_e / (2.0 * math.pi)) % 1.0,
                rotationY=0.0, speed=_EGO_MS, userSteer=0.0,
            ),
            traffic_buf=_traffic_buf(
                _ROAD_R + r_t * math.cos(phi_t), r_t * math.sin(phi_t),
                math.pi - phi_t, _TGT_MS,
            ),
            parked_buf=None,
        ))
        ticks.append(AEBTickRecord(
            t_mono=t, radar_t_mono=t,
            consumed=ConsumedContext(max_brake_ms2=10.0, aeb_enabled=True),
            live_aeb=LiveAEB(),
        ))
    return Clip(
        metadata=ClipMetadata.create(trigger_source="auto_engagement",
                                     session_kind="SP"),
        radar_frames=frames, aeb_ticks=ticks,
    )


def test_wrong_way_driver_on_the_same_bend_still_engages():
    """Oncoming sharing ego's own path: measured miss is ~0, no veto may touch it."""
    assert any(e.aeb_brake for e in run_headless(_bend_clip(0.0)))


def test_oncoming_that_measurably_clears_does_not_engage():
    """Same bend, 3 m of measured clearance: pose says in-lane, the track says pass."""
    clip = _bend_clip(_CLEAR_OFFSET)
    assert any(e.aeb_brake for e in run_headless(clip, cal=_OLD)), (
        "precondition: this engaged before the engagement-entry vetoes existed"
    )
    assert not any(e.aeb_brake for e in run_headless(clip)), (
        "a target measured to pass 3 m clear must not trigger a brake"
    )


def test_vetoes_leave_engagement_alone_on_clear_pass():
    """Engage stays off on a measured clear pass; warn may stay quiet on short encounters.

    Persistence windows (`aeb_warn_confirm_oblique_s`, `aeb_warn_confirm_vetoed_s`)
    deliberately drop the cue when the raw condition flickers for under ~0.3 s.
    Brake must still stay off (see test_oncoming_that_measurably_clears_does_not_engage).
    """
    clip = _bend_clip(_CLEAR_OFFSET)
    evs = run_headless(clip)
    assert not any(e.aeb_brake for e in evs)

def test_narrow_clearance_is_still_a_collision():
    """2 m of centreline clearance between two 2.5 m bodies is contact, not a pass."""
    assert any(e.aeb_brake for e in run_headless(_bend_clip(2.0)))


def test_far_obstacle_still_engages_through_the_confirm_window():
    """The lane-confidence range costs instant certainty, never the engagement.

    A stopped obstacle past the range takes the oblique confirm window instead
    of the instant path, and still brakes well before reaching that range.
    """
    n, dt = 45, 1.0 / _HZ
    frames, ticks = [], []
    for i in range(n):
        t = i * dt
        frames.append(RadarFrameRecord(
            t_wall=1000.0 + t, t_mono=t,
            ego=EgoTelemetry(coordinateX=0.0, coordinateZ=_EGO_MS * t,
                             rotationX=0.5, rotationY=0.0, speed=_EGO_MS),
            traffic_buf=_traffic_buf(0.0, 55.0, 0.0, 0.0),
            parked_buf=None,
        ))
        ticks.append(AEBTickRecord(
            t_mono=t, radar_t_mono=t,
            consumed=ConsumedContext(max_brake_ms2=10.0, aeb_enabled=True),
            live_aeb=LiveAEB(),
        ))
    clip = Clip(metadata=ClipMetadata.create(session_kind="SP"),
                radar_frames=frames, aeb_ticks=ticks)
    ev = run_headless(clip)
    braked = [e for e in ev if e.aeb_brake]
    assert braked, "a stopped obstacle dead ahead must brake at any range"
    # Range at first brake, to prove the gate did not defer it to close quarters.
    assert 55.0 - _EGO_MS * braked[0].t_rel > CAL.lane_confidence_range_m


def test_los_veto_bar_scopes_the_tight_threshold_to_head_on():
    general = (CAL.los_veto_min_range_m, CAL.los_veto_miss_dist_m)
    headon = (CAL.los_veto_headon_min_range_m, CAL.los_veto_headon_miss_dist_m)

    assert _los_veto_bar(-0.99, 20.0, 0.001, CAL) == headon
    assert _los_veto_bar(0.99, 20.0, 0.001, CAL) == general      # co-directional
    assert _los_veto_bar(0.0, 20.0, 0.001, CAL) == general       # crossing
    # A manoeuvring target breaks the straight-line CBDR assumption.
    assert _los_veto_bar(-0.99, 20.0, 0.09, CAL) == general
    # kappa = yaw_rate / v is meaningless near zero speed, so it cannot disarm it.
    assert _los_veto_bar(-0.99, 0.5, 0.09, CAL) == headon


class _Ctx:
    """Minimal FilterContext stand-in for the pure veto predicate."""

    def __init__(self, lane, co_directional, ego_speed, ego_curvature):
        self.lane = lane
        self.co_directional = co_directional
        self.ego_speed = ego_speed
        self.ego_curvature = ego_curvature


def _veto(lane, co_dir, ego_ms, ego_k, ttc, tgt_along, in_lane_body=False,
          d_miss=9.0):
    return _extrapolation_veto(
        _Ctx(lane, co_dir, ego_ms, ego_k), ttc, tgt_along, in_lane_body, CAL,
        d_miss,
    )


def test_extrapolation_veto_never_touches_ego_lane():
    far_turn = dict(ttc=3.0, tgt_along=0.0)
    assert not _veto(Lane.EGO, False, 12.0, 0.05, **far_turn)
    # A trailer swung into ego's lane keeps the rig out of the veto (clip 434f0401).
    assert not _veto(Lane.OPPOSITE_OR_OUTER, False, 12.0, 0.05,
                     in_lane_body=True, **far_turn)


def test_turn_veto_needs_both_a_real_turn_and_a_far_hit():
    turning, straight = 0.05, 0.0
    far, near = 3.0, 0.4
    assert _veto(Lane.OFF_ROAD, False, 12.0, turning, far, 0.0)
    assert not _veto(Lane.OFF_ROAD, False, 12.0, straight, far, 0.0)
    assert not _veto(Lane.OFF_ROAD, False, 12.0, turning, near, 0.0)


def test_codir_veto_is_a_band_around_matched_speed():
    args = (Lane.OPPOSITE_OR_OUTER, True, 12.0, 0.0, 3.0)
    assert _veto(*args, 11.5)          # neighbour 0.5 m/s slower: lateral contact
    assert not _veto(*args, 6.0)       # ego closing 6 m/s: a real rear-end
    assert not _veto(*args, 16.0)      # faster overtaker: braking_worsens owns it


def test_codir_veto_yields_to_a_measurably_converging_track():
    """A neighbour drifting into us is a side contact, not lane-keeping traffic."""
    args = (Lane.OPPOSITE_OR_OUTER, True, 12.0, 0.0, 3.0, 11.5)
    assert _veto(*args, d_miss=4.0)
    assert not _veto(*args, d_miss=0.6)
    # No measurement yet: fail closed, the veto needs positive evidence.
    assert not _veto(*args, d_miss=None)


def test_veto_can_be_disabled_wholesale():
    off = replace(CAL, extrap_veto_enabled=False)
    assert not _extrapolation_veto(
        _Ctx(Lane.OFF_ROAD, False, 12.0, 0.05), 3.0, 0.0, False, off,
    )
