"""TurnCompletionFilter: a crosser turning into the oncoming lane straightens out beside ego.

README "TurnCompletionFilter". A held turn curls past anti-parallel into ego's lane;
the stage predicts turn-then-hold and drops the target only when that path never
reaches ego's corridor.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.filters import build_pipeline
from core.aeb.lane_frame import Lane
from core.aeb.turn_completion import TurnCompletionFilter, turn_completed_path
from core.radar.traffic import build_arc, capsule_extents

_EGO_OFFSET = (CAL.arc_start_pctg - 0.5) * (2.0 * CAL.ego_half_length)
_CAP_FWD, _CAP_BACK = capsule_extents(
    CAL.ego_half_length, CAL.ego_half_length, _EGO_OFFSET,
)
_EGO_V = 10.0
_V = 9.0
_R = 17.0
_Z0 = -30.0


def _yaw(fx: float, fz: float) -> float:
    """Yaw whose heading (-sin, -cos) points along (fx, fz)."""
    return math.atan2(-fx, -fz)


def _kappa_toward(x: float, z: float, yaw: float, want_dz: float) -> float:
    """Curvature sign that bends a target at (x, z) toward +z (want_dz > 0) or -z."""
    for k in (1.0 / _R, -1.0 / _R):
        px, pz = build_arc(x, z, yaw, _V, k, 0.9, 3.0).position_at_dist(2.0)
        if (pz - z) * want_dz > 0.0:
            return k
    raise AssertionError("no curvature bends the probe the asked way")


def _ctx(x: float, *, toward_ego: bool = True, lane: Lane = Lane.OFF_ROAD,
         latched: bool = False):
    """Ego at the origin heading -z at 10 m/s; a car 30 m ahead driving +x, turning."""
    ego_arc = build_arc(
        0.0, 0.0, 0.0, _EGO_V, 0.0, CAL.ego_half_width, 3.0,
        fwd_len=_CAP_FWD, back_len=_CAP_BACK,
        parallel_margin_scale=CAL.capsule_parallel_margin_scale,
    )
    yaw = _yaw(1.0, 0.0)
    kappa = _kappa_toward(x, _Z0, yaw, 1.0 if toward_ego else -1.0)
    body = build_arc(x, _Z0, yaw, _V, kappa, 0.9, 3.0, fwd_len=2.25, back_len=2.25)
    v = SimpleNamespace(
        id=11, position=SimpleNamespace(x=x, z=_Z0), speed=_V,
        size=SimpleNamespace(length=4.5, width=1.9), trailers=[], is_trailer=False,
    )
    return SimpleNamespace(
        v=v, v_yaw_rad=yaw, veh_fwd_x=-math.sin(yaw), veh_fwd_z=-math.cos(yaw),
        abs_v_speed=_V, v_curvature=kappa, dx=x, dz=_Z0,
        ego_fwd_x=0.0, ego_fwd_z=-1.0, ego_hw=CAL.ego_half_width, ego_arc=ego_arc,
        ego_speed=_EGO_V, all_target_arcs=[body], lane=lane, dynamic_horizon=3.0,
        head_on=False, near_head_on=False, co_directional=False, reversing=False,
        latched_threat_ids={11} if latched else set(), follow_threat_ids=set(),
    )


def _suppressed(ctx, cal=CAL) -> bool:
    return TurnCompletionFilter(cal).apply(ctx).suppressed


def test_turn_into_the_oncoming_lane_is_dropped():
    # Ends heading back at ego 4.5 m to its side: past the body bar plus the pad.
    assert _suppressed(_ctx(-4.5 - _R))


def test_the_completed_turn_holds_anti_parallel():
    path = turn_completed_path(_ctx(-4.5 - _R), CAL)
    _, _, ux, uz = path[-1]
    assert uz > 0.99 and abs(ux) < 0.15


def test_a_turn_across_ego_lane_is_kept():
    # Same turn started nearer: the path sweeps through ego's lane to reach the far side.
    assert not _suppressed(_ctx(-12.0))


def test_a_turn_toward_ego_heading_is_kept():
    assert turn_completed_path(_ctx(-4.5 - _R, toward_ego=False), CAL) is None
    assert not _suppressed(_ctx(-4.5 - _R, toward_ego=False))


def test_in_lane_and_latched_targets_are_kept():
    assert not _suppressed(_ctx(-4.5 - _R, lane=Lane.EGO))
    assert not _suppressed(_ctx(-4.5 - _R, latched=True))


def test_switch_off_restores_the_pipeline():
    from dataclasses import replace
    off = replace(CAL, turn_complete_enabled=False)
    assert not _suppressed(_ctx(-4.5 - _R), off)
    names = [s.name for s in build_pipeline(CAL)]
    assert names.index("TurnCompletionFilter") == names.index("TurningCrossTrafficFilter") + 1
