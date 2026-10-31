"""Turn completion: a crosser turning into the oncoming lane straightens out beside ego.

A held turn keeps rotating past anti-parallel and curls into ego's lane. See
``core/aeb/README.md``, "TurnCompletionFilter".
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from core.aeb.calibration import AEBCalibration
from core.aeb.filters import _PASS, FilterResult, _suppress, travel_sign
from core.aeb.lane_frame import Lane, _any_body_in_ego_lane, project_to_ego_arc
from core.radar.traffic import build_arc

if TYPE_CHECKING:
    from core.aeb.filters import FilterContext


def turn_completed_path(
    ctx: "FilterContext", cal: AEBCalibration, n: int = 24,
) -> list[tuple[float, float, float, float]] | None:
    """(x, z, ux, uz) samples of a target that turns until anti-parallel to ego, then holds.

    None when the target is not turning toward anti-parallel.
    """
    v = ctx.v
    kappa = abs(ctx.v_curvature)
    length = ctx.abs_v_speed * ctx.dynamic_horizon
    if length < 1.0 or kappa < 1e-6:
        return None
    sign = travel_sign(v.speed, cal)
    ux, uz = sign * ctx.veh_fwd_x, sign * ctx.veh_fwd_z
    # Turn direction from the target's own arc, not heading_at_dist (README arc headings).
    probe = build_arc(v.position.x, v.position.z, ctx.v_yaw_rad, v.speed,
                      ctx.v_curvature, 0.5, ctx.dynamic_horizon)
    px, pz = probe.position_at_dist(min(2.0, probe.arc_length))
    turn = ux * (pz - v.position.z) - uz * (px - v.position.x)
    if abs(turn) < 1e-9:
        return None
    rot = math.copysign(kappa, turn)
    ax, az = -ctx.ego_fwd_x, -ctx.ego_fwd_z
    step = length / n
    c, s = math.cos(rot * step), math.sin(rot * step)
    # Rotating toward anti-parallel raises u . (-ego_fwd).
    if (ux * c - uz * s) * ax + (ux * s + uz * c) * az <= ux * ax + uz * az:
        return None
    x, z = v.position.x, v.position.z
    out = [(x, z, ux, uz)]
    turning = True
    for _ in range(n):
        if turning:
            nx, nz = ux * c - uz * s, ux * s + uz * c
            if nx * ax + nz * az < ux * ax + uz * az:
                turning = False
            else:
                ux, uz = nx, nz
        x += ux * step
        z += uz * step
        out.append((x, z, ux, uz))
    return out


class TurnCompletionFilter:
    """Crosser turning into the oncoming lane whose turn-then-hold path misses ego's corridor."""
    name = "TurnCompletionFilter"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal

    def apply(self, ctx: "FilterContext") -> FilterResult:
        cal = self._cal
        if not cal.turn_complete_enabled:
            return _PASS
        if ctx.head_on or ctx.co_directional or ctx.reversing:
            return _PASS
        if ctx.v.id in ctx.latched_threat_ids or ctx.v.id in ctx.follow_threat_ids:
            return _PASS
        if ctx.v.is_trailer or ctx.v.trailers:
            return _PASS
        if (ctx.abs_v_speed < cal.turn_complete_min_speed_ms
                or abs(ctx.v_curvature) < cal.turn_complete_min_kappa):
            return _PASS
        if ctx.lane == Lane.EGO or _any_body_in_ego_lane(
                ctx.ego_arc, ctx.all_target_arcs, cal.lane_half_width):
            return _PASS
        path = turn_completed_path(ctx, cal)
        if path is None:
            return _PASS
        bar = (ctx.ego_hw + ctx.v.size.width / 2.0 + cal.corridor_margin
               + cal.turn_complete_clear_m)
        half_len = ctx.v.size.length / 2.0
        reach = ctx.ego_arc.arc_length
        for x, z, ux, uz in path:
            for off in (-half_len, 0.0, half_len):
                s, d_abs = project_to_ego_arc(ctx.ego_arc, x + off * ux, z + off * uz)
                if 0.0 <= s <= reach and d_abs < bar:
                    return _PASS
        return _suppress("TurnCompletionFilter")


class TightTurnFilter:
    """Walking-pace turn: the arc wraps onto bodies the measured line says ego pulls away from."""
    name = "TightTurnFilter"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal

    def apply(self, ctx: "FilterContext") -> FilterResult:
        from core.aeb.avoidability import braking_deadline_passed, measured_miss_inside_body
        cal = self._cal
        if not cal.tight_turn_enabled or abs(ctx.ego_curvature) < cal.tight_turn_min_kappa:
            return _PASS
        if ctx.v.id in ctx.latched_threat_ids or ctx.v.id in ctx.follow_threat_ids:
            return _PASS
        rate = getattr(ctx, "d_miss_rate", None)
        if getattr(ctx, "d_miss", None) is None or rate is None or rate < cal.tight_turn_min_rate_mps:
            return _PASS
        # The guess ends where braking could no longer stop short (README avoidability gate).
        if measured_miss_inside_body(ctx) and braking_deadline_passed(ctx, cal):
            ctx.deadline_released = True
            return _PASS
        return _suppress("TightTurnFilter")
