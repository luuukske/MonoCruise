"""Avoidability gate: a guessed drop of a parked body ends at the braking deadline.

The wrapped stages drop a stationary body on a guess about ego's own path. The
guess stands while AEB could still stop short of the body if it is wrong. See
``core/aeb/README.md``, "Avoidability gate".
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from core.aeb.calibration import AEBCalibration
from core.aeb.cross_zone import _apply_cross_zone
from core.aeb.filters import _PASS, FilterResult, _earliest_hit
from core.aeb.lane_frame import Lane
from core.radar.traffic import ArcPath

if TYPE_CHECKING:
    from core.aeb.filters import FilterContext


def _cross_arc_groups(ctx: "FilterContext", cal: AEBCalibration) -> list[ArcPath]:
    """The padded target arcs the collision test uses for this vehicle."""
    fix_a = ctx.near_head_on and ctx.lane in (Lane.OPPOSITE_OR_OUTER, Lane.OFF_ROAD)
    out: list[ArcPath] = []
    for idx, base in enumerate(ctx.all_target_arcs):
        if fix_a:
            out.extend(_apply_cross_zone(
                base, ctx.cross_padding * cal.near_head_on_cross_scale, cal))
        elif ctx.precomputed_cross_arcs:
            out.extend(ctx.precomputed_cross_arcs[idx])
        else:
            out.extend(_apply_cross_zone(base, ctx.cross_padding, cal))
    return out


def _braking_deadline_passed(ctx: "FilterContext", cal: AEBCalibration) -> bool:
    clearance_fn = getattr(ctx, "clearance_fn", None)
    deadline = getattr(ctx, "brake_deadline_ms2", 0.0)
    if (clearance_fn is None or deadline <= 0.0 or getattr(ctx, "ego_arc", None) is None
            or not ctx.all_target_arcs):
        return False
    arcs = _cross_arc_groups(ctx, cal)
    hit = _earliest_hit(ctx.ego_arc, arcs, cal.corridor_margin, cal.collision_samples,
                        ctx.lateral_gap)
    # A body ego already overlaps is no prediction: no steer or brake changes it.
    if hit is None or hit[0] <= 0.0:
        return False
    cres = clearance_fn(arcs)
    return cres is not None and cres.required_ms2 >= deadline


def braking_deadline_passed(ctx: "FilterContext", cal: AEBCalibration) -> bool:
    """AEB engaging now could no longer stop ego short of the predicted hit."""
    verdict = getattr(ctx, "deadline_passed", None)
    if verdict is None:
        verdict = _braking_deadline_passed(ctx, cal)
        ctx.deadline_passed = verdict
    return verdict


def _body_corners(ctx: "FilterContext") -> list[tuple[float, float]]:
    """World corners of the target and each trailer, every body symmetric about its centre."""
    v = ctx.v
    bodies = [(v.position.x, v.position.z, ctx.v_yaw_rad, v.size.length, v.size.width)]
    for tr in getattr(v, "trailers", ()):
        bodies.append((tr.position.x, tr.position.z, math.radians(tr.rotation.euler()[1]),
                       tr.size.length, tr.size.width))
    out: list[tuple[float, float]] = []
    for cx, cz, yaw, length, width in bodies:
        fx, fz = -math.sin(yaw), -math.cos(yaw)
        for sl in (-0.5, 0.5):
            for sw in (-0.5, 0.5):
                out.append((cx + sl * length * fx - sw * width * fz,
                            cz + sl * length * fz + sw * width * fx))
    return out


def measured_miss_inside_body(ctx: "FilterContext") -> bool:
    """Measured CBDR line passes through some body, trailers included; no track fails closed."""
    miss = getattr(ctx, "d_miss", None)
    if miss is None:
        return False
    nx, nz = -ctx.ego_fwd_z, ctx.ego_fwd_x
    side = 1.0 if ctx.dx * nx + ctx.dz * nz >= 0.0 else -1.0
    px, pz = ctx.v.position.x, ctx.v.position.z
    toward = min(side * ((cx - px) * nx + (cz - pz) * nz) for cx, cz in _body_corners(ctx))
    return miss + toward <= ctx.ego_hw


def measured_stationary(ctx: "FilterContext", cal: AEBCalibration) -> bool:
    """Speed field and raw position track both say parked; an unknown track says no."""
    ground = getattr(ctx, "v_ground_meas", None)
    if ground is None:
        return False
    return max(ctx.abs_v_speed, ground) < cal.sweep_pass_max_target_speed


class AvoidabilityGate:
    """Wraps a stage that drops a stationary body on a guess about ego's own path.

    The drop stands only while braking could still stop short of the body if the guess is wrong.
    """

    def __init__(self, inner, cal: AEBCalibration) -> None:
        self._inner = inner
        self._cal = cal
        self.name = inner.name

    def apply(self, ctx: FilterContext) -> FilterResult:
        res = self._inner.apply(ctx)
        cal = self._cal
        if (res.suppressed and cal.avoidability_gate_enabled
                and measured_stationary(ctx, cal)
                and measured_miss_inside_body(ctx)
                and braking_deadline_passed(ctx, cal)):
            ctx.deadline_released = True
            return _PASS
        return res
