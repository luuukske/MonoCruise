"""Bend chain: a stopped oncoming car whose neighbours bend their lane past ego.

Vehicles ahead of and behind the target fit one circle through their positions and
headings. See ``core/aeb/README.md``, "BendChainFilter".
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from core.aeb.avoidability import (
    braking_deadline_passed, measured_miss_inside_body, measured_stationary,
)
from core.aeb.calibration import AEBCalibration
from core.aeb.filters import _PASS, FilterResult, _suppress

if TYPE_CHECKING:
    from core.aeb.filters import FilterContext


def _wrap_half(a: float) -> float:
    """Fold an angle into [-pi/2, pi/2]: a lane-mate may face either way."""
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    if a > math.pi / 2.0:
        a -= math.pi
    elif a < -math.pi / 2.0:
        a += math.pi
    return a


def _forward(v) -> tuple[float, float]:
    """Forward unit vector (x, z) from the raw euler yaw."""
    yaw = math.radians(v.rotation.euler()[1])
    return -math.sin(yaw), -math.cos(yaw)


def fit_chain(
    target, others, cal: AEBCalibration,
) -> tuple[float, list[tuple[float, float, float]]] | None:
    """Curvature and members (s, lo, dh) of the lane the target's neighbours form with it.

    Frame: s along the target's heading, lo along its left normal, dh the heading change.
    None unless enough mates, on both sides of the target, agree on one circle.
    """
    ux, uz = _forward(target)
    nx, nz = -uz, ux
    px, pz = target.position.x, target.position.z
    mates: list[tuple[float, float, float, float, float]] = []
    for n in others:
        dx, dz = n.position.x - px, n.position.z - pz
        s = dx * ux + dz * uz
        if not cal.bend_chain_s_min_m <= abs(s) <= cal.bend_chain_s_max_m:
            continue
        lo = dx * nx + dz * nz
        mx, mz = _forward(n)
        dh = _wrap_half(math.atan2(mx * nx + mz * nz, mx * ux + mz * uz))
        k_pos = 2.0 * lo / (s * s)
        k_head = dh / s
        if (abs(k_pos) > cal.bend_chain_k_max
                or abs(k_pos - k_head) > cal.bend_chain_k_tol
                or abs(lo - 0.5 * k_head * s * s) > cal.bend_chain_lat_tol_m):
            continue
        mates.append((s, lo, dh, k_pos, k_head))
    if len(mates) < cal.bend_chain_min_mates:
        return None
    if not any(m[0] > 0.0 for m in mates) or not any(m[0] < 0.0 for m in mates):
        return None
    k = sum(m[3] + m[4] for m in mates) / (2.0 * len(mates))
    return k, [(0.0, 0.0, 0.0)] + [(m[0], m[1], m[2]) for m in mates]


def ego_against_lane(
    target, ego_x: float, ego_z: float, ego_fwd_x: float, ego_fwd_z: float,
    k: float, members: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """(gap, offset, angle) of ego against the fitted lane, from the member nearest ego along it.

    Gap is ego's distance along that member's heading, offset its lateral distance from the
    lane there, angle its heading against the lane tangent (folded: the lane may face either way).
    """
    ux, uz = _forward(target)
    nx, nz = -uz, ux
    px, pz = target.position.x, target.position.z
    theta_t = math.atan2(uz, ux)
    theta_e = math.atan2(ego_fwd_z, ego_fwd_x)
    best = (math.inf, 0.0, 0.0)
    for s0, lo0, dh0 in members:
        h0 = theta_t + dh0
        c, sn = math.cos(h0), math.sin(h0)
        qx = ego_x - (px + s0 * ux + lo0 * nx)
        qz = ego_z - (pz + s0 * uz + lo0 * nz)
        se = qx * c + qz * sn
        if abs(se) >= abs(best[0]):
            continue
        le = -qx * sn + qz * c
        best = (se, le - 0.5 * k * se * se, _wrap_half(theta_e - (h0 + k * se)))
    return best


class BendChainFilter:
    """Stopped oncoming car in a queue whose lane bends beside ego, not through it."""
    name = "BendChainFilter"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal

    def apply(self, ctx: "FilterContext") -> FilterResult:
        cal = self._cal
        if not cal.bend_chain_enabled:
            return _PASS
        v = ctx.v
        if v.id in ctx.latched_threat_ids or v.id in ctx.follow_threat_ids:
            return _PASS
        if getattr(v, "is_trailer", False) and not cal.bend_chain_trailer_targets:
            return _PASS
        if cal.bend_chain_max_speed_ms > 0.0:
            ground = ctx.v_ground_meas
            if ground is None or max(ctx.abs_v_speed, ground) > cal.bend_chain_max_speed_ms:
                return _PASS
        elif not measured_stationary(ctx, cal):
            return _PASS
        facing = ctx.veh_fwd_x * ctx.ego_fwd_x + ctx.veh_fwd_z * ctx.ego_fwd_z
        if facing > -cal.bend_chain_min_facing:
            return _PASS
        others = [n for n in ctx.frame_vehicles
                  if n.id != v.id and not getattr(n, "is_trailer", False)]
        if len(others) < cal.bend_chain_min_mates:
            return _PASS
        chain = fit_chain(v, others, cal)
        if chain is None:
            return _PASS
        k, members = chain
        gap, off, ang = ego_against_lane(
            v, ctx.ego_x, ctx.ego_z, ctx.ego_fwd_x, ctx.ego_fwd_z, k, members)
        if not (abs(gap) <= cal.bend_chain_max_gap_m
                and abs(off) >= cal.bend_chain_clear_m
                and abs(ang) <= math.radians(cal.bend_chain_max_angle_deg)):
            return _PASS
        # A crawling car is outside the gate's parked scope: hold it to the same deadline here.
        if (not measured_stationary(ctx, cal) and measured_miss_inside_body(ctx)
                and braking_deadline_passed(ctx, cal)):
            ctx.deadline_released = True
            return _PASS
        return _suppress("BendChainFilter")
