"""Bend pose: a parked body turned with the bend, standing a lane or more beside ego's way round it.

No map. Moving traffic near the body, or else the parked row it stands in, says how far the road
turns by there; the body must be turned the same way, and where that turn puts ego's lane against
where the body stands says whether it is in ego's lane or beside it. See ``core/aeb/README.md``,
"BendPoseFilter".
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from core.aeb.avoidability import measured_stationary
from core.aeb.calibration import AEBCalibration
from core.aeb.filters import _PASS, FilterResult, _pass, _suppress, lane_track_held

if TYPE_CHECKING:
    from core.aeb.filters import FilterContext


def fold_heading(rel: float) -> float:
    """Relative heading folded into [-pi/2, pi/2]: a body facing ego stands on the same road."""
    rel = math.atan2(math.sin(rel), math.cos(rel))
    if rel > math.pi / 2.0:
        rel -= math.pi
    elif rel < -math.pi / 2.0:
        rel += math.pi
    return rel


def outside_offset(ahead: float, left: float, bend: float, ego_kappa: float) -> float:
    """Metres the body stands outside ego's lane, curvature running linearly from ego's to ``bend``."""
    side = 1.0 if bend > 0.0 else -1.0
    lane_left = ego_kappa * ahead * ahead / 6.0 + bend * ahead / 3.0
    return side * (lane_left - left)


def traffic_bend(ctx: "FilterContext", ahead: float, side: float, cal: AEBCalibration,
                 moving: bool = True) -> float | None:
    """The road's turn at ``ahead`` read from traffic turned toward ``side``; None without any.

    Moving traffic keeps to its lane: the median of its turns. A parked row bends with the road
    only near the body, so the nearest parked body's turn is read.
    """
    fx, fz = ctx.ego_fwd_x, ctx.ego_fwd_z
    floor = math.radians(cal.bend_pose_mover_min_deg)
    turns: list[tuple[float, float]] = []
    for o in ctx.frame_vehicles:
        # A parked TruckersMP trailer record is a body of the row; a moving one is its tractor again.
        if o.id == ctx.v.id or (moving and getattr(o, "is_trailer", False)):
            continue
        if (abs(o.speed) >= cal.bend_pose_mover_min_speed_ms) != moving:
            continue
        dx, dz = o.position.x - ctx.ego_x, o.position.z - ctx.ego_z
        o_ahead = dx * fx + dz * fz
        if o_ahead < cal.bend_pose_mover_min_ahead_m or abs(o_ahead - ahead) > cal.bend_pose_mover_window_m:
            continue
        yaw = o._smooth_yaw if getattr(o, "_smooth_yaw", None) is not None else math.radians(
            o.rotation.euler()[1])
        turn = side * fold_heading(yaw - ctx.ego_yaw_rad)
        if turn >= floor:
            # The road's turn grows with distance: scale each one's to the body's range.
            turns.append((abs(o_ahead - ahead), turn * ahead / o_ahead))
    if not turns:
        return None
    if not moving:
        return side * min(turns)[1]
    turns.sort(key=lambda t: t[1])
    return side * turns[len(turns) // 2][1]


class BendPoseFilter:
    """Parked body turned with the bend traffic drives, standing outside ego's lane on that bend."""
    name = "BendPoseFilter"

    def __init__(self, cal: AEBCalibration) -> None:
        self._cal = cal

    def _agreeing_bend(self, ctx: "FilterContext", ahead: float, side: float, body: float) -> float | None:
        """The bend the body is turned with, moving traffic's first, then the parked vehicles'.

        A body parked askew may be turned less than the road: the bend is then taken between the two.
        """
        cal = self._cal
        tiers = [(True, cal.bend_pose_body_tol_frac)]
        if cal.bend_pose_parked_tol_frac > 0.0:
            tiers.append((False, cal.bend_pose_parked_tol_frac))
        for moving, tol in tiers:
            bend = traffic_bend(ctx, ahead, side, cal, moving=moving)
            if bend is None:
                continue
            over = max(math.radians(cal.bend_pose_body_tol_deg), tol * abs(bend))
            if cal.bend_pose_body_min_frac * abs(bend) <= abs(body) <= abs(bend) + over:
                low = min(abs(body), abs(bend))
                return side * (low + cal.bend_pose_bend_blend * (abs(bend) - low))
        return None

    def apply(self, ctx: "FilterContext") -> FilterResult:
        cal = self._cal
        if not cal.bend_pose_enabled:
            return _PASS
        v = ctx.v
        if v.id in ctx.latched_threat_ids or v.id in ctx.follow_threat_ids:
            return _PASS
        if getattr(v, "crash_confirmed", False) or not measured_stationary(ctx, cal):
            return _PASS
        # The map outranks the pose: a body it holds on ego's own track is kept (README lane track).
        if cal.lane_track_stationary and lane_track_held(ctx, cal):
            return _pass("lane_track")
        fx, fz = ctx.ego_fwd_x, ctx.ego_fwd_z
        ahead = ctx.dx * fx + ctx.dz * fz
        if not cal.bend_pose_min_ahead_m <= ahead <= cal.bend_pose_max_ahead_m:
            return _PASS
        body = fold_heading(ctx.v_yaw_rad - ctx.ego_yaw_rad)
        if not math.radians(cal.bend_pose_min_deg) <= abs(body) <= math.radians(cal.bend_pose_max_deg):
            return _PASS
        side = 1.0 if body > 0.0 else -1.0
        # Ego turning toward the body's side is a swerve or lane change toward it, not the road.
        if -side * ctx.ego_curvature * ctx.ego_speed * ctx.ego_speed > cal.bend_pose_max_toward_ms2:
            return _PASS
        bend = self._agreeing_bend(ctx, ahead, side, body)
        if bend is None:
            return _PASS
        # A bend ego could not follow at this speed is no road ego is on.
        if abs(bend) / ahead * ctx.ego_speed * ctx.ego_speed > cal.bend_pose_max_lat_ms2:
            return _PASS
        left = ctx.dx * fz - ctx.dz * fx
        if outside_offset(ahead, left, bend, ctx.ego_curvature) < cal.bend_pose_min_outside_m:
            return _PASS
        return _suppress("BendPoseFilter")
