"""Arc-projected lane membership: replaces cross-product lateral_offset."""

from __future__ import annotations

import enum
import math

from core.radar.traffic import ArcPath
from core.aeb.calibration import AEBCalibration


class Lane(enum.IntEnum):
    EGO = 0
    ADJACENT = 1
    OPPOSITE_OR_OUTER = 2
    OFF_ROAD = 3


def project_to_ego_arc(ego_arc: ArcPath, x: float, z: float) -> tuple[float, float]:
    """Return (s, d_abs). Curved arcs use max(d_arc, d_straight); see core/aeb/README.md.

    A road-bent path answers along its own centreline: the bend already leaves ego
    on its steering, so the straight-line guard would only undo the road's lead.
    """
    if getattr(ego_arc, "bent", False):
        s, side = ego_arc.project(x, z)
        return s, abs(side)
    dx = x - ego_arc.start_x
    dz = z - ego_arc.start_z
    s_straight = dx * ego_arc.fwd_x + dz * ego_arc.fwd_z
    lat_x = dx - s_straight * ego_arc.fwd_x
    lat_z = dz - s_straight * ego_arc.fwd_z
    d_straight = math.sqrt(lat_x * lat_x + lat_z * lat_z)

    if ego_arc.is_straight:
        return s_straight, d_straight

    # Curved arc: project point onto arc circle
    cx, cz = ego_arc.center_x, ego_arc.center_z
    r_to_point = math.sqrt((x - cx) ** 2 + (z - cz) ** 2)
    d_arc = abs(ego_arc.radius - r_to_point)

    # Arc-length coordinate: angle from start to point, along arc direction
    angle_to_point = math.atan2(z - cz, x - cx)
    angle_delta = (angle_to_point - ego_arc.angle0) * (-ego_arc._sign)
    angle_delta = (angle_delta + math.pi) % (2 * math.pi) - math.pi
    s = angle_delta * ego_arc.radius

    return s, max(d_arc, d_straight)


# Fractions along the rigid body capsule (rear -> front) for _any_body_in_ego_lane.
# Five points bracket a long trailer with rear in-lane while centre rides outer.
_BODY_LANE_SAMPLES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _any_body_in_ego_lane(
    ego_arc: ArcPath, target_arcs: list[ArcPath], lane_half_width: float,
) -> bool:
    """Any target body centreline sample in ego lane at t=0 (trailer-in-lane rescue)."""
    return any(
        d <= lane_half_width
        for d in _body_centreline_d_abs(ego_arc, target_arcs)
    )


def _any_body_ahead_in_ego_lane(
    ego_arc: ArcPath, target_arcs: list[ArcPath], lane_half_width: float,
) -> bool:
    """Like _any_body_in_ego_lane, but only samples ahead of ego's front (the arc start)."""
    return any(
        s > 0.0 and d <= lane_half_width
        for s, d in _body_centreline_samples(ego_arc, target_arcs)
    )


# A mover counts as ahead in ego's lane only while travelling within this of the lane.
_ROAD_AHEAD_MAX_ANGLE_COS = math.cos(math.radians(45.0))
_ROAD_AHEAD_MOVING_MS = 2.0


def road_ahead_in_lane(
    road_arc: ArcPath, plain_arc: ArcPath, target_arcs: list[ArcPath], lane_half_width: float,
) -> bool:
    """A body ahead in ego's lane along the road-bent path but not along the plain arc.

    A moving target must also travel along the lane there: one crossing it or coming the
    other way at a junction curve stays with the plain arcs.
    """
    if not target_arcs:
        return False
    # One projection rules out the bodies nowhere near the road path before five each.
    lead = target_arcs[0]
    reach = lane_half_width + sum(a.fwd_len + a.back_len for a in target_arcs)
    s0, side0 = road_arc.project(lead.start_x, lead.start_z)
    if abs(side0) > reach or s0 < -reach:
        return False
    if _any_body_ahead_in_ego_lane(plain_arc, target_arcs, lane_half_width):
        return False
    moving = abs(target_arcs[0].speed) > _ROAD_AHEAD_MOVING_MS
    for s, d in _body_centreline_samples(road_arc, target_arcs):
        if s <= 0.0 or d > lane_half_width:
            continue
        if not moving:
            return True
        ax, az = road_arc.position_at_dist(s)
        bx, bz = road_arc.position_at_dist(s + 1.0)
        n = math.hypot(bx - ax, bz - az)
        tx, tz = target_arcs[0].fwd_x, target_arcs[0].fwd_z
        if n > 1e-6 and ((bx - ax) * tx + (bz - az) * tz) / n >= _ROAD_AHEAD_MAX_ANGLE_COS:
            return True
    return False


def _body_centreline_d_abs(
    ego_arc: ArcPath, target_arcs: list[ArcPath],
) -> list[float]:
    """Arc-projected |d| for centreline samples along each target body at t=0."""
    return [d for _, d in _body_centreline_samples(ego_arc, target_arcs)]


def _body_centreline_samples(
    ego_arc: ArcPath, target_arcs: list[ArcPath],
) -> list[tuple[float, float]]:
    """Arc-projected (s, |d|) for centreline samples along each target body at t=0."""
    out: list[tuple[float, float]] = []
    for arc in target_arcs:
        # Heading, not arc.fwd_*: capsule extents are heading-relative, fwd flips on reverse.
        fx, fz = -math.sin(arc.yaw_rad), -math.cos(arc.yaw_rad)
        back = -arc.back_len
        span = arc.fwd_len + arc.back_len
        for frac in _BODY_LANE_SAMPLES:
            s = back + span * frac
            px = arc.start_x + s * fx
            pz = arc.start_z + s * fz
            out.append(project_to_ego_arc(ego_arc, px, pz))
    return out


def classify(d_abs: float, cal: AEBCalibration) -> Lane:
    """Map absolute lateral offset to Lane enum."""
    if d_abs <= cal.lane_half_width:
        return Lane.EGO
    if d_abs < cal.lane_separation - cal.lane_half_width:
        return Lane.ADJACENT
    if d_abs < 2.0 * cal.lane_separation:
        return Lane.OPPOSITE_OR_OUTER
    return Lane.OFF_ROAD


def in_lane_closing(
    dx: float, dz: float, ego_fwd_x: float, ego_fwd_z: float,
    ego_speed: float, v_travel_speed: float, fwd_dot: float,
    lane_half_width: float, min_closing: float = 1.0,
) -> bool:
    """Ahead, |lat| inside the lane band, and closing. Straight frame, not arc d_abs."""
    axial = dx * ego_fwd_x + dz * ego_fwd_z
    if axial <= 0.0:
        return False
    lat = abs(-dx * ego_fwd_z + dz * ego_fwd_x)
    if lat > lane_half_width:
        return False
    closing = ego_speed - v_travel_speed * max(fwd_dot, 0.0)
    return closing > min_closing


def shares_bend(ego_curvature: float, v_curvature: float,
                cal: AEBCalibration) -> bool:
    """Both paths on one bend, so a wide arc offset is real; see core/aeb/README.md."""
    v_kappa = abs(v_curvature)
    if v_kappa < cal.turning_diverge_kappa:
        return False
    return v_kappa >= abs(ego_curvature) * cal.oncoming_shared_bend_ratio
