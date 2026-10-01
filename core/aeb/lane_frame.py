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
    """Return (s, d_abs). Curved arcs use max(d_arc, d_straight); see core/aeb/README.md."""
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
