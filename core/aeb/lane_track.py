"""Lane track: where a body sits against ego's own track along its map lane.

A second opinion on a stage's verdict, never a verdict of its own. No map, no
sight, and every stage decides as before. See ``core/aeb/README.md``, "Lane track".
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.aeb.calibration import AEBCalibration

if TYPE_CHECKING:
    from core.aeb.road_paths import RoadPaths
    from core.radar.traffic import Vehicle


@dataclass(frozen=True)
class LaneSight:
    """One body in ego's lane frame: ``gap`` is its centre right of ego's own track."""

    ahead: float
    gap: float
    # Smallest distance of the body's centreline (rear, centre, front) from ego's track.
    reach: float
    # Cosine of the body's travel direction against the lane's.
    along: float


def lane_sight(road_paths: "RoadPaths", v: "Vehicle", yaw_rad: float, travel: float,
               keep: bool = False) -> LaneSight | None:
    """The body in ego's lane frame, or None with no lane bound or the centre out of reach."""
    centre = road_paths.lane_offset(v.position.x, v.position.z, keep)
    if centre is None:
        return None
    ahead, gap, tx, tz = centre
    fx, fz = -math.sin(yaw_rad), -math.cos(yaw_rad)
    reach = abs(gap)
    half = v.size.length * 0.5
    for off in (-half, half):
        end = road_paths.lane_offset(v.position.x + off * fx, v.position.z + off * fz, keep)
        if end is not None:
            reach = min(reach, abs(end[1]))
    return LaneSight(ahead, gap, reach, travel * (fx * tx + fz * tz))


def on_ego_track(sight: LaneSight | None, moving: bool, clear_bar: float, cal: AEBCalibration) -> bool:
    """The body overlaps ego's own track ahead; a mover must also travel along the lane."""
    if sight is None or sight.ahead > cal.lane_track_range_m:
        return False
    if sight.reach > clear_bar - cal.lane_track_overlap_m:
        return False
    return not moving or abs(sight.along) >= cal.lane_track_min_along


def clear_of_ego_track(
    sight: LaneSight | None, clear_bar: float, ego_right: float, drift_right: float,
    ego_speed: float, cal: AEBCalibration,
) -> bool:
    """The body sits clear of ego's own track, ego centred and calm in its lane (README lane clear)."""
    if (sight is None or abs(ego_right) > cal.lane_clear_max_offset_m
            or abs(drift_right) > cal.lane_clear_max_drift_ms):
        return False
    free = sight.reach - clear_bar
    if free < cal.lane_clear_min_free_m:
        return False
    toward = drift_right if sight.gap > 0.0 else -drift_right
    t_reach = min(sight.ahead / max(ego_speed, 1.0), cal.lane_clear_drift_horizon_s)
    return free - max(toward, 0.0) * t_reach >= cal.lane_clear_arrival_free_m


def lane_change_clear(road_paths: "RoadPaths", v: "Vehicle", yaw_rad: float, ego_hw: float,
                      ego_speed: float, cal: AEBCalibration) -> bool:
    """A parked body clear of both lanes of ego's lane change (README lane change clear)."""
    seen = road_paths.lane_change_free(v, yaw_rad, ego_hw)
    keep = road_paths.keep_frame
    if seen is None or keep is None:
        return False
    far, near, ahead, side = seen
    # On the side ego leaves, a turn back toward the body counts (1d7eb731 swerved back at 5 m/s^2).
    t_reach = min(ahead / max(ego_speed, 1.0), cal.lane_clear_drift_horizon_s)
    near -= _excursion(-side * keep[1], -side * keep[2], t_reach)
    if max(far, near) < cal.lane_change_min_free_m:
        return False
    return lane_drivable(road_paths, ahead, ego_speed, cal, keep=True)


def lane_drivable(road_paths: "RoadPaths", ahead: float, ego_speed: float,
                  cal: AEBCalibration, keep: bool = False) -> bool:
    """Ego's lane over the next ``ahead`` metres bends no harder than a truck at this speed can follow."""
    bend = road_paths.lane_bend(ahead, keep)
    return bend * ego_speed * ego_speed <= cal.lane_follow_max_lat_ms2


def _excursion(rate: float, accel: float, t: float) -> float:
    """Furthest a body gets toward the other lane within ``t``, moving across at ``rate`` and ``accel``."""
    peak = 0.0
    if accel < 0.0 < rate < -accel * t:
        peak = rate * rate / (-2.0 * accel)
    return max(rate * t + 0.5 * accel * t * t, peak, 0.0)


def lane_keep_free(
    road_paths: "RoadPaths", v: "Vehicle", yaw_rad: float, travel: float, kappa: float,
    ego_speed: float, clear_bar: float, cal: AEBCalibration,
    min_free: float | None = None, turn_credit: bool | None = None, out: dict | None = None,
) -> float | None:
    """Metres to spare between ego's track and a mover keeping its own map lane.

    None when the map cannot say: no lane frame, ego or the mover off its lane, a branch unknown.
    ``out``, when given, receives the mover's turn toward ego beyond its lane (``toward``, m/s^2).
    """
    min_free = cal.lane_keep_min_free_m if min_free is None else min_free
    turn_credit = cal.lane_keep_own_turn_credit if turn_credit is None else turn_credit
    ego = road_paths.keep_frame
    if ego is None or travel < 0.0:
        return None
    ego_right, ego_drift, ego_accel, ego_heading = ego
    if (abs(ego_right) > cal.lane_keep_max_ego_offset_m
            or abs(ego_heading) > math.radians(cal.lane_keep_max_ego_heading_deg)):
        return None
    sight = lane_sight(road_paths, v, yaw_rad, travel, keep=True)
    if sight is None or sight.reach - clear_bar < min_free:
        return None
    if not lane_drivable(road_paths, sight.ahead, ego_speed, cal, keep=True):
        return None
    speed = abs(v.speed)
    if speed < cal.lane_keep_min_speed_ms:
        return None
    own = road_paths.target_lane(
        v.id, v.position.x, v.position.z, max(speed * cal.lane_keep_follow_s, cal.lane_keep_follow_min_m))
    if own is None:
        return None
    offset, heading_err, kappa_lo, kappa_hi, left_toward = own
    if (abs(offset) > cal.lane_keep_max_offset_m
            or abs(heading_err) > math.radians(cal.lane_keep_max_heading_deg)):
        return None
    pose = None
    if not cal.lane_keep_all_branches:
        pose = (v.position.x, v.position.z, -math.sin(yaw_rad), -math.cos(yaw_rad), kappa, speed)
    gap = road_paths.route_gap(
        v.id, max(speed * cal.lane_keep_route_s, cal.lane_keep_route_min_m), pose)
    if gap is None:
        return None
    # Until they have passed, either may still move across its lane toward the other.
    closing = ego_speed - speed * sight.along
    t = cal.lane_keep_horizon_s
    if closing > 1.0:
        t = min(t, sight.ahead / closing)
    side = 1.0 if sight.gap > 0.0 else -1.0
    # Ego steering toward it counts once ego also moves that way or the turn is a swerve:
    # a steering transient is not a plan (README).
    ego_toward = side * ego_accel
    if (0.0 < ego_toward < cal.lane_keep_swerve_ms2
            and side * ego_drift < cal.lane_keep_turn_in_drift_ms):
        ego_toward = 0.0
    ego_across = _excursion(side * ego_drift, ego_toward, t)
    slack = min(abs(heading_err), math.radians(cal.lane_keep_heading_slack_deg))
    own_rate = speed * math.sin(heading_err - math.copysign(slack, heading_err))
    # Its turn counts only beyond what its lane does over the next stretch: a bend is no swerve.
    own_kappa = left_toward * kappa
    own_toward = max(speed * speed * (own_kappa - kappa_hi) - cal.lane_keep_follow_tol_ms2, 0.0)
    if turn_credit and own_kappa < kappa_lo:
        own_toward = speed * speed * (own_kappa - kappa_lo)
    across = ego_across + _excursion(own_rate, own_toward, t)
    if out is not None:
        out["toward"] = own_toward
    return gap - max(offset, 0.0) - clear_bar - across


def keeps_own_lane(
    road_paths: "RoadPaths", v: "Vehicle", yaw_rad: float, travel: float, kappa: float,
    ego_speed: float, clear_bar: float, cal: AEBCalibration, miss_rate: float | None = None,
) -> bool:
    """A mover on its own map lane whose lane stays clear of ego's track, ego on its own."""
    calm_bar = min(cal.lane_keep_min_free_m, cal.lane_keep_calm_free_m)
    seen: dict = {}
    free = lane_keep_free(road_paths, v, yaw_rad, travel, kappa, ego_speed, clear_bar, cal,
                          calm_bar, out=seen)
    # A mover turning toward ego beyond its lane's own turn keeps the full room (993fdee4).
    if free is not None and free >= (calm_bar if seen.get("toward", 1.0) <= 0.0
                                     else cal.lane_keep_min_free_m):
        return True
    # Lane return: the measured miss is opening and it turns away from ego's track.
    if (not cal.lane_return_enabled or miss_rate is None
            or not math.isfinite(miss_rate) or miss_rate < 0.0):
        return False
    own = road_paths.target_lane(v.id, v.position.x, v.position.z, cal.lane_keep_follow_min_m)
    if own is None or own[-1] * kappa >= 0.0:
        return False
    free = lane_keep_free(
        road_paths, v, yaw_rad, travel, kappa, ego_speed, clear_bar, cal,
        cal.lane_return_min_free_m, True)
    return free is not None and free >= cal.lane_return_min_free_m


# A body not judged for this long (out of range, off the surface) starts its hold over.
_MAX_GAP_S = 0.25


class LaneTrackHold:
    """Seconds each body has sat on ego's track without a break."""

    def __init__(self) -> None:
        # vid -> (on the track since, last frame it was judged on it).
        self._since: dict[int, tuple[float, float]] = {}

    def step(self, vid: int, on_track: bool, now: float) -> float:
        prev = self._since.get(vid)
        if not on_track:
            self._since.pop(vid, None)
            return 0.0
        if prev is None or now - prev[1] > _MAX_GAP_S:
            self._since[vid] = (now, now)
            return 0.0
        self._since[vid] = (prev[0], now)
        return now - prev[0]

    def prune(self, live_ids) -> None:
        for vid in list(self._since):
            if vid not in live_ids:
                del self._since[vid]
