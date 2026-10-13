"""Road-bent arcs for AEB: ego's and each target's predicted path leaned toward its lane.

The map worker publishes a route per vehicle (ego's and nearby movers'). Here
each frame picks, per vehicle, the candidate route needing the least change of
lateral force from what the vehicle does now, and bends its arcs with
``road_bend.bend_profile``. No route, no bend: the plain arc stands. See
core/aeb/README.md, "Road-bent trajectories".
"""

from __future__ import annotations

import math

from core.aeb.calibration import AEBCalibration
from core.radar.bent_path import bend_arc
from core.radar.traffic import ArcPath
from core.road_layout.road_bend import (
    BendParams, bend_profile, foot_kappa, lateral_offset, rank_routes, road_curve,
)

_MIN_SPEED_MS = 3.0
# Candidate routes tried, in order, for one the vehicle is following.
_ROUTE_TRIES = 3
_CACHE_MAX = 256
# A vehicle's profile is rebuilt this often; in between it is shifted along the path.
_PROFILE_REFRESH_S = 0.1
_PROFILE_MAX_SHIFT_M = 4.0
# Road curvature at each vehicle over this much of its recent travel joins the following
# test: the road-curve smoothing reaches about this far ahead of the foot.
_RECENT_M = 7.5
# Targets bent only within this range and not well behind ego (a cheap relevance cut).
_TARGET_RANGE_M = 120.0
_TARGET_BEHIND_M = 10.0


def bend_params(cal: AEBCalibration) -> BendParams:
    return BendParams(
        share=cal.road_bend_share,
        ramp_s=cal.road_bend_ramp_s,
        max_lat_ms2=cal.road_bend_max_lat_ms2,
    )


class RoadPaths:
    """Per-thread route curvature cache; ``frame`` binds one map snapshot."""

    def __init__(self) -> None:
        # (id(route), pick) -> (route, curve); holding the route keeps its id from being reused.
        self._curves: dict[tuple[int, int], tuple[object, object]] = {}
        self._snap = None
        self._params: BendParams | None = None
        self._length_s = 5.0
        self._now = 0.0
        # key -> (route, pick, t, x, z, ss, ks) of the last full profile build.
        self._profiles: dict = {}
        # key -> [(x, z, road curvature at the vehicle)] over the last _RECENT_M.
        self._recent: dict = {}
        # The lane ego's road path follows this frame, or None.
        self._ego_curve = None

    def frame(self, snap, cal: AEBCalibration, now: float = 0.0) -> None:
        """Bind this frame's map snapshot; None or a disabled calibration bends nothing."""
        self._snap = snap if (snap is not None and cal.road_bend_enabled) else None
        self._now = now
        self._ego_curve = None
        self._params = bend_params(cal)
        self._length_s = max(cal.arc_horizon_max, cal.clearance_horizon_s)
        self._targets = cal.road_bend_targets
        if len(self._curves) > _CACHE_MAX:
            self._curves.clear()
        if len(self._profiles) > _CACHE_MAX:
            self._profiles.clear()
        if len(self._recent) > _CACHE_MAX:
            self._recent.clear()

    @property
    def active(self) -> bool:
        return self._snap is not None

    def bend_ego(self, arcs: list[ArcPath], kappa: float) -> list[ArcPath]:
        route = getattr(self._snap, "route", None) if self._snap is not None else None
        return self._bend("ego", route, arcs, kappa)

    def bend_target(self, vid: int, arcs: list[ArcPath], kappa: float,
                    ego_pose: tuple[float, float, float, float] | None = None) -> list[ArcPath]:
        if self._snap is None or not self._targets or not arcs:
            return arcs
        if ego_pose is not None:
            ex, ez, efx, efz = ego_pose
            dx, dz = arcs[0].start_x - ex, arcs[0].start_z - ez
            if dx * dx + dz * dz > _TARGET_RANGE_M ** 2 or dx * efx + dz * efz < -_TARGET_BEHIND_M:
                return arcs
        route = (getattr(self._snap, "target_routes", None) or {}).get(vid)
        return self._bend(vid, route, arcs, kappa)

    def _bend(self, key, route, arcs: list[ArcPath], kappa: float) -> list[ArcPath]:
        if route is None or not arcs or self._params is None:
            return arcs
        lead = arcs[0]
        if lead.speed < _MIN_SPEED_MS or _reversing(lead):
            return arcs
        candidates = [route.pts] + list(getattr(route, "alternatives", ()))
        order = rank_routes(
            candidates, lead.start_x, lead.start_z, lead.fwd_x, lead.fwd_z,
            kappa, lead.speed, self._params,
        )
        # Least change of curvature first, but only a route the vehicle is following: of two
        # lanes becoming one, either may be the one it is on (README road-bent).
        ss, ks, curve = (), (), None
        merge_pick = getattr(route, "merge_index", None)
        merge_pick = None if merge_pick is None else merge_pick + 1
        tried: list[tuple[int, int]] = []
        for pick in order:
            # Branches share the route's start: one try covers every copy of the same lane.
            start = _start_key(candidates[pick])
            if start in tried:
                continue
            tried.append(start)
            if len(tried) > _ROUTE_TRIES:
                break
            ckey = (id(route), pick)
            entry = self._curves.get(ckey)
            if entry is None or entry[0] is not route:
                entry = (route, road_curve(candidates[pick]))
                self._curves[ckey] = entry
            if entry[1] is None:
                continue
            merge_s = getattr(route, "merge_s", None) if pick in (0, merge_pick) else None
            ss, ks = self._profile(key, route, pick, entry[1], lead, kappa, merge_s)
            if ss:
                curve = entry[1]
                break
        if curve is None:
            return arcs
        # Ego following its lane is answered along its path even in a steady bend.
        keep = key == "ego" and len(ss) >= 3
        if keep:
            self._ego_curve = curve
        # One profile from the lead arc: braked and trailer arcs share its geometry.
        return [bend_arc(arc, ss, ks, keep) for arc in arcs]

    def _profile(self, key, route, pick, curve, lead: ArcPath, kappa: float, merge_s=None):
        """Last profile shifted by the distance travelled, or a fresh one when it is stale."""
        prev = self._profiles.get(key)
        if prev is not None and prev[0] is route and prev[1] == pick:
            age = self._now - prev[2]
            moved = math.hypot(lead.start_x - prev[3], lead.start_z - prev[4])
            if 0.0 <= age < _PROFILE_REFRESH_S and moved < _PROFILE_MAX_SHIFT_M:
                if not prev[5]:
                    return (), ()
                ss = [0.0]
                ks = [kappa]
                for s, k in zip(prev[5][1:], prev[6][1:]):
                    if s - moved > 0.5:
                        ss.append(s - moved)
                        ks.append(k)
                if len(ss) >= 3:
                    return tuple(ss), tuple(ks)
        length = lead.speed * self._length_s + 10.0
        x, z = lead.start_x, lead.start_z
        recent = [r for r in self._recent.get(key, ()) if math.hypot(x - r[0], z - r[1]) <= _RECENT_M]
        ss, ks = bend_profile(
            curve, (x, z, lead.fwd_x, lead.fwd_z), kappa, lead.speed,
            length, 0.0, self._params, [r[2] for r in recent], merge_s,
        )
        recent.append((x, z, foot_kappa(curve, x, z)))
        self._recent[key] = recent
        self._profiles[key] = (route, pick, self._now, lead.start_x, lead.start_z, ss, ks)
        return ss, ks


    def in_ego_lane(self, x: float, z: float, half_width: float) -> bool:
        """``(x, z)`` ahead on the lane ego's road path follows, within ``half_width`` of its centre."""
        curve = self._ego_curve
        if curve is None:
            return False
        s, right = lateral_offset(curve, x, z)
        return 0.0 < s < curve.length and abs(right) <= half_width


def _start_key(pts) -> tuple[int, int]:
    """A candidate's start, to the metre, so copies of one lane are tried once."""
    return round(pts[0][0]), round(pts[0][1])


def _reversing(arc: ArcPath) -> bool:
    return arc.fwd_x * -math.sin(arc.yaw_rad) + arc.fwd_z * -math.cos(arc.yaw_rad) < 0.0
