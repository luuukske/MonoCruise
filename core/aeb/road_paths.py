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
from core.radar.bent_path import BentPath, bend_arc
from core.radar.traffic import ArcPath
from core.road_layout.ego_lane import ranked_lanes
from core.road_layout.ego_route import build_route, route_behind
from core.road_layout.road_bend import (
    BendParams, _arc_xy, _curve_heading, _curve_point, _frange, _held_offset, _smoothstep, _wrap,
    bend_profile, fast_offset, foot_kappa, foot_offset, lateral_offset, rank_routes, road_curve,
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
# Ego's lane frame ends where another branch has left the route by this much.
_LANE_BRANCH_APART_M = 1.5
_LANE_REJOINED_M = 0.5
# A vehicle's lane is read against ego's track this often along it, at no more points than this.
_GAP_STEP_M = 3.0
_GAP_POINTS = 240
# The neighbour lane is searched this far ahead of ego, this far to the side, and must lie
# within this distance of the probe.
_CHANGE_AHEAD_M = 10.0
_CHANGE_PROBES_M = (3.5, 4.5, 5.5)
_CHANGE_SNAP_M = 1.5
# Settling back in its own lane needs ego already braking its way across by this share.
_SETTLE_SHARE = 0.5


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
        # (curve, ego s, ego right, certain reach) of ego's lane frame this frame, or None.
        self._lane = None
        # The same on the branch ego follows through a fork; the lane-keep verdict reads this one.
        self._keep = None
        self.lane_ego_right = 0.0
        self.lane_drift_right = 0.0
        # (side, target curve or None to settle in ego's own lane) while ego changes lanes.
        self._change = None
        # (window, its lanes' points) for the neighbour-lane search.
        self._lane_pts = None

    def frame(self, snap, cal: AEBCalibration, now: float = 0.0) -> None:
        """Bind this frame's map snapshot; None or a disabled calibration bends nothing."""
        self._snap = snap if (snap is not None and cal.road_bend_enabled) else None
        self._now = now
        self._ego_curve = None
        self._lane = None
        self._keep = None
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

    def target_following(self, vid: int, x: float, z: float, kappa: float, speed: float,
                         cal: AEBCalibration) -> bool:
        """The mover sits on its own map lane, turned along it and turning as it does (README)."""
        route = (getattr(self._snap, "target_routes", None) or {}).get(vid) if self._snap else None
        if route is None or speed < _MIN_SPEED_MS or self._params is None or len(route.pts) < 2:
            return False
        if abs(route.offset_m) > cal.target_follow_max_offset_m:
            return False
        # Its turn against its lane's over the next stretch on any branch and the stretch
        # behind, as lane keep reads it: a measured turn trails the lane (README lane keep).
        reach = max(speed * cal.lane_keep_follow_s, cal.lane_keep_follow_min_m)
        lead = max(speed * cal.target_follow_heading_lead_s, _FOLLOW_STEP_M)
        ks: list[float] = []
        lane_turn = 0.0
        branches = [route.pts] + list(getattr(route, "alternatives", ()))
        tol = cal.target_follow_tol_ms2 / max(speed * speed, 25.0)

        def holds(turn: float) -> bool:
            # Turning a little early or late reads as heading error where the lane turns: the
            # lane's turn within the lead time is excused, the rest is a move across it (README).
            excess = max(abs(route.heading_err_rad) - turn, 0.0)
            return bool(ks) and (excess <= math.radians(cal.target_follow_max_heading_deg)
                                 and speed * math.sin(excess) <= cal.target_follow_max_lat_rate_ms
                                 and min(ks) - tol <= kappa <= max(ks) + tol)

        # Other branches and the lane behind only widen both allowances: each is read only
        # when what came before falls short, the vehicle's own lane first.
        for pick, pts in enumerate(branches[:cal.target_follow_max_branches]):
            curve = self._route_curve(route, pick, pts)
            if curve is None:
                continue
            s_off = foot_offset(curve, x, z)
            if abs(_held_offset(curve, s_off, (x, z, 1.0, 0.0))) > cal.target_follow_max_offset_m:
                continue
            ks.extend(curve.kappa_at(s_off + d) for d in _frange(0.0, reach, _GAP_STEP_M))
            lane_turn = max(lane_turn, abs(_lane_turn(curve, s_off, lead)))
            if holds(lane_turn):
                return True
        if not ks:
            return False
        back = self._behind_curve(route, x, z)
        if back is not None:
            # The lane behind runs backwards from the vehicle: its turns read mirrored.
            ks.extend(-back.kappa_at(d) for d in _frange(0.0, back.length, _GAP_STEP_M))
            lane_turn = max(lane_turn, abs(_lane_turn(back, 0.0, lead)))
        return holds(lane_turn)

    def _behind_curve(self, route, x: float, z: float):
        """The vehicle's lane over the stretch just behind it, run backwards, or None."""
        (ax, az), (bx, bz) = route.pts[0], route.pts[1]
        hop = math.hypot(bx - ax, bz - az)
        if hop <= 1e-6:
            return None
        behind = route_behind(getattr(self._snap, "window", None), route.lane_index, x, z,
                              (bx - ax) / hop, (bz - az) / hop, 2.0 * _RECENT_M)
        return road_curve(behind) if len(behind) >= 2 else None

    def follow_target(self, vid: int, arcs: list[ArcPath], cal: AEBCalibration,
                      ego_xz: tuple[float, float] | None = None) -> list[list[ArcPath]] | None:
        """The mover's arcs along every branch of its lane, joined by position (README target follow).

        None when any branch cannot be followed: no verdict, the plain arcs stand.
        """
        route = (getattr(self._snap, "target_routes", None) or {}).get(vid) if self._snap else None
        if route is None or not arcs or self._params is None:
            return None
        lead = arcs[0]
        if lead.speed < _MIN_SPEED_MS or _reversing(lead):
            return None
        branches = [route.pts] + list(getattr(route, "alternatives", ()))
        if len(branches) > cal.target_follow_max_branches:
            return None
        merge_pick = getattr(route, "merge_index", None)
        merge_pick = None if merge_pick is None else merge_pick + 1
        length = lead.arc_length + 10.0
        ramp = max(cal.target_follow_ramp_min_m, lead.speed * cal.target_follow_ramp_s)
        out: list[list[ArcPath]] = []
        for pick, pts in enumerate(branches):
            curve = self._route_curve(route, pick, pts)
            if curve is None:
                return None
            s_off = foot_offset(curve, lead.start_x, lead.start_z)
            pose = (lead.start_x, lead.start_z, lead.fwd_x, lead.fwd_z)
            # An overlapping lane the vehicle is not on is not one of its branches.
            if abs(_held_offset(curve, s_off, pose)) > cal.target_follow_max_offset_m:
                continue
            # Nor is one it could not drive at its speed (a prefab's U-turn curve).
            if any(lead.speed * lead.speed * abs(curve.kappa_at(s_off + d)) > cal.target_follow_max_lat_ms2
                   for d in _frange(0.0, min(length, curve.length - s_off), _GAP_STEP_M)):
                continue
            merge_s = getattr(route, "merge_s", None) if pick in (0, merge_pick) else None
            drift = _lane_drift(curve, s_off, lead, cal)
            # Only a move toward ego counts, as in lane keep: one away is no room to rely on.
            if ego_xz is None:
                drift = _NO_DRIFT
            else:
                ego_side = _held_offset(curve, foot_offset(curve, *ego_xz), (*ego_xz, 1.0, 0.0))
                toward = ego_side - _held_offset(curve, s_off, (lead.start_x, lead.start_z, 1.0, 0.0))
                if drift[0] * toward <= 0.0:
                    drift = _NO_DRIFT
            branch: list[ArcPath] = []
            # Each body (tractor, trailers) joins the lane from where it is.
            for arc in arcs:
                path = _follow_path(curve, arc, length, ramp, merge_s, drift)
                if path is None:
                    return None
                branch.append(path)
            out.append(branch)
        return out or None

    def follow_ego(self, ego_arc: ArcPath, cal: AEBCalibration) -> list[ArcPath]:
        """Ego along every lane of its route it could take, the way a target is followed (README).

        Through a junction ego's branch is not known until it steers for it.
        """
        route = getattr(self._snap, "route", None) if self._snap is not None else None
        if route is None or ego_arc.speed < 1e-3:
            return []
        branches = [route.pts] + list(getattr(route, "alternatives", ()))
        length = ego_arc.arc_length + 10.0
        ramp = max(cal.target_follow_ramp_min_m, ego_arc.speed * cal.target_follow_ramp_s)
        pose = (ego_arc.start_x, ego_arc.start_z, ego_arc.fwd_x, ego_arc.fwd_z)
        out: list[ArcPath] = []
        for pick, pts in enumerate(branches[:cal.target_follow_max_branches]):
            curve = self._route_curve(route, pick, pts)
            if curve is None:
                continue
            s_off = foot_offset(curve, ego_arc.start_x, ego_arc.start_z)
            if abs(_held_offset(curve, s_off, pose)) > cal.lane_track_max_offset_m:
                continue
            path = _follow_path(curve, ego_arc, length, ramp)
            if path is not None:
                out.append(path)
        return out

    def _route_curve(self, route, pick: int, pts):
        ckey = (id(route), pick)
        entry = self._curves.get(ckey)
        if entry is None or entry[0] is not route:
            entry = (route, road_curve(pts))
            self._curves[ckey] = entry
        return entry[1]

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

    def bind_ego_lane(self, x: float, z: float, cal: AEBCalibration,
                      vel_x: float = 0.0, vel_z: float = 0.0, kappa: float = 0.0) -> bool:
        """Place ego on its map lane for ``lane_offset``, by position only (README lane track)."""
        self._lane = None
        self._keep = None
        self._change = None
        self.lane_ego_right = 0.0
        self.lane_drift_right = 0.0
        route = getattr(self._snap, "route", None) if self._snap is not None else None
        if route is None or not cal.lane_track_enabled:
            return False
        # The branch ego's steering follows carries the lane-keep frame through a fork.
        if self._ego_curve is not None and cal.lane_keep_follow_branch:
            self._keep = _frame(self._ego_curve, self._ego_curve.length, x, z, vel_x, vel_z, kappa)
        if (abs(route.offset_m) > cal.lane_track_max_offset_m
                or abs(route.heading_err_rad) > math.radians(cal.lane_track_max_heading_deg)):
            return False
        ckey = (id(route), -1)
        entry = self._curves.get(ckey)
        if entry is None or entry[0] is not route:
            entry = (route, road_curve(route.pts))
            self._curves[ckey] = entry
        curve = entry[1]
        if curve is None:
            return False
        reach = curve.length
        div = getattr(route, "divergence", ())
        for k, (s, apart) in enumerate(div):
            if apart < _LANE_BRANCH_APART_M:
                continue
            # A lane that starts just ahead of ego reads as apart at the first sample only.
            if k == 0 and len(div) > 1 and div[1][1] < _LANE_REJOINED_M:
                continue
            reach = min(reach, s)
            break
        frame = _frame(curve, reach, x, z, vel_x, vel_z, kappa)
        if frame is None:
            return False
        self._lane = frame
        if self._keep is None:
            self._keep = frame
        self.lane_ego_right = frame[2]
        # Ego's speed across its lane, rightward: the drift a lane-clear verdict must survive.
        self.lane_drift_right = frame[4]
        return True

    @property
    def keep_frame(self) -> tuple[float, float, float, float] | None:
        """Ego on the lane it keeps: metres right of it, speed and accel across it, heading error."""
        if self._keep is None:
            return None
        return self._keep[2], self._keep[4], self._keep[6], self._keep[5]

    def lane_offset(self, x: float, z: float,
                    keep: bool = False) -> tuple[float, float, float, float] | None:
        """Metres ahead along ego's lane, metres right of ego's own track, and the lane direction.

        None behind ego, past the lane's certain reach, or with no lane bound.
        """
        frame = self._keep if keep else self._lane
        if frame is None:
            return None
        curve, ego_s, ego_right, reach = frame[:4]
        s, right = fast_offset(curve, x, z)
        if not math.isfinite(right) or s <= ego_s or s >= reach:
            return None
        h = _curve_heading(curve, s)
        if h is None:
            return None
        return s - ego_s, right - ego_right, math.cos(h), math.sin(h)

    def lane_bend(self, ahead: float, keep: bool = False) -> float:
        """Sharpest curvature of ego's lane over the next ``ahead`` metres; 0 with no frame."""
        frame = self._keep if keep else self._lane
        if frame is None:
            return 0.0
        curve, ego_s = frame[0], frame[1]
        return max(abs(curve.kappa_at(ego_s + d)) for d in _frange(0.0, max(ahead, 0.0), _GAP_STEP_M))

    def target_lane(self, vid: int, x: float, z: float,
                    reach: float) -> tuple[float, float, float, float, float] | None:
        """A mover against its own map lane, every value positive toward ego's own track.

        ``(metres off it, heading error, least and most lane curvature over the next ``reach``
        metres, +1 when its own left turn leads toward ego's track)``. None with no route or
        outside ego's lane frame.
        """
        route = (getattr(self._snap, "target_routes", None) or {}).get(vid) if self._snap else None
        at = self.lane_offset(x, z, keep=True)
        if route is None or at is None or len(route.pts) < 2:
            return None
        (ax, az), (bx, bz) = route.pts[0], route.pts[1]
        hop = math.hypot(bx - ax, bz - az)
        if hop < 1e-6:
            return None
        # Its lane's right-hand normal against the direction from it to ego's track.
        nx, nz = -(bz - az) / hop, (bx - ax) / hop
        _ahead, right, tx, tz = at
        side = -1.0 if right > 0.0 else 1.0
        sign = 1.0 if side * (nx * -tz + nz * tx) >= 0.0 else -1.0
        ckey = (id(route), -2)
        entry = self._curves.get(ckey)
        if entry is None or entry[0] is not route:
            entry = (route, road_curve(route.pts))
            self._curves[ckey] = entry
        # Curvature is positive to the left; ``sign`` says the lane's right points at ego's track.
        ks = [0.0]
        if entry[1] is not None:
            ks = [-sign * entry[1].kappa_at(d) for d in _frange(0.0, reach, _GAP_STEP_M)]
        # The lane just behind counts too: a vehicle's turn trails the bend it has left.
        behind = route_behind(
            getattr(self._snap, "window", None), route.lane_index, x, z,
            (bx - ax) / hop, (bz - az) / hop, 2.0 * _RECENT_M)
        back = road_curve(behind) if len(behind) >= 2 else None
        if back is not None:
            ks.extend(sign * back.kappa_at(d) for d in _frange(0.0, back.length, _GAP_STEP_M))
        return sign * route.offset_m, -sign * route.heading_err_rad, min(ks), max(ks), -sign

    def route_gap(self, vid: int, length: float, pose: tuple | None = None) -> float | None:
        """Smallest distance from ego's own track to the vehicle's lane, ``length`` m along it.

        Every branch of that lane counts, or with ``pose`` (x, z, fwd_x, fwd_z, kappa, speed)
        only the branch it steers for. None with no route or a branch leaving ego's lane frame.
        """
        route = (getattr(self._snap, "target_routes", None) or {}).get(vid) if self._snap else None
        if route is None or self._keep is None:
            return None
        branches = [route.pts, *getattr(route, "alternatives", ())]
        if pose is not None and len(branches) > 1 and self._params is not None:
            order = rank_routes(branches, *pose, self._params)
            branches = [branches[order[0]]] if order else branches[:1]
        curve, ego_s, ego_right, reach = self._keep[:4]
        best = math.inf
        seen: set[tuple[int, int]] = set()
        for pts in branches:
            for x, z in _walk(pts, length):
                key = (round(x), round(z))
                if key in seen:
                    continue
                seen.add(key)
                if len(seen) > _GAP_POINTS:
                    return None
                s, right = fast_offset(curve, x, z)
                # Behind ego it has passed; past the frame nobody knows.
                if s <= ego_s:
                    continue
                if s >= reach or not math.isfinite(right):
                    return None
                best = min(best, abs(right - ego_right))
        return best

    def bind_lane_change(self, x: float, z: float, cal: AEBCalibration) -> bool:
        """Ego moving across its lane toward one it can settle in (README lane change clear).

        The first half heads for the neighbour lane on that side, the second half settles in
        ego's own lane. False when ego is calm, heads off too steeply or could not settle.
        """
        self._change = None
        keep = self._keep
        if keep is None or self._snap is None:
            return False
        curve, ego_s, ego_right, _reach, drift, heading, accel = keep[:7]
        if (abs(drift) < cal.lane_change_min_drift_ms
                or abs(heading) > math.radians(cal.lane_change_max_heading_deg)):
            return False
        side = 1.0 if drift > 0.0 else -1.0

        def settles(remaining: float) -> float | None:
            """Lateral decel that stops ego by ``remaining`` plus the overshoot, or None."""
            room = remaining + cal.lane_change_overshoot_m
            if room <= 0.0 or drift * drift > 2.0 * cal.lane_change_settle_ms2 * room:
                return None
            return drift * drift / (2.0 * room)

        # Back to its own lane centre only while ego already slows its way across (ac7bb3ab
        # crossed its centre toward parked cars at 1 m/s); else on into the next lane.
        need = settles(-side * ego_right) if side * ego_right < 0.0 else None
        if need is not None and -side * accel >= _SETTLE_SHARE * need:
            self._change = (side, None)
            return True
        target = self._neighbour_lane(curve, ego_s, side, cal)
        if target is None or not settles(-side * fast_offset(target, x, z)[1]):
            return False
        self._change = (side, target)
        return True

    def _neighbour_lane(self, curve, ego_s: float, side: float, cal: AEBCalibration):
        """The lane beside ego's on ``side``, as a road curve from just ahead of ego, or None."""
        window = getattr(self._snap, "window", None)
        lanes = getattr(window, "lanes", ())
        near = getattr(window, "index", None)
        if not lanes or near is None:
            return None
        s = min(ego_s + _CHANGE_AHEAD_M, curve.length - 1.0)
        p, h = _curve_point(curve, s), _curve_heading(curve, s)
        if p is None or h is None:
            return None
        tx, tz = math.cos(h), math.sin(h)
        if self._lane_pts is None or self._lane_pts[0] is not window:
            self._lane_pts = (window, tuple(lane.pts for lane in lanes))
        pts = self._lane_pts[1]
        for w in _CHANGE_PROBES_M:
            px, pz = p[0] - side * w * tz, p[1] + side * w * tx
            for index, dist, _cost in ranked_lanes(pts, px, pz, tx, tz, near, 0.0, limit=4):
                if dist > _CHANGE_SNAP_M:
                    continue
                route = build_route(lanes, near, index, px, pz, tx, tz)
                found = road_curve(route.pts) if route is not None else None
                if found is None:
                    continue
                off = side * fast_offset(curve, *found.pts[0])[1]
                if cal.lane_change_lane_min_m <= off <= cal.lane_change_lane_max_m:
                    return found
        return None

    def lane_change_free(self, v, yaw_rad: float,
                         ego_hw: float) -> tuple[float, float, float, float] | None:
        """A body against ego's lane change: ``(free beyond where ego settles, free on the side it
        leaves, metres ahead, side ego moves to)``. None with no lane change bound or a corner
        behind ego or past the lane frame.
        """
        if self._change is None or self._keep is None:
            return None
        side, target = self._change
        curve, ego_s, ego_right, reach = self._keep[:4]
        fx, fz = -math.sin(yaw_rad), -math.cos(yaw_rad)
        half_l, half_w = v.size.length * 0.5, v.size.width * 0.5
        far = near = math.inf
        ahead = math.inf
        for a in (-half_l, half_l):
            for b in (-half_w, half_w):
                x = v.position.x + a * fx - b * fz
                z = v.position.z + a * fz + b * fx
                s, right = fast_offset(curve, x, z)
                if not math.isfinite(right) or s <= ego_s or s >= reach:
                    return None
                ahead = min(ahead, s - ego_s)
                beyond = side * (fast_offset(target, x, z)[1] if target is not None else right)
                far = min(far, beyond)
                near = min(near, -side * right)
        # Beyond the lane ego settles in, or on the side it leaves, past where ego is now.
        return far - ego_hw, near - ego_hw - max(0.0, -side * ego_right), ahead, side

    def in_ego_lane(self, x: float, z: float, half_width: float) -> bool:
        """``(x, z)`` ahead on the lane ego's road path follows, within ``half_width`` of its centre."""
        curve = self._ego_curve
        if curve is None:
            return False
        s, right = lateral_offset(curve, x, z)
        return 0.0 < s < curve.length and abs(right) <= half_width


_FOLLOW_STEP_M = 2.0
_NO_DRIFT = (0.0, 0.0)


def _lane_turn(curve, s_offset: float, length: float) -> float:
    """Heading change of ``curve`` over ``length`` metres from ``s_offset``, radians."""
    end = min(s_offset + length, curve.length)
    return sum(curve.kappa_at(s) * _FOLLOW_STEP_M
               for s in _frange(s_offset + 0.5 * _FOLLOW_STEP_M, end, _FOLLOW_STEP_M))


def _lane_drift(curve, s_offset: float, lead: ArcPath, cal: AEBCalibration) -> tuple[float, float]:
    """The vehicle's move across its lane, beyond the lane's own turn and lane keep's slack.

    ``(metres right per metre of travel, metres it lasts)``: lane keep's 1.5 s horizon.
    """
    h = _curve_heading(curve, s_offset)
    if h is None or lead.speed < 1e-3:
        return _NO_DRIFT
    # Lane's right-hand normal is (-sin h, cos h) in the curve's atan2(dz, dx) convention.
    across = lead.fwd_x * -math.sin(h) + lead.fwd_z * math.cos(h)
    excused = abs(_lane_turn(curve, s_offset, max(lead.speed * cal.target_follow_heading_lead_s,
                                                  _FOLLOW_STEP_M)))
    excused = math.sin(min(excused + math.radians(cal.lane_keep_heading_slack_deg), math.pi / 2))
    rate = math.copysign(max(abs(across) - excused, 0.0), across)
    return rate, lead.speed * cal.lane_keep_horizon_s


def _follow_points(curve, s_offset: float, pose, kappa_now: float, length: float,
                   ramp: float, merge_s: float | None = None,
                   drift: tuple[float, float] = (0.0, 0.0)) -> list[tuple[float, float]]:
    """World points every 2 m from the vehicle onto its lane at the offset it holds (README).

    ``drift`` (metres right per metre of travel, metres of travel it lasts) carries a move
    across the lane on, as lane keep counts it.
    """
    x0, z0, fx, fz = pose
    rx, rz = -fz, fx
    held0 = _held_offset(curve, s_offset, pose)
    pts = []
    s = 0.0
    while s <= length + _FOLLOW_STEP_M + 1e-9 and s_offset + s <= curve.length:
        lp = _curve_point(curve, s_offset + s)
        lh = _curve_heading(curve, s_offset + s)
        if lp is None or lh is None:
            break
        held = held0 * (1.0 - _smoothstep(s / merge_s)) if merge_s else held0
        held += drift[0] * min(s, drift[1])
        wx = lp[0] - held * math.sin(lh) - x0
        wz = lp[1] + held * math.cos(lh) - z0
        lx, ly = wx * fx + wz * fz, wx * rx + wz * rz
        ax, ay = _arc_xy(kappa_now, s)
        g = _smoothstep(s / ramp) if ramp > 0.0 else 1.0
        px, py = ax + g * (lx - ax), ay + g * (ly - ay)
        pts.append((x0 + px * fx + py * rx, z0 + px * fz + py * rz))
        s += _FOLLOW_STEP_M
    return pts


def _follow_path(curve, arc: ArcPath, length: float, ramp: float,
                 merge_s: float | None = None,
                 drift: tuple[float, float] = (0.0, 0.0)) -> BentPath | None:
    """``arc`` joining its lane from where its body is (README target follow), or None.

    The path is the points themselves, not a curvature profile: integrating one on the
    path's own grid put a 10 m radius prefab turn up to 0.9 m off its lane.
    """
    pose = (arc.start_x, arc.start_z, arc.fwd_x, arc.fwd_z)
    pts = _follow_points(curve, foot_offset(curve, arc.start_x, arc.start_z), pose,
                         arc.curvature, length, ramp, merge_s, drift)
    if len(pts) < 3:
        return None
    path = BentPath(
        start_x=arc.start_x, start_z=arc.start_z, yaw_rad=arc.yaw_rad, speed=arc.speed,
        curvature=arc.curvature, half_width=arc.half_width, horizon=arc.horizon,
        decel=arc.decel, accel=arc.accel, fwd_len=arc.fwd_len, back_len=arc.back_len,
        parallel_margin_scale=arc.parallel_margin_scale,
        kappa_s=(0.0,), kappa_k=(arc.curvature,),
    ).build()
    n, ds = len(path._xs), path._ds
    cum = [0.0]
    for (ax, az), (bx, bz) in zip(pts, pts[1:]):
        cum.append(cum[-1] + math.hypot(bx - ax, bz - az))
    xs, zs = [], []
    j = 0
    for k in range(n):
        d = k * ds
        while j < len(pts) - 2 and cum[j + 1] < d:
            j += 1
        seg = max(cum[j + 1] - cum[j], 1e-9)
        f = (d - cum[j]) / seg
        (ax, az), (bx, bz) = pts[j], pts[j + 1]
        xs.append(ax + f * (bx - ax))
        zs.append(az + f * (bz - az))
    # Bodies point along the motion here, not the §8 mirror: a new consumer, nothing tuned on it.
    body = arc.yaw_rad - math.atan2(-arc.fwd_x, -arc.fwd_z)
    hs: list[float] = []
    h_prev = None
    for k in range(n):
        a, b = max(k - 1, 0), min(k + 1, n - 1)
        if k == 0 or b == a:
            h = math.atan2(-arc.fwd_x, -arc.fwd_z)
        else:
            h = math.atan2(-(xs[b] - xs[a]), -(zs[b] - zs[a]))
        if h_prev is not None:
            h = h_prev + _wrap(h - h_prev)
        h_prev = h
        hs.append(h + body)
    path._xs, path._zs, path._hs = xs, zs, hs
    return path


def _frame(curve, reach: float, x: float, z: float, vel_x: float, vel_z: float, kappa: float = 0.0):
    """``(curve, ego s, ego right, reach, speed across, heading error, accel across)`` on ``curve``."""
    ego_s, ego_right = lateral_offset(curve, x, z)
    h = _curve_heading(curve, ego_s)
    if not math.isfinite(ego_right) or h is None:
        return None
    along = vel_x * math.cos(h) + vel_z * math.sin(h)
    across = vel_x * -math.sin(h) + vel_z * math.cos(h)
    # Curvature is positive to the left, so steering tighter than the lane pulls ego left.
    accel = -(vel_x * vel_x + vel_z * vel_z) * (kappa - curve.kappa_at(ego_s))
    return curve, ego_s, ego_right, reach, across, math.atan2(across, max(along, 1e-6)), accel


def _walk(pts, length: float):
    """Points every ``_GAP_STEP_M`` along ``pts`` up to ``length``, the start included."""
    if not pts:
        return
    yield pts[0]
    done, due = 0.0, _GAP_STEP_M
    for (ax, az), (bx, bz) in zip(pts, pts[1:]):
        hop = math.hypot(bx - ax, bz - az)
        while hop > 1e-9 and due <= done + hop:
            if due > length:
                return
            f = (due - done) / hop
            yield ax + f * (bx - ax), az + f * (bz - az)
            due += _GAP_STEP_M
        done += hop


def _start_key(pts) -> tuple[int, int]:
    """A candidate's start, to the metre, so copies of one lane are tried once."""
    return round(pts[0][0]), round(pts[0][1])


def _reversing(arc: ArcPath) -> bool:
    return arc.fwd_x * -math.sin(arc.yaw_rad) + arc.fwd_z * -math.cos(arc.yaw_rad) < 0.0
