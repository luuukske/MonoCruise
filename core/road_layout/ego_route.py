"""Ego's lane and the route ahead of it, with where that route stops being certain.

The lane choice is ``ego_lane``'s score with hysteresis: the held lane keeps
the truck until another lane scores clearly better. The route follows the lane
forward the same way the lane view does. Wherever another lane leaves the same
tip (a junction, an exit, a temporary divider) the route records how far that
branch drifts from it, so a consumer can stop trusting the route where the
branches part instead of guessing which one the driver takes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from core.road_layout.ego_lane import (
    EgoLaneTrack,
    LineIndex,
    extend_lane,
    join_candidates,
    path_curvature,
    ranked_lanes,
    travel_direction,
)

# A lane change has to beat the held lane by this much score (metres-equivalent).
_HOLD_MARGIN_M = 0.8
_HOLD_MAX_DIST_M = 3.6
# Stacked roads: a lane this far above or below ego is another level.
_HEIGHT_GATE_M = 4.5
_RESET_M = 80.0
_RANKED = 6
_ALT_REACH_M = 160.0
# Lanes this close to ego's (distance and score) could be the one ego is really on.
_SIBLING_DIST_M = 1.0
_SIBLING_COST_M = 1.5
_ROUTE_MAX_M = 250.0
# A sibling lane at least this far off the route at ego that comes within _MERGE_JOIN_M
# of it inside _MERGE_REACH_M is a merge: two lanes becoming one.
_MERGE_APART_M = 1.5
_MERGE_JOIN_M = 0.75
_MERGE_REACH_M = 150.0
_DIV_STEP_M = 5.0


@dataclass(frozen=True)
class LaneRoute:
    """World route ahead of ego along its lane.

    ``pts`` start at ego's foot point on the lane and run along travel. ``offset_m``
    is ego's distance right of the lane centre. ``divergence`` holds ``(s, metres)``:
    how far the widest other branch has left the route by arc length ``s``.
    """

    pts: tuple[tuple[float, float], ...]
    offset_m: float
    heading_err_rad: float
    lane_index: int
    kind: str
    lane_count: int
    divergence: tuple[tuple[float, float], ...]
    reach_m: float
    branch_s: float | None
    # Other routes from ego's foot: branches past a tip, lanes ego could equally be on.
    alternatives: tuple[tuple[tuple[float, float], ...], ...] = ()
    # Arc length where a lane ego could equally be on runs into the route (two lanes
    # becoming one), or None. Paths fade ego's held offset out by here.
    merge_s: float | None = None
    # Ego's offset right of that merging lane: the lane it is really on.
    merge_offset_m: float | None = None
    # Index in ``alternatives`` of that merging lane's route.
    merge_index: int | None = None


class EgoRouter:
    """Holds ego's lane across frames. One instance per consumer of the map."""

    def __init__(self) -> None:
        self._track = EgoLaneTrack()
        self._held: tuple[str, int, int] | None = None
        self._last: tuple[float, float] | None = None
        self._pts_for: object | None = None
        self._pts: tuple = ()

    def reset(self) -> None:
        self._track = EgoLaneTrack()
        self._held = None
        self._last = None

    def step(self, window, x: float, y: float, z: float, fwd_x: float, fwd_z: float) -> LaneRoute | None:
        """Route for ego at ``(x, y, z)`` heading ``(fwd_x, fwd_z)``. None off the map."""
        if self._last is not None and math.hypot(x - self._last[0], z - self._last[1]) > _RESET_M:
            self.reset()
        self._last = (x, z)
        lanes = getattr(window, "lanes", ())
        near = getattr(window, "index", None)
        self._track.note(x, z)
        if not lanes or near is None:
            return None
        if self._pts_for is not window:
            self._pts = tuple(lane.pts for lane in lanes)
            self._pts_for = window
        history = self._track.points()
        dx, dz = travel_direction(history, fwd_x, fwd_z)
        kappa = path_curvature(history)
        ranked = ranked_lanes(self._pts, x, z, dx, dz, near, kappa, limit=_RANKED)
        ranked = [r for r in ranked if _level_ok(lanes[r[0]], x, z, y)]
        if not ranked:
            self._held = None
            return None
        chosen = ranked[0]
        if self._held is not None:
            for index, dist, cost in ranked:
                if _key(lanes[index]) == self._held:
                    if cost <= chosen[2] + _HOLD_MARGIN_M and dist <= _HOLD_MAX_DIST_M:
                        chosen = (index, dist, cost)
                    break
        index = chosen[0]
        self._held = _key(lanes[index])
        siblings = [
            r[0] for r in ranked
            if r[0] != index
            and r[1] <= chosen[1] + _SIBLING_DIST_M
            and r[2] <= chosen[2] + _SIBLING_COST_M
        ]
        return build_route(lanes, near, index, x, z, dx, dz, siblings)


def build_route(
    lanes: Sequence,
    near: LineIndex,
    index: int,
    x: float,
    z: float,
    dx: float,
    dz: float,
    siblings: Sequence[int] = (),
) -> LaneRoute | None:
    """Route along ``lanes[index]`` from the foot of ``(x, z)``, with branch divergence.

    ``siblings`` are lanes ego could equally be on. They count as branches from
    the start, compared at ego's held offset, so two parallel lanes cost nothing.
    """
    main = _assemble(lanes, near, index, x, z, dx, dz)
    if main is None:
        return None
    spans, route, boundaries, (fx, fz, tx, tz) = main
    right = (x - fx) * (-tz) + (z - fz) * tx
    heading_err = math.atan2(dx * tz - dz * tx, dx * tx + dz * tz)
    divergence, branch_s, branches = _divergence(lanes, near, spans, route, boundaries)
    alternatives = [
        a for a in (_clip(_prefix(route, s_at) + list(pts[1:]), _ROUTE_MAX_M) for s_at, pts in branches)
        if len(a) >= 2
    ]
    held = _offset_polyline(route, right)
    merge_s = None
    merge_offset = None
    merge_index = None
    for sib in siblings:
        other = _assemble(lanes, near, sib, x, z, dx, dz)
        if other is None:
            continue
        _sp, sib_route, _b, (sfx, sfz, stx, stz) = other
        sib_right = (x - sfx) * (-stz) + (z - sfz) * stx
        alternatives.append(sib_route)
        joined = _joins_at(route, sib_route)
        if joined is not None:
            # A lane that runs into the route is no branch: held offset fades out by the merge.
            if merge_s is None or joined < merge_s:
                merge_s = joined
                merge_offset = sib_right
                merge_index = len(alternatives) - 1
            continue
        divergence, branch_s = _merge_divergence(
            divergence, branch_s, held, _offset_polyline(sib_route, sib_right),
        )
    lane = lanes[index]
    return LaneRoute(
        route, right, heading_err, index, lane.kind, getattr(lane, "lane_count", 0),
        divergence, _length(route), branch_s, tuple(alternatives),
        merge_s, merge_offset, merge_index,
    )


def _assemble(lanes, near, index, x, z, dx, dz):
    """``(spans, route pts, span boundaries by s, foot and tangent)`` for one lane."""
    spans = extend_lane(lanes, index, x, z, dx, dz, near)
    if not spans:
        return None
    own = lanes[index].pts
    foot = _foot(own, x, z, dx, dz)
    if foot is None:
        return None
    seg, t, sign, fx, fz, tx, tz = foot
    pts: list[tuple[float, float]] = [(fx, fz)]
    pts.extend(_ahead(own, seg, t, sign))
    boundaries: list[tuple[float, int]] = []
    for k in range(1, len(spans)):
        boundaries.append((_length(pts), k))
        chunk = _oriented(spans[k].pts, pts[-1])
        if chunk and math.hypot(chunk[0][0] - pts[-1][0], chunk[0][1] - pts[-1][1]) < 0.5:
            chunk = chunk[1:]
        pts.extend(chunk)
    route = _clip(pts, _ROUTE_MAX_M)
    if len(route) < 2:
        return None
    return spans, route, boundaries, (fx, fz, tx, tz)


def _merge_divergence(divergence, branch_s, held, other_held):
    out = []
    for s, gap in divergence:
        p = _route_point(held, s)
        if p is not None:
            gap = max(gap, _distance_to(other_held, p[0], p[1]))
        if gap > 0.5 and (branch_s is None or s < branch_s):
            branch_s = s
        out.append((s, gap))
    return tuple(out), branch_s


def _offset_polyline(pts, right: float) -> tuple[tuple[float, float], ...]:
    """``pts`` moved ``right`` metres to the right of travel."""
    if abs(right) < 1e-6 or len(pts) < 2:
        return tuple(pts)
    out = []
    for i in range(len(pts)):
        a = pts[max(i - 1, 0)]
        b = pts[min(i + 1, len(pts) - 1)]
        hx, hz = b[0] - a[0], b[1] - a[1]
        n = math.hypot(hx, hz)
        if n < 1e-9:
            out.append(pts[i])
            continue
        out.append((pts[i][0] - right * hz / n, pts[i][1] + right * hx / n))
    return tuple(out)


def _divergence(lanes, near, spans, route, boundaries):
    """``(s, metres)`` the widest branch leaving any span tip has drifted from the route."""
    seen = {span.index for span in spans}
    branches: list[tuple[float, tuple[tuple[float, float], ...]]] = []
    tips = [(s_at, spans[k - 1].index, spans[k].index) for s_at, k in boundaries]
    for s_at, prev_index, next_index in tips:
        tip = _route_point(route, s_at)
        tip_dir = _route_dir(route, s_at)
        if tip is None or tip_dir is None:
            continue
        for _cost, cand, _end in join_candidates(near, tip, tip_dir, {prev_index, next_index}):
            if cand in seen:
                continue
            alt = extend_lane(lanes, cand, tip[0], tip[1], tip_dir[0], tip_dir[1], near)
            pts = _join(alt, tip)
            if len(pts) >= 2:
                branches.append((s_at, pts))
    total = _length(route)
    out: list[tuple[float, float]] = []
    first: float | None = None
    s = 0.0
    while s <= total + 1e-6:
        widest = 0.0
        p = _route_point(route, s)
        for s_at, pts in branches:
            if s < s_at or p is None or s > s_at + _ALT_REACH_M:
                continue
            gap = _distance_to(pts, p[0], p[1])
            if gap > widest:
                widest = gap
        if widest > 0.5 and first is None:
            first = s
        out.append((s, widest))
        s += _DIV_STEP_M
    return tuple(out), first, branches


def _oriented(chunk, tail) -> list[tuple[float, float]]:
    """A span's points running away from ``tail``: whole joined lines keep stored order."""
    out = list(chunk)
    if len(out) >= 2 and math.hypot(out[-1][0] - tail[0], out[-1][1] - tail[1]) < math.hypot(
            out[0][0] - tail[0], out[0][1] - tail[1]):
        out.reverse()
    return out


def _join(spans, tip) -> tuple[tuple[float, float], ...]:
    pts: list[tuple[float, float]] = [tip]
    for span in spans:
        chunk = _oriented(span.pts, pts[-1])
        if not chunk:
            continue
        pts.extend(chunk)
        if _length(pts) > _ALT_REACH_M:
            break
    return tuple(pts)


def _key(lane) -> tuple[str, int, int]:
    return (lane.kind, int(getattr(lane, "uid", 0)), int(getattr(lane, "lane_index", -1)))


def _level_ok(lane, x: float, z: float, y: float) -> bool:
    ys = getattr(lane, "ys", None)
    if not ys:
        return True
    best = None
    for (px, pz), py in zip(lane.pts, ys):
        d = (px - x) ** 2 + (pz - z) ** 2
        if best is None or d < best[0]:
            best = (d, py)
    return best is None or abs(best[1] - y) <= _HEIGHT_GATE_M


def _foot(pts, x, z, dx, dz):
    """Closest point on ``pts``: segment, param, travel sign, foot xz and unit travel tangent."""
    best = None
    for i in range(len(pts) - 1):
        ax, az = pts[i]
        bx, bz = pts[i + 1]
        sx, sz = bx - ax, bz - az
        length2 = sx * sx + sz * sz
        if length2 < 1e-12:
            continue
        t = max(0.0, min(1.0, ((x - ax) * sx + (z - az) * sz) / length2))
        px, pz = ax + t * sx, az + t * sz
        d = (x - px) ** 2 + (z - pz) ** 2
        if best is None or d < best[0]:
            best = (d, i, t, px, pz, sx, sz)
    if best is None:
        return None
    _d, i, t, px, pz, sx, sz = best
    n = math.hypot(sx, sz)
    tx, tz = sx / n, sz / n
    sign = 1
    if tx * dx + tz * dz < 0.0:
        sign = -1
        tx, tz = -tx, -tz
    return i, t, sign, px, pz, tx, tz


def _ahead(pts, seg, t, sign):
    if sign > 0:
        idx = range(seg + 1, len(pts))
        if t >= 0.999:
            idx = range(seg + 2, len(pts))
    else:
        idx = range(seg, -1, -1)
        if t <= 0.001:
            idx = range(seg - 1, -1, -1)
    return [pts[i] for i in idx]


def _prefix(pts, s_end: float) -> list[tuple[float, float]]:
    """Points of ``pts`` up to arc length ``s_end``, ending exactly there."""
    out = [pts[0]]
    walked = 0.0
    for i in range(1, len(pts)):
        hop = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        if walked + hop >= s_end:
            p = _route_point(pts, s_end)
            if p is not None:
                out.append(p)
            return out
        walked += hop
        out.append(pts[i])
    return out


def _clip(pts, limit: float) -> tuple[tuple[float, float], ...]:
    out = [pts[0]] if pts else []
    walked = 0.0
    for i in range(1, len(pts)):
        hop = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        if walked + hop >= limit:
            f = (limit - walked) / hop if hop > 1e-9 else 0.0
            out.append((pts[i - 1][0] + f * (pts[i][0] - pts[i - 1][0]),
                        pts[i - 1][1] + f * (pts[i][1] - pts[i - 1][1])))
            return tuple(out)
        walked += hop
        out.append(pts[i])
    return tuple(out)


def _length(pts) -> float:
    total = 0.0
    for i in range(1, len(pts)):
        total += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    return total


def _joins_at(route, other) -> float | None:
    """Route arc length where ``other``, starting a lane or so beside it, runs into it."""
    if len(other) < 2 or _distance_to(route, other[0][0], other[0][1]) < _MERGE_APART_M:
        return None
    walked = 0.0
    for i in range(1, len(other)):
        walked += math.hypot(other[i][0] - other[i - 1][0], other[i][1] - other[i - 1][1])
        if walked > _MERGE_REACH_M:
            return None
        if _distance_to(route, other[i][0], other[i][1]) <= _MERGE_JOIN_M:
            return _arc_length_at(route, other[i][0], other[i][1])
    return None


def _arc_length_at(pts, x: float, z: float) -> float:
    best_d, best_s, walked = math.inf, 0.0, 0.0
    for i in range(len(pts) - 1):
        ax, az = pts[i]
        sx, sz = pts[i + 1][0] - ax, pts[i + 1][1] - az
        length2 = sx * sx + sz * sz
        hop = math.sqrt(length2)
        t = 0.0 if length2 < 1e-12 else max(0.0, min(1.0, ((x - ax) * sx + (z - az) * sz) / length2))
        d = (x - ax - t * sx) ** 2 + (z - az - t * sz) ** 2
        if d < best_d:
            best_d, best_s = d, walked + t * hop
        walked += hop
    return best_s


def _route_point(pts, s: float):
    if not pts:
        return None
    left = s
    for i in range(1, len(pts)):
        hop = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        if hop >= left:
            f = left / hop if hop > 1e-9 else 0.0
            return (pts[i - 1][0] + f * (pts[i][0] - pts[i - 1][0]),
                    pts[i - 1][1] + f * (pts[i][1] - pts[i - 1][1]))
        left -= hop
    return pts[-1]


def _route_dir(pts, s: float):
    left = s
    for i in range(1, len(pts)):
        hop = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        if hop >= left or i == len(pts) - 1:
            if hop < 1e-9:
                continue
            return ((pts[i][0] - pts[i - 1][0]) / hop, (pts[i][1] - pts[i - 1][1]) / hop)
        left -= hop
    return None


def _distance_to(pts, x: float, z: float) -> float:
    best = math.inf
    for i in range(len(pts) - 1):
        ax, az = pts[i]
        bx, bz = pts[i + 1]
        sx, sz = bx - ax, bz - az
        length2 = sx * sx + sz * sz
        if length2 < 1e-12:
            d = math.hypot(x - ax, z - az)
        else:
            t = max(0.0, min(1.0, ((x - ax) * sx + (z - az) * sz) / length2))
            d = math.hypot(x - (ax + t * sx), z - (az + t * sz))
        if d < best:
            best = d
    return best
