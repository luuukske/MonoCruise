"""Choose the lane ego is in, then follow it forward.

Distance to the centerline is the main term. Direction still counts, so a
sharp branch does not win by passing only a little closer. The highlight runs
forward until just before the sixth prefab, or 200 m, whichever comes first.
A lane that ends beside another continues on that lane.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Metres of offset one radian of mismatch is worth. Distance is the larger term.
_NOW_M_PER_RAD = 2.5
_AHEAD_M_PER_RAD = 1.5
_JOIN_M = 4.0
_MERGE_M = 6.0
_MERGE_AHEAD_M = 8.0
_CHAIN_M_PER_RAD = 2.0
_CHAIN_MAX = 64
_REACH_M = 200.0
_PREFAB_KEEP = 5
_REVERSE_RAD = math.radians(150.0)
_MERGE_ANG = math.radians(35.0)
_AHEAD_M = 16.0
_OFFSET_LIMIT_M = 8.0
_MIN_STEP_M = 1.5
_PATH_M = 12.0
_DIR_M = 5.0
_MIN_CHORD_M = 2.0
_MAX_KAPPA = 0.15
_RESET_M = 80.0
_CELL_M = 80.0


class LineIndex:
    """Cells of lane lines so a join only looks beside its tip."""

    def __init__(self, lines: Sequence[object]) -> None:
        self.lines = lines
        self._grid: dict[tuple[int, int], list[int]] = {}
        for index, line in enumerate(lines):
            self._cover(index, line, _pts(line))

    def around(self, x: float, z: float, pad: float) -> list[int]:
        x0, z0 = _cell(x - pad, z - pad)
        x1, z1 = _cell(x + pad, z + pad)
        found: list[int] = []
        seen: set[int] = set()
        for ix in range(x0, x1 + 1):
            for iz in range(z0, z1 + 1):
                for index in self._grid.get((ix, iz), ()):
                    if index in seen:
                        continue
                    seen.add(index)
                    found.append(index)
        return found

    def _cover(self, index: int, line: object, pts: Sequence[tuple[float, float]]) -> None:
        min_x = getattr(line, "min_x", None)
        if min_x is not None:
            self._cover_box(index, min_x, line.max_x, line.min_z, line.max_z)
            return
        seen: set[tuple[int, int]] = set()
        for p in pts:
            key = _cell(p[0], p[1])
            if key in seen:
                continue
            seen.add(key)
            self._grid.setdefault(key, []).append(index)

    def _cover_box(self, index: int, min_x: float, max_x: float, min_z: float, max_z: float) -> None:
        x0, z0 = _cell(min_x, min_z)
        x1, z1 = _cell(max_x, max_z)
        if x1 - x0 > 40 or z1 - z0 > 40:
            return
        for ix in range(x0, x1 + 1):
            for iz in range(z0, z1 + 1):
                self._grid.setdefault((ix, iz), []).append(index)


def _cell(x: float, z: float) -> tuple[int, int]:
    return math.floor(x / _CELL_M), math.floor(z / _CELL_M)


@dataclass(frozen=True)
class LaneSpan:
    """One piece of the highlighted lane. ``pts`` may start where a merge joins."""

    index: int
    pts: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class EgoLaneChoice:
    """The lane the truck is in. ``angle_rad`` is the heading error beside it."""

    index: int
    distance_m: float
    angle_rad: float


class EgoLaneTrack:
    """Recent positions used to line the truck's heading up with a lane."""

    def __init__(self) -> None:
        self._frozen: list[tuple[float, float]] = []
        self._live: tuple[float, float] | None = None

    def note(self, x: float, z: float) -> None:
        """Record a pose. Samples stay about a step apart over the last stretch."""
        point = (float(x), float(z))
        self._live = point
        if not self._frozen:
            self._frozen.append(point)
            return
        anchor = self._frozen[-1]
        gap = math.hypot(point[0] - anchor[0], point[1] - anchor[1])
        if gap > _RESET_M:
            self._frozen = [point]
            return
        if gap < _MIN_STEP_M:
            return
        self._frozen.append(point)
        while len(self._frozen) > 2 and _length(self.points()) > _PATH_M:
            del self._frozen[0]

    def points(self) -> list[tuple[float, float]]:
        if not self._frozen:
            return []
        if self._live is None or self._live == self._frozen[-1]:
            return list(self._frozen)
        return self._frozen + [self._live]

    def choose(
        self,
        lines: Sequence[Sequence[tuple[float, float]]],
        x: float,
        z: float,
        yaw_x: float,
        yaw_z: float,
        boxes: Sequence[tuple[float, float, float, float]] | None = None,
        near: "LineIndex | None" = None,
    ) -> EgoLaneChoice | None:
        """Lane under ``(x, z)``. ``yaw`` is the truck's ahead vector in XZ."""
        self.note(x, z)
        pts = self.points()
        dx, dz = travel_direction(pts, yaw_x, yaw_z)
        return choose_lane(lines, x, z, dx, dz, path_curvature(pts), boxes, near)


def travel_direction(
    history: Sequence[tuple[float, float]],
    yaw_x: float,
    yaw_z: float,
) -> tuple[float, float]:
    """Unit XZ direction: the recent path lined up with the truck's heading."""
    yaw = _unit(yaw_x, yaw_z)
    path = _chord(_tail(history, _DIR_M))
    if yaw is None and path is None:
        return 0.0, -1.0
    if path is None:
        return yaw if yaw is not None else (0.0, -1.0)
    if yaw is None:
        return path
    if path[0] * yaw[0] + path[1] * yaw[1] < 0.0:
        return path
    both = _unit(path[0] + yaw[0], path[1] + yaw[1])
    return both if both is not None else yaw


def path_curvature(history: Sequence[tuple[float, float]]) -> float:
    """Signed turn rate (1/m) of the recent path. Zero until it has a bend."""
    if len(history) < 3:
        return 0.0
    a = history[0]
    b = history[len(history) // 2]
    c = history[-1]
    abx, abz = b[0] - a[0], b[1] - a[1]
    bcx, bcz = c[0] - b[0], c[1] - b[1]
    lab = math.hypot(abx, abz)
    lbc = math.hypot(bcx, bcz)
    if lab < 1.0 or lbc < 1.0:
        return 0.0
    turn = math.atan2(abx * bcz - abz * bcx, abx * bcx + abz * bcz)
    # Each chord points along the middle of its leg, so the baseline is half the span.
    kappa = turn / (0.5 * (lab + lbc))
    return max(-_MAX_KAPPA, min(_MAX_KAPPA, kappa))


def choose_lane(
    lines: Sequence[Sequence[tuple[float, float]]],
    x: float,
    z: float,
    dir_x: float,
    dir_z: float,
    curvature: float = 0.0,
    boxes: Sequence[tuple[float, float, float, float]] | None = None,
    near: "LineIndex | None" = None,
) -> EgoLaneChoice | None:
    """Line that best continues ``(dir_x, dir_z)``. Points are ``(x, z)``."""
    direction = _unit(dir_x, dir_z)
    if direction is None:
        return None
    dx, dz = direction
    if boxes is not None and len(boxes) != len(lines):
        boxes = None
    best: tuple[float, EgoLaneChoice] | None = None
    order = range(len(lines)) if near is None else near.around(x, z, _OFFSET_LIMIT_M)
    for index in order:
        pts = lines[index]
        if near is None and boxes is not None and _outside(boxes[index], x, z):
            continue
        scored = _score_line(pts, index, x, z, dx, dz, curvature)
        if scored is None:
            continue
        cost, choice = scored
        if best is None or cost < best[0]:
            best = (cost, choice)
    return None if best is None else best[1]


def ranked_lanes(
    lines: Sequence[Sequence[tuple[float, float]]],
    x: float,
    z: float,
    dir_x: float,
    dir_z: float,
    near: "LineIndex | None" = None,
    curvature: float = 0.0,
    limit: int = 2,
) -> list[tuple[int, float, float]]:
    """Lowest-cost lanes as ``(index, distance_m, cost)``."""
    direction = _unit(dir_x, dir_z)
    if direction is None or limit <= 0:
        return []
    dx, dz = direction
    order = range(len(lines)) if near is None else near.around(x, z, _OFFSET_LIMIT_M)
    found: list[tuple[float, int, float]] = []
    for index in order:
        scored = _score_line(lines[index], index, x, z, dx, dz, curvature)
        if scored is None:
            continue
        cost, choice = scored
        found.append((cost, choice.index, choice.distance_m))
    found.sort()
    return [(index, dist, cost) for cost, index, dist in found[:limit]]


def extend_lane(
    lines: Sequence[object],
    index: int,
    x: float,
    z: float,
    dir_x: float,
    dir_z: float,
    near: "LineIndex | None" = None,
) -> tuple[LaneSpan, ...]:
    """The lane forward from ``index``, at most 200 m and five prefabs.

    Each line exposes ``pts`` as ``(x, z)`` pairs and an optional ``prefab_id``.
    Curves that share a ``prefab_id`` are one junction. The sixth prefab is not
    entered. A lane that ends beside the one it joins continues on that lane.
    """
    if index < 0 or index >= len(lines):
        return ()
    direction = _unit(dir_x, dir_z)
    own = _pts(lines[index])
    if direction is None or len(own) < 2:
        return (LaneSpan(index, _xy(own)),)
    dx, dz = direction
    if near is None:
        near = LineIndex(lines)
    spans = [LaneSpan(index, _xy(own))]
    seen = {index}
    prefabs: list[int] = []
    pid = _prefab_id(lines[index])
    if pid is not None:
        prefabs.append(pid)
    tip = _forward_tip(own, float(x), float(z), dx, dz)
    dist, seg, param = _closest(own, float(x), float(z))
    sign = 1 if tip == len(own) - 1 else -1
    ahead = 0.0 if dist is None else _ahead_length(own, seg, param, sign)
    if ahead >= _REACH_M:
        spans[0] = LaneSpan(index, _clip_span(own, seg, param, sign, _REACH_M))
        return tuple(spans)
    remain = _REACH_M - ahead
    for _ in range(_CHAIN_MAX):
        pts = _pts(lines[spans[-1].index])
        if len(pts) < 2:
            break
        tip_pt = (pts[tip][0], pts[tip][1])
        tip_dir = _tip_dir(pts, tip)
        if tip_dir is None:
            break
        nxt = _best_join(near, tip_pt, tip_dir, seen, prefabs)
        if nxt is not None and not nxt[3]:
            _cost, j_index, j_pid, _second, joined_end = nxt
            joined = _pts(lines[j_index])
            taken, remain = _take(joined, joined_end, remain)
            if len(taken) >= 2:
                spans.append(LaneSpan(j_index, taken))
            seen.add(j_index)
            if j_pid is not None and j_pid not in prefabs:
                prefabs.append(j_pid)
            if remain <= 0.0:
                break
            tip = 0 if joined_end == len(joined) - 1 else len(joined) - 1
            continue
        merged = _merge_onto(near, tip_pt, tip_dir, seen, prefabs)
        if merged is None:
            break
        j_index, j_pid, far_end, slice_pts = merged
        taken, remain = _take(slice_pts, 0, remain)
        if len(taken) >= 2:
            spans.append(LaneSpan(j_index, taken))
        seen.add(j_index)
        if j_pid is not None and j_pid not in prefabs:
            prefabs.append(j_pid)
        if remain <= 0.0:
            break
        tip = far_end
    return tuple(spans)


def join_candidates(
    near: "LineIndex",
    tip: tuple[float, float],
    tip_dir: tuple[float, float],
    seen: set[int],
) -> list[tuple[float, int, int]]:
    """Lines that start where ``tip`` ends, as ``(cost, index, joined_end)``, cheapest first."""
    found: list[tuple[float, int, int]] = []
    tx, tz = tip
    for index in near.around(tx, tz, _JOIN_M):
        if index in seen:
            continue
        pts = _pts(near.lines[index])
        if len(pts) < 2:
            continue
        hit: tuple[float, int] | None = None
        for end in (0, len(pts) - 1):
            ex, ez = pts[end][0], pts[end][1]
            dist = math.hypot(ex - tx, ez - tz)
            if dist > _JOIN_M:
                continue
            enter = _leave_dir(pts, end)
            if enter is None:
                continue
            ang = _angle(tip_dir[0], tip_dir[1], enter[0], enter[1])
            if ang > _REVERSE_RAD:
                continue
            cost = dist + _CHAIN_M_PER_RAD * ang
            if hit is None or cost < hit[0]:
                hit = (cost, end)
        if hit is not None:
            found.append((hit[0], index, hit[1]))
    found.sort()
    return found


def _best_join(
    near: "LineIndex",
    tip: tuple[float, float],
    tip_dir: tuple[float, float],
    seen: set[int],
    prefabs: list[int],
) -> tuple[float, int, int | None, bool, int] | None:
    found = join_candidates(near, tip, tip_dir, seen)
    if not found:
        return None
    cost, index, end = found[0]
    pid = _prefab_id(near.lines[index])
    return cost, index, pid, _stops(pid, prefabs), end


def _stops(pid: int | None, prefabs: list[int]) -> bool:
    """True when this prefab would be the sixth one on the lane."""
    if pid is None or pid in prefabs:
        return False
    return len(prefabs) >= _PREFAB_KEEP


def _take(
    pts: Sequence[tuple[float, float]],
    end: int,
    remain: float,
) -> tuple[tuple[tuple[float, float], ...], float]:
    """Points from ``end`` along the line, limited to ``remain`` metres."""
    if remain <= 0.0 or len(pts) < 2:
        return (), 0.0
    if end == 0:
        seg, param, sign = 0, 0.0, 1
    else:
        seg, param, sign = len(pts) - 2, 1.0, -1
    length = _ahead_length(pts, seg, param, sign)
    if length <= remain:
        return _xy(pts), remain - length
    return _clip_span(pts, seg, param, sign, remain), 0.0


def _clip_span(
    pts: Sequence[tuple[float, float]],
    seg: int,
    t: float,
    sign: int,
    meters: float,
) -> tuple[tuple[float, float], ...]:
    ax, az = pts[seg][0], pts[seg][1]
    bx, bz = pts[seg + 1][0], pts[seg + 1][1]
    foot = (ax + t * (bx - ax), az + t * (bz - az))
    out = [foot]
    left = meters
    if sign > 0:
        nxt_i = seg + 1
        step = 1
        stop = len(pts)
    else:
        nxt_i = seg
        step = -1
        stop = -1
    prev = foot
    while nxt_i != stop:
        nxt = (pts[nxt_i][0], pts[nxt_i][1])
        hop = math.hypot(nxt[0] - prev[0], nxt[1] - prev[1])
        if hop >= left:
            if hop > 1e-8:
                frac = left / hop
                out.append((prev[0] + frac * (nxt[0] - prev[0]), prev[1] + frac * (nxt[1] - prev[1])))
            break
        left -= hop
        if hop > 1e-6:
            out.append(nxt)
        prev = nxt
        nxt_i += step
    return tuple(out)


def _merge_onto(
    near: "LineIndex",
    tip: tuple[float, float],
    tip_dir: tuple[float, float],
    seen: set[int],
    prefabs: list[int],
) -> tuple[int, int | None, int, tuple[tuple[float, float], ...]] | None:
    """Lane whose body the tip runs into, clipped to the part still ahead."""
    best = None
    tx, tz = tip
    for index in near.around(tx, tz, _MERGE_M):
        if index in seen:
            continue
        line = near.lines[index]
        pts = _pts(line)
        if len(pts) < 2:
            continue
        pid = _prefab_id(line)
        dist, seg, t = _closest(pts, tx, tz)
        if dist is None or dist > _MERGE_M:
            continue
        if _stops(pid, prefabs):
            continue
        sense = _segment_sense(pts, seg, tip_dir[0], tip_dir[1])
        tangent, sign = sense
        if tangent is None or sign is None:
            continue
        ang = _angle(tip_dir[0], tip_dir[1], tangent[0], tangent[1])
        if ang > _MERGE_ANG:
            continue
        if _ahead_length(pts, seg, t, sign) < _MERGE_AHEAD_M:
            continue
        cost = dist + _CHAIN_M_PER_RAD * ang
        if best is None or cost < best[0]:
            far = len(pts) - 1 if sign > 0 else 0
            best = (cost, index, pid, far, _slice_forward(pts, seg, t, sign))
    if best is None:
        return None
    _cost, index, pid, far, slice_pts = best
    return index, pid, far, slice_pts


def _ahead_length(pts: Sequence[tuple[float, float]], seg: int, t: float, sign: int) -> float:
    ax, az = pts[seg][0], pts[seg][1]
    bx, bz = pts[seg + 1][0], pts[seg + 1][1]
    seg_len = math.hypot(bx - ax, bz - az)
    if sign > 0:
        left = (1.0 - t) * seg_len
        i = seg + 1
        while i + 1 < len(pts):
            left += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            i += 1
        return left
    left = t * seg_len
    i = seg
    while i > 0:
        left += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        i -= 1
    return left


def _slice_forward(
    pts: Sequence[tuple[float, float]],
    seg: int,
    t: float,
    sign: int,
) -> tuple[tuple[float, float], ...]:
    ax, az = pts[seg][0], pts[seg][1]
    bx, bz = pts[seg + 1][0], pts[seg + 1][1]
    foot = (ax + t * (bx - ax), az + t * (bz - az))
    out = [foot]
    if sign > 0:
        start = seg + 1 if t < 0.999 else seg + 2
        for i in range(start, len(pts)):
            out.append((pts[i][0], pts[i][1]))
    else:
        start = seg if t > 0.001 else seg - 1
        for i in range(start, -1, -1):
            out.append((pts[i][0], pts[i][1]))
    return tuple(out)


def _xy(pts: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    return tuple((p[0], p[1]) for p in pts)


def _pts(line: object) -> Sequence[tuple[float, float]]:
    pts = getattr(line, "pts", None)
    if pts is None:
        return ()
    return pts


def _prefab_id(line: object) -> int | None:
    pid = getattr(line, "prefab_id", None)
    return pid if isinstance(pid, int) else None


def _forward_tip(
    pts: Sequence[tuple[float, float]],
    x: float,
    z: float,
    dx: float,
    dz: float,
) -> int:
    s0 = (pts[0][0] - x) * dx + (pts[0][1] - z) * dz
    last = len(pts) - 1
    s1 = (pts[last][0] - x) * dx + (pts[last][1] - z) * dz
    return 0 if s0 > s1 else last


def _tip_dir(pts: Sequence[tuple[float, float]], tip: int) -> tuple[float, float] | None:
    if tip == 0:
        return _unit(pts[0][0] - pts[1][0], pts[0][1] - pts[1][1])
    last = len(pts) - 1
    return _unit(pts[last][0] - pts[last - 1][0], pts[last][1] - pts[last - 1][1])


def _leave_dir(pts: Sequence[tuple[float, float]], end: int) -> tuple[float, float] | None:
    if end == 0:
        return _unit(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    last = len(pts) - 1
    return _unit(pts[last - 1][0] - pts[last][0], pts[last - 1][1] - pts[last][1])


def _score_line(
    pts: Sequence[tuple[float, float]],
    index: int,
    x: float,
    z: float,
    dx: float,
    dz: float,
    curvature: float,
) -> tuple[float, EgoLaneChoice] | None:
    if len(pts) < 2:
        return None
    dist, seg, t = _closest(pts, x, z)
    if dist is None or dist > _OFFSET_LIMIT_M:
        return None
    tangent, sign = _segment_sense(pts, seg, dx, dz)
    if tangent is None or sign is None:
        return None
    ang = _angle(dx, dz, tangent[0], tangent[1])
    ahead, walked = _ahead_dir(pts, seg, t, sign, _AHEAD_M, tangent)
    ref = _rotate(dx, dz, curvature * walked)
    ahead_ang = _angle(ref[0], ref[1], ahead[0], ahead[1])
    cost = dist + _NOW_M_PER_RAD * ang + _AHEAD_M_PER_RAD * ahead_ang
    return cost, EgoLaneChoice(index, dist, ang)


def _closest(
    pts: Sequence[tuple[float, float]],
    x: float,
    z: float,
) -> tuple[float | None, int, float]:
    best_d: float | None = None
    best_i = 0
    best_t = 0.0
    for i in range(len(pts) - 1):
        ax, az = pts[i][0], pts[i][1]
        bx, bz = pts[i + 1][0], pts[i + 1][1]
        sx = bx - ax
        sz = bz - az
        length2 = sx * sx + sz * sz
        if length2 < 1e-12:
            d = math.hypot(x - ax, z - az)
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((x - ax) * sx + (z - az) * sz) / length2))
            d = math.hypot(x - (ax + t * sx), z - (az + t * sz))
        if best_d is None or d < best_d:
            best_d = d
            best_i = i
            best_t = t
    return best_d, best_i, best_t


def _segment_sense(
    pts: Sequence[tuple[float, float]],
    index: int,
    dx: float,
    dz: float,
) -> tuple[tuple[float, float] | None, int | None]:
    ax, az = pts[index][0], pts[index][1]
    bx, bz = pts[index + 1][0], pts[index + 1][1]
    tangent = _unit(bx - ax, bz - az)
    if tangent is None:
        return None, None
    if tangent[0] * dx + tangent[1] * dz < 0.0:
        return (-tangent[0], -tangent[1]), -1
    return tangent, 1


def _ahead_dir(
    pts: Sequence[tuple[float, float]],
    index: int,
    t: float,
    sign: int,
    distance: float,
    fallback: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    ax, az = pts[index][0], pts[index][1]
    bx, bz = pts[index + 1][0], pts[index + 1][1]
    prev = (ax + t * (bx - ax), az + t * (bz - az))
    cursor = index + 1 if sign > 0 else index
    step = 1 if sign > 0 else -1
    stop = len(pts) if sign > 0 else -1
    remain = distance
    tangent = fallback
    walked = 0.0
    while cursor != stop:
        vx, vz = pts[cursor][0], pts[cursor][1]
        sx = vx - prev[0]
        sz = vz - prev[1]
        seg = math.hypot(sx, sz)
        if seg > 1e-8:
            tangent = (sx / seg, sz / seg)
            if seg >= remain:
                return tangent, walked + remain
            remain -= seg
            walked += seg
        prev = (vx, vz)
        cursor += step
    return tangent, walked


def _outside(box: tuple[float, float, float, float], x: float, z: float) -> bool:
    min_x, max_x, min_z, max_z = box
    limit = _OFFSET_LIMIT_M
    return x < min_x - limit or x > max_x + limit or z < min_z - limit or z > max_z + limit


def _tail(pts: Sequence[tuple[float, float]], length_m: float) -> list[tuple[float, float]]:
    if len(pts) < 2:
        return list(pts)
    acc = 0.0
    i = len(pts) - 1
    while i > 0:
        acc += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        if acc >= length_m:
            return list(pts[i - 1:])
        i -= 1
    return list(pts)


def _chord(pts: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    if len(pts) < 2:
        return None
    dx = pts[-1][0] - pts[0][0]
    dz = pts[-1][1] - pts[0][1]
    if math.hypot(dx, dz) < _MIN_CHORD_M:
        return None
    return _unit(dx, dz)


def _length(pts: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(1, len(pts)):
        total += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    return total


def _rotate(dx: float, dz: float, phi: float) -> tuple[float, float]:
    cos = math.cos(phi)
    sin = math.sin(phi)
    return dx * cos - dz * sin, dx * sin + dz * cos


def _angle(ax: float, az: float, bx: float, bz: float) -> float:
    return abs(math.atan2(ax * bz - az * bx, ax * bx + az * bz))


def _unit(x: float, z: float) -> tuple[float, float] | None:
    n = math.hypot(x, z)
    if n < 1e-8:
        return None
    return x / n, z / n
