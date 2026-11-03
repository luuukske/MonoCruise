"""Traffic that makes moving over a lane unsafe right now: beside ego's rig, closing from behind, or oncoming.

Pure logic. Toward an unsafe side the lane departure detector ignores the driver's intent (blinker, steering), and
LDW/LKA acting there shows as an alert. See core/lane_departure/README.md, "Side traffic".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.road_layout.ego_route import route_behind

from . import edges as edge_kinds

BESIDE = "beside"
# Catching us from behind (we are being overtaken), being caught up by us (we overtake it), or coming the other way.
BEHIND = "behind"
AHEAD = "ahead"
ONCOMING = "oncoming"
LEFT = -1
RIGHT = 1
_SIDES = (LEFT, RIGHT)


@dataclass(frozen=True)
class SideTuning:
    """Placeholders set by reasoning, like the rest of LDW; tune them on drives."""

    min_speed_kmh: float = 30.0
    # A body ahead of or behind the rig is one lane over while its centre sits this far off ego's lane centre.
    near_m: float = 2.6
    far_m: float = 7.0
    beside_front_m: float = 2.0
    beside_rear_m: float = 1.0
    # A body leaving the beside zone this soon, at its speed relative to the rig, no longer counts.
    exit_s: float = 0.4
    # Time lookahead: a body that reaches the rig within this long, at today's closing speed.
    same_way_s: float = 3.0
    oncoming_s: float = 2.5
    reach_m: float = 200.0
    parallel_cos: float = 0.5
    ahead_route_m: float = 240.0
    behind_route_m: float = 120.0
    behind_refresh_m: float = 10.0
    locate_slack_m: float = 1.0
    hold_s: float = 0.3


@dataclass(frozen=True)
class Hit:
    """The vehicle that makes a side unsafe. ``gap_m`` is 0 beside the rig, ``time_s`` until it reaches the rig."""

    kind: str
    vid: int
    tmp: bool
    gap_m: float
    lateral_m: float
    closing_ms: float
    time_s: float
    lane: str
    length_m: float


@dataclass(frozen=True)
class SideState:
    left: Hit | None = None
    right: Hit | None = None
    reason: str = "off"

    def at(self, side: int) -> Hit | None:
        return self.left if side == LEFT else self.right


@dataclass(frozen=True)
class Ego:
    """Ego's tractor at the radar pose, and its trailers in the tractor's frame (``TrailerBody``)."""

    x: float
    z: float
    yaw_rad: float
    speed_ms: float
    half_width_m: float
    front_m: float
    rear_m: float
    trailers: tuple = ()


def _forward(yaw_rad: float) -> tuple[float, float]:
    return -math.sin(yaw_rad), -math.cos(yaw_rad)


def box(x: float, z: float, yaw_rad: float, ahead_m: float, behind_m: float, half_width_m: float):
    """Corners front right, front left, back left, back right, as ``Vehicle.get_corners``."""
    fx, fz = _forward(yaw_rad)
    rx, rz = -fz, fx
    return (
        (x + ahead_m * fx + half_width_m * rx, z + ahead_m * fz + half_width_m * rz),
        (x + ahead_m * fx - half_width_m * rx, z + ahead_m * fz - half_width_m * rz),
        (x - behind_m * fx - half_width_m * rx, z - behind_m * fz - half_width_m * rz),
        (x - behind_m * fx + half_width_m * rx, z - behind_m * fz + half_width_m * rz),
    )


def rig_boxes(ego: Ego) -> list:
    """Ego's tractor and each trailer as corner boxes in the world."""
    boxes = [box(ego.x, ego.z, ego.yaw_rad, ego.front_m, ego.rear_m, ego.half_width_m)]
    fx, fz = _forward(ego.yaw_rad)
    rx, rz = -fz, fx
    for tr in ego.trailers:
        cx = ego.x + tr.ahead_m * fx + tr.right_m * rx
        cz = ego.z + tr.ahead_m * fz + tr.right_m * rz
        hl = tr.half_length_m
        boxes.append(box(cx, cz, ego.yaw_rad + tr.yaw_rel_rad, hl, hl, tr.half_width_m))
    return boxes


class LaneFrame:
    """A polyline along travel; ``locate`` gives a point's arc length and metres right of it, None past its ends."""

    def __init__(self, pts) -> None:
        self.pts = tuple(pts)
        self._segs: list[tuple[float, float, float, float, float, float]] = []
        walked = 0.0
        for (ax, az), (bx, bz) in zip(self.pts, self.pts[1:]):
            hop = math.hypot(bx - ax, bz - az)
            if hop < 1e-6:
                continue
            self._segs.append((ax, az, (bx - ax) / hop, (bz - az) / hop, hop, walked))
            walked += hop
        self.length_m = walked

    def locate(self, x: float, z: float, slack: float) -> tuple[float, float, float, float] | None:
        best = None
        for i, (ax, az, tx, tz, hop, _s0) in enumerate(self._segs):
            px, pz = x - ax, z - az
            along = px * tx + pz * tz
            u = hop if along > hop else 0.0 if along < 0.0 else along
            ex, ez = px - u * tx, pz - u * tz
            d2 = ex * ex + ez * ez
            if best is None or d2 < best[0]:
                best = (d2, i, along, u)
        if best is None:
            return None
        _d2, i, along, u = best
        ax, az, tx, tz, hop, s0 = self._segs[i]
        last = len(self._segs) - 1
        if (i == 0 and along < -slack) or (i == last and along > hop + slack):
            return None
        if (i == 0 and along < 0.0) or (i == last and along > hop):
            u = along
        return s0 + u, (x - ax) * -tz + (z - az) * tx, tx, tz


def _clip(pts, limit: float) -> list[tuple[float, float]]:
    out = [pts[0]]
    walked = 0.0
    for a, b in zip(pts, pts[1:]):
        walked += math.hypot(b[0] - a[0], b[1] - a[1])
        out.append(b)
        if walked >= limit:
            break
    return out


def _vehicle_boxes(v):
    """``(corners, speed)`` of a vehicle's body and of each nested trailer (README §12)."""
    try:
        yield v.get_corners(), float(v.speed)
    except (AttributeError, TypeError, ValueError):
        return
    for tr in getattr(v, "trailers", None) or ():
        try:
            if tr.is_zero():
                continue
            pos = tr.correct_position() if tr.is_tmp else tr.position
            half = 0.5 * float(tr.size.length)
            yaw = math.radians(tr.rotation.euler()[1])
            yield box(pos.x, pos.z, yaw, half, half, 0.5 * float(tr.size.width)), float(v.speed)
        except (AttributeError, TypeError, ValueError):
            continue


class SideTraffic:
    """Per side, the vehicle that makes moving over unsafe now, held ``hold_s`` after it clears.

    A vehicle dropped because it is leaving the beside zone is not held: the hold would undo ``exit_s``.
    """

    def __init__(self, tuning: SideTuning | None = None) -> None:
        self.tune = tuning or SideTuning()
        self._behind: tuple = ()
        self._behind_for: tuple | None = None
        self._behind_at: tuple[float, float] | None = None
        self._held: dict[int, tuple[Hit | None, float]] = {s: (None, -math.inf) for s in _SIDES}
        self.rig_length_m = math.nan

    def reset(self) -> None:
        self._behind = ()
        self._behind_for = None
        self._behind_at = None
        self._held = {s: (None, -math.inf) for s in _SIDES}

    def update(self, t: float, ego: Ego | None, vehicles, skip_ids, route, window) -> SideState:
        tn = self.tune
        if ego is None:
            return self._out(t, "no_data", {})
        if abs(ego.speed_ms) * 3.6 < tn.min_speed_kmh:
            return self._out(t, "speed", {})
        pts = getattr(route, "pts", ()) or ()
        if len(pts) < 2:
            return self._out(t, "no_map", {})
        frame = self._frame(route, window, ego)
        rig = self._rig(frame, ego)
        if rig is None:
            return self._out(t, "off_route", {})
        found: dict[int, Hit] = {}
        leaving: dict[int, set[int]] = {s: set() for s in _SIDES}
        lanes: dict[tuple[int, int], str] = {}
        reach = tn.reach_m + rig[1] - rig[0] + 20.0
        for v in vehicles or ():
            vid = getattr(v, "id", None)
            if vid in skip_ids or getattr(v, "is_parked", False):
                continue
            for corners, speed in _vehicle_boxes(v):
                judged = self._judge(frame, rig, ego, corners, speed, v, reach, window, lanes)
                if judged is None:
                    continue
                side, hit, exiting = judged
                if exiting:
                    leaving[side].add(hit.vid)
                elif _more_urgent(hit, found.get(side)):
                    found[side] = hit
        return self._out(t, "clear", found, leaving)

    def _frame(self, route, window, ego: Ego) -> LaneFrame:
        """Ego's lane from ``behind_route_m`` behind to ``ahead_route_m`` ahead, along travel."""
        tn = self.tune
        ahead = _clip(route.pts, tn.ahead_route_m)
        (ax, az), (bx, bz) = ahead[0], ahead[1]
        hop = math.hypot(bx - ax, bz - az) or 1.0
        tx, tz = (bx - ax) / hop, (bz - az) / hop
        key = (id(window), getattr(route, "lane_index", -1))
        moved = self._behind_at is None or math.hypot(ego.x - self._behind_at[0], ego.z - self._behind_at[1]) \
            > tn.behind_refresh_m
        if key != self._behind_for or moved:
            self._behind_for = key
            self._behind_at = (ego.x, ego.z)
            try:
                self._behind = route_behind(window, int(getattr(route, "lane_index", -1)), ax, az, tx, tz,
                                            tn.behind_route_m)
            except (TypeError, ValueError, AttributeError):
                self._behind = ()
        behind = list(reversed(self._behind))
        if len(behind) < 2:
            # No lane behind on the map: a straight line back from ego's foot.
            behind = [(ax - tx * tn.behind_route_m, az - tz * tn.behind_route_m), (ax, az)]
        return LaneFrame(behind + list(ahead))

    def _rig(self, frame: LaneFrame, ego: Ego) -> tuple[float, float, float, float] | None:
        """``(rear s, front s, leftmost, rightmost)`` of ego's rig in the lane frame."""
        ss, ds = [], []
        for corners in rig_boxes(ego):
            for x, z in corners:
                hit = frame.locate(x, z, self.tune.locate_slack_m)
                if hit is None:
                    return None
                ss.append(hit[0])
                ds.append(hit[1])
        self.rig_length_m = max(ss) - min(ss)
        return min(ss), max(ss), min(ds), max(ds)

    def _judge(self, frame, rig, ego, corners, speed, v, reach, window, lanes) -> tuple[int, Hit, bool] | None:
        """``(side, hit, leaving)`` for a body that makes a side unsafe, ``leaving`` when it clears the zone in time."""
        tn = self.tune
        cx = 0.25 * sum(c[0] for c in corners)
        cz = 0.25 * sum(c[1] for c in corners)
        if math.hypot(cx - ego.x, cz - ego.z) > reach:
            return None
        here = frame.locate(cx, cz, tn.locate_slack_m)
        if here is None:
            return None
        s_c, d_c, tx, tz = here
        if abs(d_c) > tn.far_m:
            return None
        fx = 0.5 * (corners[0][0] + corners[1][0]) - 0.5 * (corners[2][0] + corners[3][0])
        fz = 0.5 * (corners[0][1] + corners[1][1]) - 0.5 * (corners[2][1] + corners[3][1])
        length = math.hypot(fx, fz)
        if length < 1e-6:
            return None
        dot = (fx * tx + fz * tz) / length
        if abs(dot) < tn.parallel_cos:
            return None
        ss = [s_c + (x - cx) * tx + (z - cz) * tz for x, z in corners]
        ds = [d_c + (x - cx) * -tz + (z - cz) * tx for x, z in corners]
        lo, hi, d_lo, d_hi = min(ss), max(ss), min(ds), max(ds)
        rear, front, rig_left, rig_right = rig
        side = RIGHT if d_c > 0.0 else LEFT
        along = speed * dot
        v_ego = abs(ego.speed_ms)
        kind = ""
        gap = closing = 0.0
        when = math.inf
        exiting = False
        zone = (rear - tn.beside_rear_m, front + tn.beside_front_m)
        if hi >= zone[0] and lo <= zone[1]:
            clear = d_lo > rig_right if side == RIGHT else d_hi < rig_left
            if not clear:
                return None
            kind, closing = BESIDE, along - v_ego
            exiting = _exit_time(lo, hi, zone, closing) <= tn.exit_s
        elif abs(d_c) < tn.near_m:
            return None
        else:
            if hi < rear and dot > 0.0:
                kind, gap, closing, horizon = BEHIND, rear - hi, along - v_ego, tn.same_way_s
            elif lo > front and dot > 0.0:
                kind, gap, closing, horizon = AHEAD, lo - front, v_ego - along, tn.same_way_s
            elif lo > front:
                kind, gap, closing, horizon = ONCOMING, lo - front, v_ego - along, tn.oncoming_s
            else:
                return None
            if closing <= 0.0 or gap > tn.reach_m or gap > closing * horizon:
                return None
            when = gap / closing
        lane = edge_kinds.UNKNOWN
        if abs(d_c) >= tn.near_m:
            key = (side, int(s_c // 10.0))
            if key not in lanes:
                lanes[key] = edge_kinds.beyond(window, frame.pts, s_c, float(side))
            lane = lanes[key]
            if lane == edge_kinds.EDGE:
                return None
        hit = Hit(kind, int(getattr(v, "id", -1)), bool(getattr(v, "is_tmp", False)), gap, d_c, closing, when, lane,
                  length)
        return side, hit, exiting

    def _out(self, t: float, reason: str, found: dict[int, Hit], leaving=None) -> SideState:
        sides: dict[int, Hit | None] = {}
        for side in _SIDES:
            hit = found.get(side)
            held, until = self._held[side]
            gone = held is not None and leaving is not None and held.vid in leaving[side]
            if hit is not None:
                self._held[side] = (hit, t + self.tune.hold_s)
            elif held is not None and reason == "clear" and t < until and not gone:
                hit = held
            else:
                self._held[side] = (None, -math.inf)
            sides[side] = hit
        return SideState(sides[LEFT], sides[RIGHT], reason)


def _exit_time(lo: float, hi: float, zone: tuple[float, float], rel: float) -> float:
    """Seconds until a body spanning ``lo``..``hi`` along the lane leaves ``zone`` at ``rel`` relative to the rig."""
    if rel > 0.0:
        return (zone[1] - lo) / rel
    if rel < 0.0:
        return (hi - zone[0]) / -rel
    return math.inf


def _more_urgent(hit: Hit, than: Hit | None) -> bool:
    """Beside first, then whichever reaches the rig sooner."""
    if than is None:
        return True
    if (hit.kind == BESIDE) != (than.kind == BESIDE):
        return hit.kind == BESIDE
    return hit.time_s < than.time_s
