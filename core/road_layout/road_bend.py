"""Curvature profile that bends a vehicle's predicted path along the road.

Only a vehicle that is following its lane is bent: on it, near its own heading,
and turning within the road's curvature over the next second (early or late
turn-in counts). Its profile then starts at its own curvature (steering for
ego, measured yaw rate for others) and blends into the road's over the ramp, a
lead into the bend. A corner that needs more lateral acceleration than the limit
at the vehicle's speed is ignored from where it starts: the path keeps the
vehicle's own curvature. Of several candidate routes the one needing the least
change of lateral force from what the vehicle does now wins. See
core/aeb/README.md, "Road-bent trajectories".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

_RESAMPLE_M = 2.5
_SMOOTH_M = 5.0
_MAIN_BIAS = 1e-4


@dataclass(frozen=True)
class BendParams:
    """``share`` of the road curvature the path takes; ``ramp_s`` of travel to reach it."""

    share: float = 1.0
    ramp_s: float = 1.0
    ramp_min_m: float = 10.0
    ramp_max_m: float = 50.0
    max_lat_ms2: float = 25.0
    choose_s: float = 4.0
    choose_min_m: float = 30.0
    choose_max_m: float = 150.0
    step_m: float = 5.0
    # A vehicle this far off its lane, or turned this far from it, is not following it.
    max_offset_m: float = 2.5
    max_heading_deg: float = 20.0
    # Following the road: own curvature inside the road's range over the next follow_s of
    # travel, give or take follow_tol_ms2 of lateral accel plus follow_tol_frac of the road.
    follow_s: float = 1.0
    follow_min_m: float = 10.0
    follow_tol_ms2: float = 0.2
    follow_tol_frac: float = 0.25


DEFAULT_BEND = BendParams()


@dataclass(frozen=True)
class RoadCurve:
    """A route resampled from its start: arc length, world points, curvature (left positive)."""

    s: tuple[float, ...]
    pts: tuple[tuple[float, float], ...]
    kappa: tuple[float, ...]

    def kappa_at(self, s: float) -> float:
        if not self.s:
            return 0.0
        if s <= 0.0:
            return self.kappa[0]
        i = min(int(s / _RESAMPLE_M), len(self.s) - 2)
        if i < 0:
            return self.kappa[0]
        f = (s - self.s[i]) / _RESAMPLE_M
        return self.kappa[i] + max(0.0, min(1.0, f)) * (self.kappa[i + 1] - self.kappa[i])

    @property
    def length(self) -> float:
        return self.s[-1] if self.s else 0.0


def road_curve(pts: Sequence[tuple[float, float]]) -> RoadCurve | None:
    """Resample a world polyline every 2.5 m and attach a smoothed curvature."""
    res = _resample(pts, _RESAMPLE_M)
    if len(res) < 3:
        return None
    heads = []
    for i in range(len(res) - 1):
        heads.append(math.atan2(res[i + 1][1] - res[i][1], res[i + 1][0] - res[i][0]))
    heads.append(heads[-1])
    half = max(1, int(round(_SMOOTH_M / _RESAMPLE_M)))
    kappa = []
    for i in range(len(res)):
        a = max(0, i - half)
        b = min(len(heads) - 2, i + half)
        if b <= a:
            kappa.append(0.0)
            continue
        dh = _wrap(heads[b] - heads[a])
        # World Z points south, so a left turn lowers atan2(dz, dx).
        kappa.append(-dh / ((b - a) * _RESAMPLE_M))
    s = tuple(i * _RESAMPLE_M for i in range(len(res)))
    return RoadCurve(s, tuple(res), tuple(kappa))


def choose_route(
    candidates: Sequence[Sequence[tuple[float, float]]],
    x: float,
    z: float,
    fwd_x: float,
    fwd_z: float,
    kappa_now: float,
    speed: float,
    params: BendParams = DEFAULT_BEND,
) -> int:
    """Index of the candidate whose end point needs the least change of curvature."""
    order = rank_routes(candidates, x, z, fwd_x, fwd_z, kappa_now, speed, params)
    return order[0] if order else 0


def rank_routes(
    candidates: Sequence[Sequence[tuple[float, float]]],
    x: float,
    z: float,
    fwd_x: float,
    fwd_z: float,
    kappa_now: float,
    speed: float,
    params: BendParams = DEFAULT_BEND,
) -> list[int]:
    """Candidate indices, least change of curvature to their end point first.

    The first candidate (the router's own) wins ties. Lateral force is speed squared
    times curvature, so at one speed the comparison is on curvature alone.
    """
    reach = max(params.choose_min_m, min(params.choose_max_m, abs(speed) * params.choose_s))
    scored = []
    for i, pts in enumerate(candidates):
        end = _point_along(pts, reach)
        if end is None:
            continue
        dx, dz = end[0] - x, end[1] - z
        ahead = dx * fwd_x + dz * fwd_z
        right = dx * (-fwd_z) + dz * fwd_x
        dist2 = ahead * ahead + right * right
        if dist2 < 1.0:
            continue
        kappa_req = -2.0 * right / dist2
        scored.append((abs(kappa_req - kappa_now) + (0.0 if i == 0 else _MAIN_BIAS), i))
    scored.sort()
    return [i for _c, i in scored]


def bend_profile(
    curve: RoadCurve | None,
    pose: tuple[float, float, float, float],
    kappa_now: float,
    speed: float,
    length: float,
    decel: float = 0.0,
    params: BendParams = DEFAULT_BEND,
    recent_kappa: Sequence[float] = (),
    merge_s: float | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Breakpoints ``(s, kappa)`` for a path of ``length`` metres; empty when not following.

    ``pose`` is ``(x, z, fwd_x, fwd_z)`` of the path start; ``recent_kappa`` the road
    curvature at the vehicle in the last few metres (see ``_following``). ``merge_s``
    is where two lanes become one: the path moves off its offset onto the lane by then.
    """
    if curve is None or length <= 0.0 or params.share <= 0.0:
        return (), ()
    s_offset = foot_offset(curve, pose[0], pose[1])
    if not _on_lane(curve, s_offset, pose, params):
        return (), ()
    if not _following(curve, s_offset, kappa_now, speed, params, recent_kappa):
        return (), ()
    ramp = max(params.ramp_min_m, min(params.ramp_max_m, abs(speed) * params.ramp_s)) if params.ramp_s > 0.0 else 0.0
    if merge_s:
        ss, ks = _position_blend(curve, s_offset, pose, kappa_now, length, ramp, params, merge_s)
    else:
        ss, ks = _curvature_blend(curve, s_offset, kappa_now, length, ramp, params)
    return _cap_corners(ss, ks, curve, s_offset, kappa_now, speed, decel, params)


def _curvature_blend(curve, s_offset, kappa_now, length, ramp, params):
    """Own curvature blended into the road's: keeps the vehicle's heading against its lane."""
    ss, ks = [0.0], [kappa_now]
    s = params.step_m
    while s <= length + 1e-9 and s_offset + s <= curve.length:
        w = params.share * (_smoothstep(s / ramp) if ramp > 0.0 else 1.0)
        ss.append(s)
        ks.append(kappa_now + w * (curve.kappa_at(s_offset + s) - kappa_now))
        s += params.step_m
    return ss, ks


def _position_blend(curve, s_offset, pose, kappa_now, length, ramp, params, merge_s=None):
    """Curvature of ``arc + g(s) * (lane - arc)`` in the vehicle frame (x ahead, y right).

    Used where two lanes become one: the lane is taken at the vehicle's current offset,
    faded to the centre by ``merge_s``, so the path joins the merged lane (README road-bent).
    """
    x0, z0, fx, fz = pose
    rx, rz = -fz, fx
    step = params.step_m
    held0 = _held_offset(curve, s_offset, pose)
    pts = []
    s = 0.0
    while s <= length + step + 1e-9:
        at = s_offset + s
        if at > curve.length:
            break
        lp = _curve_point(curve, at)
        lh = _curve_heading(curve, at)
        if lp is None or lh is None:
            break
        held = held0 * (1.0 - _smoothstep(s / merge_s)) if merge_s else held0
        wx = lp[0] - held * math.sin(lh) - x0
        wz = lp[1] + held * math.cos(lh) - z0
        lx, ly = wx * fx + wz * fz, wx * rx + wz * rz
        ax, ay = _arc_xy(kappa_now, s)
        g = params.share * (_smoothstep(s / ramp) if ramp > 0.0 else 1.0)
        pts.append((s, ax + g * (lx - ax), ay + g * (ly - ay)))
        s += step
    if len(pts) < 3:
        return [0.0], [kappa_now]
    heads = [math.atan2(pts[i + 1][2] - pts[i][2], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)]
    ss, ks = [0.0], [kappa_now]
    for i in range(1, len(heads)):
        seg = math.hypot(pts[i + 1][1] - pts[i - 1][1], pts[i + 1][2] - pts[i - 1][2]) * 0.5
        if seg < 1e-6:
            continue
        # y points right, so a left turn lowers the heading.
        ss.append(pts[i][0])
        ks.append(-_wrap(heads[i] - heads[i - 1]) / seg)
    return ss, ks


def _arc_xy(kappa: float, s: float) -> tuple[float, float]:
    """Point at arc length ``s`` of a constant-curvature arc, vehicle frame (x ahead, y right)."""
    if abs(kappa) < 1e-6:
        return s, 0.0
    return math.sin(kappa * s) / kappa, -(1.0 - math.cos(kappa * s)) / kappa


def _held_offset(curve, s_offset, pose) -> float:
    foot = _curve_point(curve, s_offset)
    h_foot = _curve_heading(curve, s_offset)
    if foot is None or h_foot is None:
        return 0.0
    return (pose[0] - foot[0]) * (-math.sin(h_foot)) + (pose[1] - foot[1]) * math.cos(h_foot)


def _on_lane(curve, s_offset, pose, params) -> bool:
    foot = _curve_point(curve, s_offset)
    h_foot = _curve_heading(curve, s_offset)
    if foot is None or h_foot is None:
        return False
    x0, z0, fx, fz = pose
    held = (x0 - foot[0]) * (-math.sin(h_foot)) + (z0 - foot[1]) * math.cos(h_foot)
    herr = _wrap(math.atan2(fz, fx) - h_foot)
    return abs(held) <= params.max_offset_m and abs(herr) <= math.radians(params.max_heading_deg)


def _following(curve, s_offset, kappa_now, speed, params, recent_kappa=()) -> bool:
    """Own curvature inside the road's over the next stretch: turning in early or late counts.

    ``recent_kappa`` widens the range with the road just behind: the smoothed curvature at
    the foot already leans into a bend a few metres ahead, the road behind does not.
    """
    reach = max(params.follow_min_m, abs(speed) * params.follow_s)
    ks = [curve.kappa_at(s_offset + d) for d in _frange(0.0, reach, _RESAMPLE_M)]
    ks.extend(recent_kappa)
    lo, hi = min(ks), max(ks)
    tol = params.follow_tol_ms2 / max(speed * speed, 25.0) + params.follow_tol_frac * max(abs(lo), abs(hi))
    return lo - tol <= kappa_now <= hi + tol


def _frange(a: float, b: float, step: float):
    x = a
    while x <= b + 1e-9:
        yield x
        x += step


def _cap_corners(ss, ks, curve, s_offset, kappa_now, speed, decel, params):
    """Hold ``kappa_now`` from the start of the first corner that needs more than the limit.

    The corner starts where the road's curvature, walked back from the first point past
    the limit, stops growing toward it.
    """
    v0_sq = speed * speed
    over = None
    for i, s in enumerate(ss):
        v_sq = max(v0_sq - 2.0 * max(decel, 0.0) * s, 0.0)
        if s > 0.0 and v_sq * abs(curve.kappa_at(s_offset + s)) > params.max_lat_ms2:
            over = i
            break
    if over is None:
        out_s, out_k = list(ss), list(ks)
        if out_s and out_s[-1] > 0.0:
            out_s.append(out_s[-1] + params.step_m)
            out_k.append(kappa_now)
        return tuple(out_s), tuple(out_k)
    sign = math.copysign(1.0, curve.kappa_at(s_offset + ss[over]))
    start = over
    while start > 1:
        here = sign * curve.kappa_at(s_offset + ss[start - 1])
        if here <= 0.0 or here >= sign * curve.kappa_at(s_offset + ss[start]):
            break
        start -= 1
    if start <= 1:
        return (), ()
    return tuple(ss[:start]) + (ss[start],), tuple(ks[:start]) + (kappa_now,)


def _curve_point(curve: RoadCurve, s: float):
    if not curve.s or s < 0.0 or s > curve.length:
        return None
    i = min(int(s / _RESAMPLE_M), len(curve.pts) - 2)
    f = (s - curve.s[i]) / _RESAMPLE_M
    a, b = curve.pts[i], curve.pts[i + 1]
    return a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])


def _curve_heading(curve: RoadCurve, s: float):
    """``atan2(dz, dx)`` of the curve at ``s``."""
    if len(curve.pts) < 2:
        return None
    i = max(0, min(int(s / _RESAMPLE_M), len(curve.pts) - 2))
    a, b = curve.pts[i], curve.pts[i + 1]
    return math.atan2(b[1] - a[1], b[0] - a[0])


def lateral_offset(curve: RoadCurve, x: float, z: float) -> tuple[float, float]:
    """``(s, right)``: arc length of the nearest point on ``curve`` and the offset right of it."""
    s = foot_offset(curve, x, z)
    p = _curve_point(curve, s)
    h = _curve_heading(curve, s)
    if p is None or h is None:
        return s, math.inf
    return s, (x - p[0]) * (-math.sin(h)) + (z - p[1]) * math.cos(h)


def foot_kappa(curve: RoadCurve, x: float, z: float) -> float:
    """Road curvature at the point of ``curve`` nearest ``(x, z)``."""
    return curve.kappa_at(foot_offset(curve, x, z))


def foot_offset(curve: RoadCurve, x: float, z: float) -> float:
    """Arc length along ``curve`` of the point nearest ``(x, z)``."""
    best_s, best_d = 0.0, math.inf
    pts = curve.pts
    for i in range(len(pts) - 1):
        ax, az = pts[i]
        sx, sz = pts[i + 1][0] - ax, pts[i + 1][1] - az
        length2 = sx * sx + sz * sz
        if length2 < 1e-12:
            continue
        t = max(0.0, min(1.0, ((x - ax) * sx + (z - az) * sz) / length2))
        d = (x - ax - t * sx) ** 2 + (z - az - t * sz) ** 2
        if d < best_d:
            best_d, best_s = d, curve.s[i] + t * _RESAMPLE_M
    return best_s


def _point_along(pts, dist: float):
    if len(pts) < 2:
        return None
    left = dist
    for i in range(1, len(pts)):
        hop = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
        if hop >= left:
            f = left / hop if hop > 1e-9 else 0.0
            return (pts[i - 1][0] + f * (pts[i][0] - pts[i - 1][0]),
                    pts[i - 1][1] + f * (pts[i][1] - pts[i - 1][1]))
        left -= hop
    return pts[-1]


def _resample(pts, step: float) -> list[tuple[float, float]]:
    if len(pts) < 2:
        return list(pts)
    out = [tuple(pts[0])]
    carry = 0.0
    for i in range(1, len(pts)):
        ax, az = pts[i - 1]
        bx, bz = pts[i]
        hop = math.hypot(bx - ax, bz - az)
        if hop < 1e-9:
            continue
        pos = step - carry
        while pos <= hop + 1e-9:
            f = pos / hop
            out.append((ax + f * (bx - ax), az + f * (bz - az)))
            pos += step
        carry = hop - (pos - step)
    return out


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)
