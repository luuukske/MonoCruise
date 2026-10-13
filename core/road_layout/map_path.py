"""Ego's predicted path: the steering arc near the truck, the map lane further out.

Ego frame: x forward, y to the right, origin at the path start. The arc is the
consumer's own curvature (positive turns left), so the path leaves ego along its
heading and its curvature, and only bends onto the lane past ``blend_m``. The
lane is taken at ego's current offset from it, never steered back to the centre.

``shape`` is the geometry; ``trust`` says how much of it a consumer may use at
each distance. Trust falls with ego's offset from the lane centre, its heading
error, and wherever another branch has left the route.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

_STEP_M = 2.5
_STRAIGHT_KAPPA = 1e-6


@dataclass(frozen=True)
class PathParams:
    """Blend and trust knobs. ``blend_s`` times speed is the steering-led stretch."""

    blend_s: float = 2.5
    blend_min_m: float = 15.0
    blend_max_m: float = 60.0
    offset_full_m: float = 1.2
    offset_zero_m: float = 2.4
    heading_full_rad: float = math.radians(8.0)
    heading_zero_rad: float = math.radians(16.0)
    branch_zero_m: float = 1.5


DEFAULT_PARAMS = PathParams()


@dataclass(frozen=True)
class MapPath:
    """Samples every ``_STEP_M`` of arc length: position, heading (right of +x), trust."""

    s: tuple[float, ...]
    x: tuple[float, ...]
    y: tuple[float, ...]
    heading: tuple[float, ...]
    trust: tuple[float, ...]
    offset_m: float
    blend_m: float

    @property
    def reach_m(self) -> float:
        return self.s[-1] if self.s else 0.0

    def trust_at(self, s: float) -> float:
        """Trust at arc length ``s``; zero past the end."""
        if not self.s or s < 0.0 or s > self.s[-1]:
            return 0.0
        i = min(int(s / _STEP_M), len(self.s) - 2)
        f = (s - self.s[i]) / _STEP_M
        return self.trust[i] + f * (self.trust[i + 1] - self.trust[i])

    def point_at(self, s: float) -> tuple[float, float]:
        if not self.s:
            return s, 0.0
        s = max(0.0, min(s, self.s[-1]))
        i = min(int(s / _STEP_M), len(self.s) - 2)
        f = (s - self.s[i]) / _STEP_M
        return (
            self.x[i] + f * (self.x[i + 1] - self.x[i]),
            self.y[i] + f * (self.y[i + 1] - self.y[i]),
        )

    def coords(self, px: float, py: float) -> tuple[float, float] | None:
        """``(s, offset right of the path)`` of an ego-frame point, None off either end."""
        n = len(self.s)
        if n < 2:
            return None
        best_i = 0
        best_d = math.inf
        for i in range(n):
            d = (self.x[i] - px) ** 2 + (self.y[i] - py) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        found = None
        for i in (best_i - 1, best_i):
            if i < 0 or i >= n - 1:
                continue
            ax, ay = self.x[i], self.y[i]
            sx, sy = self.x[i + 1] - ax, self.y[i + 1] - ay
            length2 = sx * sx + sy * sy
            if length2 < 1e-12:
                continue
            raw = ((px - ax) * sx + (py - ay) * sy) / length2
            t = max(0.0, min(1.0, raw))
            d2 = (px - ax - t * sx) ** 2 + (py - ay - t * sy) ** 2
            if found is None or d2 < found[0]:
                side = ((px - ax) * (-sy) + (py - ay) * sx) / math.sqrt(length2)
                found = (d2, i, raw, self.s[i] + t * (self.s[i + 1] - self.s[i]), side)
        if found is None:
            return None
        _d2, i, raw, s_at, side = found
        if (i == 0 and raw < 0.0) or (i == n - 2 and raw > 1.0):
            return None
        return s_at, side


def blend_length(speed_ms: float, params: PathParams = DEFAULT_PARAMS) -> float:
    return max(params.blend_min_m, min(params.blend_max_m, abs(speed_ms) * params.blend_s))


def build_map_path(
    route,
    ego_x: float,
    ego_z: float,
    fwd_x: float,
    fwd_z: float,
    kappa: float,
    speed_ms: float,
    params: PathParams = DEFAULT_PARAMS,
    origin_m: float = 0.0,
) -> MapPath | None:
    """Fused path from a ``LaneRoute``. ``origin_m`` moves the start along ego's heading.

    ``ego_x``/``ego_z`` is the pose the route was built for; ``kappa`` is the
    consumer's own curvature at that pose.
    """
    if route is None or len(route.pts) < 2:
        return None
    ox = ego_x + origin_m * fwd_x
    oz = ego_z + origin_m * fwd_z
    local = [_to_ego(ox, oz, fwd_x, fwd_z, px, pz) for px, pz in route.pts]
    lane = _resample_from_foot(local)
    if lane is None or len(lane) < 2:
        return None
    offset = lane[0][3]
    blend = blend_length(speed_ms, params)
    base = _base_trust(route, params)
    div = route.divergence
    s_out: list[float] = []
    x_out: list[float] = []
    y_out: list[float] = []
    h_out: list[float] = []
    w_out: list[float] = []
    merge_s = getattr(route, "merge_s", None)
    for s, lx, ly, _off, lh in lane:
        ax, ay, ah = _arc(kappa, s)
        # Where two lanes become one, the offset ego holds from its lane fades out by the merge.
        held = offset * (1.0 - _smoothstep(s / merge_s)) if merge_s else offset
        mx = lx - held * math.sin(lh)
        my = ly + held * math.cos(lh)
        g = _smoothstep(s / blend) if blend > 0.0 else 1.0
        px = ax + g * (mx - ax)
        py = ay + g * (my - ay)
        dh = math.atan2(math.sin(lh - ah), math.cos(lh - ah))
        s_out.append(s)
        x_out.append(px)
        y_out.append(py)
        h_out.append(ah + g * dh)
        w_out.append(base * _branch_trust(div, s, params))
    return MapPath(
        tuple(s_out), tuple(x_out), tuple(y_out), tuple(h_out), tuple(w_out), offset, blend,
    )


def world_points(path: MapPath, ego_x: float, ego_z: float, fwd_x: float, fwd_z: float, origin_m: float = 0.0):
    """World ``(x, z)`` of every path sample, for drawing."""
    ox = ego_x + origin_m * fwd_x
    oz = ego_z + origin_m * fwd_z
    return tuple(
        (ox + x * fwd_x - y * fwd_z, oz + x * fwd_z + y * fwd_x)
        for x, y in zip(path.x, path.y)
    )


def _base_trust(route, params: PathParams) -> float:
    off = abs(route.offset_m)
    merge_off = getattr(route, "merge_offset_m", None)
    if merge_off is not None:
        # Ego on a lane that runs into the route is on its lane, not off it.
        off = min(off, abs(merge_off))
    head = abs(route.heading_err_rad)
    w_off = 1.0 - _smoothstep((off - params.offset_full_m) / max(params.offset_zero_m - params.offset_full_m, 1e-6))
    w_head = 1.0 - _smoothstep(
        (head - params.heading_full_rad) / max(params.heading_zero_rad - params.heading_full_rad, 1e-6)
    )
    return w_off * w_head


def _branch_trust(div: Sequence[tuple[float, float]], s: float, params: PathParams) -> float:
    """1 until a branch parts from the route, 0 once it is ``branch_zero_m`` away."""
    widest = 0.0
    for ds, gap in div:
        if ds > s:
            break
        widest = max(widest, gap)
    return 1.0 - _smoothstep(widest / params.branch_zero_m)


def _arc(kappa: float, s: float) -> tuple[float, float, float]:
    if abs(kappa) < _STRAIGHT_KAPPA:
        return s, 0.0, 0.0
    theta = kappa * s
    return math.sin(theta) / kappa, -(1.0 - math.cos(theta)) / kappa, -theta


def _to_ego(ox, oz, fwd_x, fwd_z, px, pz) -> tuple[float, float]:
    dx = px - ox
    dz = pz - oz
    return dx * fwd_x + dz * fwd_z, dx * (-fwd_z) + dz * fwd_x


def _resample_from_foot(local: list[tuple[float, float]]):
    """``(s, x, y, offset, heading)`` every step along the lane, starting at ego's foot.

    ``offset`` is ego's (the origin's) distance right of the lane, set at the foot.
    """
    best = None
    for i in range(len(local) - 1):
        ax, ay = local[i]
        bx, by = local[i + 1]
        sx, sy = bx - ax, by - ay
        length2 = sx * sx + sy * sy
        if length2 < 1e-12:
            continue
        t = max(0.0, min(1.0, (-ax * sx - ay * sy) / length2))
        fx, fy = ax + t * sx, ay + t * sy
        d2 = fx * fx + fy * fy
        if best is None or d2 < best[0]:
            best = (d2, i, t, fx, fy)
    if best is None:
        return None
    _d2, i0, _t, fx, fy = best
    pts = [(fx, fy)] + local[i0 + 1:]
    hx, hy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
    n = math.hypot(hx, hy)
    if n < 1e-9:
        return None
    offset = (-fx * (-hy) + -fy * hx) / n
    out = []
    walked = 0.0
    mark = 0.0
    for k in range(1, len(pts)):
        ax, ay = pts[k - 1]
        bx, by = pts[k]
        hop = math.hypot(bx - ax, by - ay)
        if hop < 1e-9:
            continue
        heading = math.atan2(by - ay, bx - ax)
        while mark <= walked + hop + 1e-9:
            f = (mark - walked) / hop
            out.append((mark, ax + f * (bx - ax), ay + f * (by - ay), offset, heading))
            mark += _STEP_M
        walked += hop
    return out


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)
