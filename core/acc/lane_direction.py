"""Lane direction for the shared road model.

The samples are the line ego would drive by holding its current offset from
the lane it is in. A sideways offset is not steered back to the lane centre.
Near ego the line is blended into the truck's own heading, so a vehicle
straight ahead is not lost to a small map heading error. Between two lanes of
similar score, and only off a centreline, the directions are averaged by
distance. See ``core/acc/README.md`` §9.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

# Full lane direction by this distance. Closer points stay with ego's heading.
_HEADING_BLEND_M = 50.0
# Inside this, ego is on that lane's centreline and the other lane is ignored.
_ON_CENTER_M = 0.6
# Second lane is averaged in only when its score is this close to the best.
_SIMILAR_COST_M = 0.9
_DIST_FLOOR_M = 0.4
_SAMPLE_STEP_M = 10.0
_SAMPLE_MIN_M = 10.0
_SAMPLE_MAX_M = 150.0


@dataclass(frozen=True)
class LaneDirection:
    """One candidate lane. ``pts`` are world ``(x, z)`` in order along the lane."""

    pts: tuple[tuple[float, float], ...]
    distance_m: float
    cost: float


def lane_direction_samples(
    routes: Sequence[LaneDirection],
    ego_x: float,
    ego_z: float,
    fwd_x: float,
    fwd_z: float,
) -> list[tuple[float, float]]:
    """Ego-frame centreline points. Empty when no route is usable."""
    chosen = _choose(routes)
    if not chosen:
        return []
    traces: list[tuple[_Trace, float]] = []
    for route, weight in chosen:
        traced = _trace(route.pts, ego_x, ego_z, fwd_x, fwd_z)
        if traced is not None and traced.reach_m >= _SAMPLE_MIN_M:
            traces.append((traced, weight))
    if not traces:
        return []
    out: list[tuple[float, float]] = []
    distance = _SAMPLE_MIN_M
    while distance <= _SAMPLE_MAX_M + 1e-6:
        num_s = 0.0
        num_f = 0.0
        den = 0.0
        for traced, weight in traces:
            sample = _at(traced, distance)
            if sample is None:
                continue
            forward, shape = sample
            num_s += weight * shape
            num_f += weight * forward
            den += weight
        if den <= 0.0:
            break
        keep = _heading_keep(distance)
        out.append((num_f / den, keep * num_s / den))
        distance += _SAMPLE_STEP_M
    if len(out) < 3:
        return []
    return out


def _choose(routes: Sequence[LaneDirection]) -> list[tuple[LaneDirection, float]]:
    usable = [route for route in routes if len(route.pts) >= 2]
    if not usable:
        return []
    usable.sort(key=lambda route: route.cost)
    best = usable[0]
    if best.distance_m <= _ON_CENTER_M or len(usable) == 1:
        return [(best, 1.0)]
    second = usable[1]
    if second.cost - best.cost > _SIMILAR_COST_M:
        return [(best, 1.0)]
    first = 1.0 / max(best.distance_m, _DIST_FLOOR_M)
    other = 1.0 / max(second.distance_m, _DIST_FLOOR_M)
    return [(best, first), (second, other)]


def _heading_keep(distance_m: float) -> float:
    """Share of the lane direction. Zero at ego, one past the blend distance."""
    t = max(0.0, min(1.0, distance_m / _HEADING_BLEND_M))
    return t * t * (3.0 - 2.0 * t)


@dataclass(frozen=True)
class _Trace:
    # (along m, ego forward, lateral change from the foot)
    samples: tuple[tuple[float, float, float], ...]
    reach_m: float


def _trace(pts, ego_x, ego_z, fwd_x, fwd_z) -> _Trace | None:
    foot = _foot(pts, ego_x, ego_z, fwd_x, fwd_z)
    if foot is None:
        return None
    seg, param, step = foot
    walked = _walk(pts, seg, param, step, _SAMPLE_MAX_M)
    if len(walked) < 2:
        return None
    fx, fz = walked[0][1], walked[0][2]
    fwd0, lat0 = _to_ego(ego_x, ego_z, fwd_x, fwd_z, fx, fz)
    samples = []
    for along, x, z in walked:
        forward, lat = _to_ego(ego_x, ego_z, fwd_x, fwd_z, x, z)
        samples.append((along, forward - fwd0, lat - lat0))
    return _Trace(tuple(samples), samples[-1][0])


def _at(traced: _Trace, along: float) -> tuple[float, float] | None:
    samples = traced.samples
    if along > samples[-1][0] + 1e-6:
        return None
    for i in range(1, len(samples)):
        a0, f0, s0 = samples[i - 1]
        a1, f1, s1 = samples[i]
        if along > a1:
            continue
        span = a1 - a0
        frac = 0.0 if span < 1e-8 else (along - a0) / span
        return f0 + frac * (f1 - f0), s0 + frac * (s1 - s0)
    return None


def _foot(pts, ego_x, ego_z, fwd_x, fwd_z):
    best = None
    for i in range(len(pts) - 1):
        ax, az = pts[i]
        bx, bz = pts[i + 1]
        sx, sz = bx - ax, bz - az
        length2 = sx * sx + sz * sz
        if length2 < 1e-12:
            continue
        t = max(0.0, min(1.0, ((ego_x - ax) * sx + (ego_z - az) * sz) / length2))
        dist = math.hypot(ego_x - (ax + t * sx), ego_z - (az + t * sz))
        if best is None or dist < best[0]:
            best = (dist, i, t, sx, sz)
    if best is None:
        return None
    _dist, seg, param, sx, sz = best
    step = 1 if sx * fwd_x + sz * fwd_z >= 0.0 else -1
    return seg, param, step


def _walk(pts, seg, param, step, limit):
    ax, az = pts[seg]
    bx, bz = pts[seg + 1]
    foot = (ax + param * (bx - ax), az + param * (bz - az))
    out = [(0.0, foot[0], foot[1])]
    if step > 0:
        cursor = seg + 1
        stop = len(pts)
    else:
        cursor = seg
        stop = -1
    prev = foot
    traveled = 0.0
    mark = _SAMPLE_STEP_M
    while cursor != stop and traveled < limit:
        vx, vz = pts[cursor]
        hop = math.hypot(vx - prev[0], vz - prev[1])
        if hop > 1e-8:
            while mark <= limit and traveled + hop >= mark - 1e-6:
                frac = (mark - traveled) / hop
                out.append((
                    mark,
                    prev[0] + frac * (vx - prev[0]),
                    prev[1] + frac * (vz - prev[1]),
                ))
                mark += _SAMPLE_STEP_M
            traveled += hop
        prev = (vx, vz)
        cursor += step
    return out


def _to_ego(ego_x, ego_z, fwd_x, fwd_z, x, z) -> tuple[float, float]:
    dx = x - ego_x
    dz = z - ego_z
    forward = dx * fwd_x + dz * fwd_z
    lateral = dx * (-fwd_z) + dz * fwd_x
    return forward, lateral


def centreline_world(
    model,
    ego_x: float,
    ego_z: float,
    fwd_x: float,
    fwd_z: float,
    reach_m: float = 150.0,
    step_m: float = 5.0,
) -> tuple[tuple[float, float], ...]:
    """World ``(x, z)`` points along the road-model centreline, ahead of ego."""
    right_x = -fwd_z
    right_z = fwd_x
    pts = []
    distance = 0.0
    while distance <= reach_m + 1e-6:
        forward, lateral = model.point_at(distance)
        pts.append((
            ego_x + forward * fwd_x + lateral * right_x,
            ego_z + forward * fwd_z + lateral * right_z,
        ))
        distance += step_m
    return tuple(pts)
