"""Road centerline: Hermite spline through the two nodes of a road item.

The tangent at each node is that node's ahead direction times the XZ chord
length. Ahead is local -Z of the node quaternion, the same yaw the map
tools use. On the base map that direction agrees with the chord on every
road (mean dot about 0.96).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.road_layout.sector import MapNode, SectorMesh


@dataclass(frozen=True)
class CenterlineHit:
    """Closest point on a sampled centerline to a query in the XZ plane."""

    distance_m: float
    road_y: float
    look: str
    road_uid: int


def node_forward_xz(node: MapNode) -> tuple[float, float]:
    """Unit ahead vector, X east and Z south."""
    yaw = math.atan2(-node.qy, node.qw) * 2.0 - math.pi / 2.0
    return math.cos(yaw), math.sin(yaw)


def sample_span(
    x0: float,
    y0: float,
    z0: float,
    fx0: float,
    fz0: float,
    x1: float,
    y1: float,
    z1: float,
    fx1: float,
    fz1: float,
    steps: int = 16,
) -> list[tuple[float, float, float, float, float]]:
    """Samples ``(x, y, z, forward_x, forward_z)`` along one Hermite span."""
    steps = max(int(steps), 1)
    dx = x1 - x0
    dz = z1 - z0
    chord = math.hypot(dx, dz)
    m0x, m0z = fx0 * chord, fz0 * chord
    m1x, m1z = fx1 * chord, fz1 * chord
    points: list[tuple[float, float, float, float, float]] = []
    for i in range(steps + 1):
        t = i / steps
        x = _hermite(t, x0, x1, m0x, m1x)
        z = _hermite(t, z0, z1, m0z, m1z)
        tx = _hermite_d(t, x0, x1, m0x, m1x)
        tz = _hermite_d(t, z0, z1, m0z, m1z)
        norm = math.hypot(tx, tz)
        if norm < 1e-8:
            fx, fz = fx0, fz0
        else:
            fx, fz = tx / norm, tz / norm
        y = (1.0 - t) * y0 + t * y1
        points.append((x, y, z, fx, fz))
    return points


def sample_centerline(
    start: MapNode,
    end: MapNode,
    steps: int = 16,
) -> list[tuple[float, float, float]]:
    """World (x, y, z) samples. Y is linear between the nodes."""
    f0x, f0z = node_forward_xz(start)
    f1x, f1z = node_forward_xz(end)
    span = sample_span(
        start.x, start.y, start.z, f0x, f0z,
        end.x, end.y, end.z, f1x, f1z,
        steps,
    )
    return [(x, y, z) for x, y, z, _fx, _fz in span]


def nearest_centerline(
    mesh: SectorMesh,
    x: float,
    z: float,
    steps: int = 16,
) -> CenterlineHit | None:
    """Minimum XZ distance from *(x, z)* to a road that has both nodes."""
    best: CenterlineHit | None = None
    for road in mesh.roads:
        start = mesh.nodes.get(road.start_uid)
        end = mesh.nodes.get(road.end_uid)
        if start is None or end is None:
            continue
        pts = sample_centerline(start, end, steps)
        for i in range(len(pts) - 1):
            dist, y = _segment_xz(x, z, pts[i], pts[i + 1])
            if best is None or dist < best.distance_m:
                best = CenterlineHit(dist, y, road.look, road.uid)
    return best


def roads_missing_a_node(mesh: SectorMesh) -> int:
    """Roads whose start or end node is not in this mesh."""
    missing = 0
    for road in mesh.roads:
        if road.start_uid not in mesh.nodes or road.end_uid not in mesh.nodes:
            missing += 1
    return missing


def _hermite_d(t: float, p0: float, p1: float, m0: float, m1: float) -> float:
    t2 = t * t
    return (
        (6.0 * t2 - 6.0 * t) * p0
        + (3.0 * t2 - 4.0 * t + 1.0) * m0
        + (-6.0 * t2 + 6.0 * t) * p1
        + (3.0 * t2 - 2.0 * t) * m1
    )


def _hermite(t: float, p0: float, p1: float, m0: float, m1: float) -> float:
    t2 = t * t
    t3 = t2 * t
    return (
        (2.0 * t3 - 3.0 * t2 + 1.0) * p0
        + (t3 - 2.0 * t2 + t) * m0
        + (-2.0 * t3 + 3.0 * t2) * p1
        + (t3 - t2) * m1
    )


def _segment_xz(
    x: float,
    z: float,
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float]:
    dx = b[0] - a[0]
    dz = b[2] - a[2]
    length2 = dx * dx + dz * dz
    if length2 < 1e-12:
        return math.hypot(x - a[0], z - a[2]), a[1]
    t = max(0.0, min(1.0, ((x - a[0]) * dx + (z - a[2]) * dz) / length2))
    y = a[1] + t * (b[1] - a[1])
    return math.hypot(x - (a[0] + t * dx), z - (a[2] + t * dz)), y
