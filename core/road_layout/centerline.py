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


def sample_centerline(
    start: MapNode,
    end: MapNode,
    steps: int = 16,
) -> list[tuple[float, float, float]]:
    """World (x, y, z) samples. Y is linear between the nodes."""
    steps = max(int(steps), 1)
    dx = end.x - start.x
    dz = end.z - start.z
    chord = math.hypot(dx, dz)
    f0x, f0z = node_forward_xz(start)
    f1x, f1z = node_forward_xz(end)
    m0x, m0z = f0x * chord, f0z * chord
    m1x, m1z = f1x * chord, f1z * chord
    points: list[tuple[float, float, float]] = []
    for i in range(steps + 1):
        t = i / steps
        points.append((
            _hermite(t, start.x, end.x, m0x, m1x),
            (1.0 - t) * start.y + t * end.y,
            _hermite(t, start.z, end.z, m0z, m1z),
        ))
    return points


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
