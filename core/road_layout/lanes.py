"""Lane centerlines: each road lane, plus every prefab driving curve.

Road lanes are the carriageway spline shifted to the middle of each lane.
Prefab curves are already one lane each. Both are in the telemetry frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.road_layout.centerline import node_forward_xz, sample_span
from core.road_layout.looks import RoadLook
from core.road_layout.prefab import PrefabLibrary, place_curve
from core.road_layout.sector import MapNode, SectorMesh

# A short junction stays at the caller's step count. A long bend grows from here.
_SAMPLE_M = 10.0
_MAX_STEPS = 80


@dataclass(frozen=True)
class LaneLine:
    """One sampled lane. ``kind`` is ``road`` or ``prefab``."""

    kind: str
    label: str
    points: tuple[tuple[float, float, float], ...]
    prefab_id: int | None = None


@dataclass(frozen=True)
class LaneHit:
    distance_m: float
    road_y: float
    kind: str
    label: str


def lane_lines(
    mesh: SectorMesh,
    looks: dict[str, RoadLook],
    prefabs: PrefabLibrary | None,
    steps: int = 12,
    *,
    apply_shift: bool = True,
) -> list[LaneLine]:
    """Every lane centerline whose endpoints are in this mesh."""
    lines: list[LaneLine] = []
    for road in mesh.roads:
        start = mesh.nodes.get(road.start_uid)
        end = mesh.nodes.get(road.end_uid)
        if start is None or end is None:
            continue
        look = looks.get(road.look)
        laterals = look.laterals(apply_shift=apply_shift) if look is not None else [0.0]
        if not laterals:
            laterals = [0.0]
        span = _road_span(start, end, _steps_for(road.length, steps))
        for index, lateral in enumerate(laterals):
            lines.append(LaneLine(
                "road",
                road.look,
                tuple(_shift(span, lateral)),
            ))
    if prefabs is None:
        return lines
    prefab_n = 0
    for item in mesh.prefabs:
        if not item.node_uids:
            continue
        world = mesh.nodes.get(item.node_uids[0])
        desc = prefabs.curves_for(item)
        if world is None or desc is None:
            continue
        prefab_n += 1
        for curve in desc.curves:
            ends = place_curve(curve, world, desc, item.origin_index)
            if ends is None:
                continue
            (x0, y0, z0, yaw0), (x1, y1, z1, yaw1) = ends
            span = sample_span(
                x0, y0, z0, math.cos(yaw0), math.sin(yaw0),
                x1, y1, z1, math.cos(yaw1), math.sin(yaw1),
                _steps_for(math.hypot(x1 - x0, z1 - z0), steps),
            )
            lines.append(LaneLine(
                "prefab",
                item.model,
                tuple((x, y, z) for x, y, z, _fx, _fz in span),
                prefab_n,
            ))
    return lines


def nearest_lane(lines: list[LaneLine], x: float, z: float) -> LaneHit | None:
    """Minimum XZ distance from *(x, z)* to a sampled lane."""
    best: LaneHit | None = None
    for line in lines:
        pts = line.points
        for i in range(len(pts) - 1):
            dist, height = _segment(x, z, pts[i], pts[i + 1])
            if best is None or dist < best.distance_m:
                best = LaneHit(dist, height, line.kind, line.label)
    return best


def _steps_for(length_m: float, floor: int) -> int:
    """More samples on a longer span. ``floor`` is the count a short span keeps."""
    counted = math.ceil(max(length_m, 0.0) / _SAMPLE_M)
    return min(_MAX_STEPS, max(int(floor), counted))


def _road_span(start: MapNode, end: MapNode, steps: int):
    f0x, f0z = node_forward_xz(start)
    f1x, f1z = node_forward_xz(end)
    return sample_span(
        start.x, start.y, start.z, f0x, f0z,
        end.x, end.y, end.z, f1x, f1z,
        steps,
    )


def _shift(span, lateral: float):
    for x, y, z, fx, fz in span:
        yield (x + lateral * (-fz), y, z + lateral * fx)


def _segment(x: float, z: float, a, b) -> tuple[float, float]:
    dx = b[0] - a[0]
    dz = b[2] - a[2]
    length2 = dx * dx + dz * dz
    if length2 < 1e-12:
        return math.hypot(x - a[0], z - a[2]), a[1]
    t = max(0.0, min(1.0, ((x - a[0]) * dx + (z - a[2]) * dz) / length2))
    return (
        math.hypot(x - (a[0] + t * dx), z - (a[2] + t * dz)),
        a[1] + t * (b[1] - a[1]),
    )
