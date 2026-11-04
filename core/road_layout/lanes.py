"""Lane centerlines: each road lane, plus every prefab driving curve.

Road lanes are the carriageway spline shifted to the middle of each lane.
Prefab curves are already one lane each. Both are in the telemetry frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.road_layout.centerline import node_forward_xz, road_tangent_m, sample_span
from core.road_layout.looks import RoadLook
from core.road_layout.prefab import PrefabLibrary, place_curve
from core.road_layout.sector import MapNode, SectorMesh

# A short junction stays at the caller's step count. A long bend grows from here.
_SAMPLE_M = 10.0
_MAX_STEPS = 80
# A continuation, not a crossing or a fork mouth. About 35 degrees.
_JOIN_COS = math.cos(math.radians(35.0))
_LATERAL_EPS_M = 0.05


@dataclass(frozen=True)
class LaneLine:
    """One sampled lane. ``kind`` is ``road`` or ``prefab``.

    ``uid`` is the road or prefab item. ``lateral_m`` is the look's own offset from the spline.
    ``travel`` is +1 when traffic runs in point order, -1 against it, 0 when unknown.
    """

    kind: str
    label: str
    points: tuple[tuple[float, float, float], ...]
    prefab_id: int | None = None
    uid: int = 0
    lane_index: int = -1
    lane_count: int = 0
    lateral_m: float = 0.0
    travel: int = 0


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
    road_tangent: str = "stored",
    prefab_tangent: str = "chord",
) -> list[LaneLine]:
    """Every lane centerline whose endpoints are in this mesh.

    ``road_tangent`` is ``chord``, ``stored`` (tangent = stored road length) or ``circle``.
    ``prefab_tangent`` is ``chord`` or ``circle``.
    """
    lines: list[LaneLine] = []
    joined_by_uid = _join_laterals(mesh, looks, apply_shift)
    for road in mesh.roads:
        start = mesh.nodes.get(road.start_uid)
        end = mesh.nodes.get(road.end_uid)
        if start is None or end is None:
            continue
        look = looks.get(road.look)
        laterals = look.laterals(apply_shift=apply_shift) if look is not None else [0.0]
        if not laterals:
            laterals = [0.0]
        span = _road_span(
            start, end, _steps_for(road.length, steps),
            _road_magnitude(start, end, road.length, road_tangent),
        )
        travels = _travels(look, len(laterals))
        joined = joined_by_uid.get(road.uid)
        for index, lateral in enumerate(laterals):
            start_lat, end_lat = (joined[0][index], joined[1][index]) if joined else (lateral, lateral)
            lines.append(LaneLine(
                "road",
                road.look,
                _lane_points(span, lateral, start_lat, end_lat),
                None,
                road.uid,
                index,
                len(laterals),
                lateral,
                travels[index],
            ))
    if prefabs is None:
        return lines
    for item in mesh.prefabs:
        if not item.node_uids:
            continue
        world = mesh.nodes.get(item.node_uids[0])
        desc = prefabs.curves_for(item)
        if world is None or desc is None:
            continue
        for c_index, curve in enumerate(desc.curves):
            ends = place_curve(curve, world, desc, item.origin_index)
            if ends is None:
                continue
            (x0, y0, z0, yaw0), (x1, y1, z1, yaw1) = ends
            f0 = (math.cos(yaw0), math.sin(yaw0))
            f1 = (math.cos(yaw1), math.sin(yaw1))
            chord = math.hypot(x1 - x0, z1 - z0)
            mag = _circle_magnitude(chord, f0, f1) if prefab_tangent == "circle" else None
            span = sample_span(
                x0, y0, z0, f0[0], f0[1],
                x1, y1, z1, f1[0], f1[1],
                _steps_for(chord, steps),
                mag,
            )
            lines.append(LaneLine(
                "prefab",
                item.model,
                tuple((x, y, z) for x, y, z, _fx, _fz in span),
                item.uid,
                item.uid,
                c_index,
                len(desc.curves),
                0.0,
                1,
            ))
    return lines


def _travels(look, count: int) -> list[int]:
    """Per lane, +1 start to end or -1: right lanes run forward, left lanes back on a two-way road.

    A road with left lanes only is one-way and runs forward: ego drove every such lane in
    the clip corpus start to end (318 of 318 samples), against 89 % on two-way left lanes.
    """
    if look is None or count <= 0:
        return [0] * max(count, 0)
    right, left = len(look.lanes_right), len(look.lanes_left)
    if right + left != count:
        return [0] * count
    if not right:
        return [1] * count
    return [1] * right + [-1] * left


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


def _road_span(start: MapNode, end: MapNode, steps: int, magnitude: float | None = None):
    f0x, f0z = node_forward_xz(start)
    f1x, f1z = node_forward_xz(end)
    return sample_span(
        start.x, start.y, start.z, f0x, f0z,
        end.x, end.y, end.z, f1x, f1z,
        steps,
        magnitude,
    )


def _road_magnitude(start: MapNode, end: MapNode, length: float, mode: str) -> float | None:
    if mode == "stored":
        return road_tangent_m(start, end, length)
    if mode == "circle":
        chord = math.hypot(end.x - start.x, end.z - start.z)
        return _circle_magnitude(chord, node_forward_xz(start), node_forward_xz(end))
    return None


def _circle_magnitude(
    chord: float,
    f0: tuple[float, float],
    f1: tuple[float, float],
) -> float | None:
    """Hermite tangent that matches a circular arc of the same turn and chord."""
    turn = abs(math.atan2(f0[0] * f1[1] - f0[1] * f1[0], f0[0] * f1[0] + f0[1] * f1[1]))
    if turn < 1e-3 or chord < 1e-6:
        return None
    return chord * 2.0 * math.tan(turn / 4.0) / math.sin(turn / 2.0)


def _lane_points(span, own: float, start: float, end: float) -> tuple:
    """Constant shift, or a fade from ``start`` at the first sample to ``end`` at the last."""
    if abs(start - own) < _LATERAL_EPS_M and abs(end - own) < _LATERAL_EPS_M:
        return tuple(_shift(span, own))
    if abs(start - end) < _LATERAL_EPS_M:
        return tuple(_shift(span, start))
    return tuple(_shift_fade(span, start, end))


def _shift(span, lateral: float):
    for x, y, z, fx, fz in span:
        yield (x + lateral * (-fz), y, z + lateral * fx)


def _shift_fade(span, start: float, end: float):
    last = max(len(span) - 1, 1)
    for i, (x, y, z, fx, fz) in enumerate(span):
        lateral = start + (end - start) * (i / last)
        yield (x + lateral * (-fz), y, z + lateral * fx)


def _join_laterals(
    mesh: SectorMesh,
    looks: dict[str, RoadLook],
    apply_shift: bool,
) -> dict[int, tuple[list[float], list[float]]]:
    """Laterals at (start, end). The piece that begins at an offset change tapers off the previous look."""
    by_node: dict[int, list] = {}
    own: dict[int, list[float]] = {}
    roads = []
    seen: set[int] = set()
    for road in mesh.roads:
        if road.uid in seen:
            continue
        if mesh.nodes.get(road.start_uid) is None or mesh.nodes.get(road.end_uid) is None:
            continue
        look = looks.get(road.look)
        laterals = look.laterals(apply_shift=apply_shift) if look is not None else None
        if not laterals:
            continue
        seen.add(road.uid)
        roads.append(road)
        own[road.uid] = laterals
        by_node.setdefault(road.start_uid, []).append(road)
        by_node.setdefault(road.end_uid, []).append(road)
    out: dict[int, tuple[list[float], list[float]]] = {}
    for road in roads:
        mine = own[road.uid]
        start_lat = _from_upstream(
            mine, _aligned(mesh, looks, by_node, road, mine, True, apply_shift),
        )
        if start_lat is not mine:
            out[road.uid] = (start_lat, mine)
    return out


def _from_upstream(mine: list[float], group: list) -> list[float]:
    """The one road that arrives at our start, when its lanes differ. Our end keeps ``mine``."""
    arriving = [item for item in group if item[2]]
    if len(arriving) != 1:
        return mine
    _nid, lats, _arrives = arriving[0]
    if _close(lats, mine):
        return mine
    return lats


def _aligned(mesh, looks, by_node, road, mine, at_start: bool, apply_shift: bool) -> list:
    node_uid = road.start_uid if at_start else road.end_uid
    start = mesh.nodes[road.start_uid]
    end = mesh.nodes[road.end_uid]
    ours = _chord(start, end)
    look = looks.get(road.look)
    if ours is None or look is None:
        return []
    found = []
    seen: set[int] = set()
    for other in by_node.get(node_uid, ()):
        if other.uid == road.uid or other.uid in seen:
            continue
        seen.add(other.uid)
        other_look = looks.get(other.look)
        if other_look is None:
            continue
        if (len(other_look.lanes_right), len(other_look.lanes_left)) != (
            len(look.lanes_right), len(look.lanes_left)
        ):
            continue
        other_start = mesh.nodes.get(other.start_uid)
        other_end = mesh.nodes.get(other.end_uid)
        if other_start is None or other_end is None:
            continue
        direction = _chord(other_start, other_end)
        if direction is None or ours[0] * direction[0] + ours[1] * direction[1] < _JOIN_COS:
            continue
        laterals = other_look.laterals(apply_shift=apply_shift)
        if len(laterals) != len(mine):
            continue
        found.append((other.uid, laterals, other.end_uid == node_uid))
    return found


def _chord(start: MapNode, end: MapNode):
    dx = end.x - start.x
    dz = end.z - start.z
    length = math.hypot(dx, dz)
    if length < 1e-3:
        return None
    return dx / length, dz / length


def _close(lats, mine: list[float]) -> bool:
    return len(lats) == len(mine) and all(
        abs(a - b) < _LATERAL_EPS_M for a, b in zip(lats, mine)
    )


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
