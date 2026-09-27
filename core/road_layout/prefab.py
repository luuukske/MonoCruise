"""Prefab driving curves from ``.ppd``, placed on the sector's origin node.

Each nav curve is one AI lane through a junction, roundabout or merge.
The curve is a Hermite segment in the prefab's own frame. The sector stores
which world node the prefab's origin node sits on, and the curve is rotated
by the difference of those two yaws.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass

from core.road_layout.archive import HashFs
from core.road_layout.looks import _def_texts, _units
from core.road_layout.sector import MapNode, MapPrefab

_PPD_VERSION = 25
_HEADER_COUNTS = 11
_HEADER_PTRS = 12
_NODE_SIZE = 104
_CURVE_SIZE = 132
_CURVE = struct.Struct("<Q I 4B 3f 3f 4f 4f f 4i 4i I I i Q I")


@dataclass(frozen=True)
class LocalCurve:
    """One nav curve in prefab space. X east, Y up, Z south."""

    x0: float
    y0: float
    z0: float
    yaw0: float
    x1: float
    y1: float
    z1: float
    yaw1: float
    start_node: int
    start_lane: int
    end_node: int
    end_lane: int


@dataclass(frozen=True)
class PrefabNode:
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class PrefabDesc:
    """Prefab-local nodes and every driving curve."""

    nodes: tuple[PrefabNode, ...]
    curves: tuple[LocalCurve, ...]


def load_prefab_paths(archives: Sequence[HashFs]) -> dict[str, str]:
    """Map a prefab unit token to its ``.ppd`` path, without the leading slash."""
    paths: dict[str, str] = {}
    for text in _def_texts(archives, "prefab"):
        for kind, name, body in _units(text):
            if kind != "prefab_model":
                continue
            desc = _field(body, "prefab_desc")
            if not desc:
                continue
            path = desc.lstrip("/")
            paths[name] = path
            alias = name.split(".", 1)[1] if "." in name else ""
            if alias:
                paths[alias] = path
    return paths


class PrefabLibrary:
    """Parsed ``.ppd`` files, keyed by archive path. Missing files stay missing."""

    def __init__(self, archives: Sequence[HashFs], paths: dict[str, str]) -> None:
        self._archives = archives
        self._paths = paths
        self._cache: dict[str, PrefabDesc | None] = {}

    def curves_for(self, prefab: MapPrefab) -> PrefabDesc | None:
        path = self._paths.get(prefab.model)
        if not path:
            return None
        if path not in self._cache:
            self._cache[path] = _read_ppd(_find_blob(self._archives, path))
        return self._cache[path]


def place_curve(
    curve: LocalCurve,
    world: MapNode,
    desc: PrefabDesc,
    origin_index: int,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None:
    """Start and end as ``(x, y, z, yaw)`` in the telemetry frame."""
    if origin_index < 0 or origin_index >= len(desc.nodes):
        return None
    origin = desc.nodes[origin_index]
    delta = _yaw(world.qw, world.qy) - origin.yaw
    y_shift = world.y - origin.y

    def _end(lx: float, ly: float, lz: float, yaw: float):
        dx = lx - origin.x
        dz = lz - origin.z
        cos = math.cos(delta)
        sin = math.sin(delta)
        return (
            world.x + cos * dx - sin * dz,
            ly + y_shift,
            world.z + sin * dx + cos * dz,
            yaw + delta,
        )

    return _end(curve.x0, curve.y0, curve.z0, curve.yaw0), _end(curve.x1, curve.y1, curve.z1, curve.yaw1)


def _yaw(qw: float, qy: float) -> float:
    return math.atan2(-qy, qw) * 2.0 - math.pi / 2.0


def _read_ppd(blob: bytes | None) -> PrefabDesc | None:
    if blob is None or len(blob) < 4 + _HEADER_COUNTS * 4 + _HEADER_PTRS * 4:
        return None
    version = struct.unpack_from("<I", blob, 0)[0]
    if version != _PPD_VERSION:
        return None
    counts = struct.unpack_from(f"<{_HEADER_COUNTS}I", blob, 4)
    ptrs = struct.unpack_from(f"<{_HEADER_PTRS}I", blob, 4 + _HEADER_COUNTS * 4)
    node_count, curve_count = counts[0], counts[1]
    node_at, curve_at = ptrs[0], ptrs[1]
    nodes: list[PrefabNode] = []
    for index in range(node_count):
        node = _node_at(blob, node_at, index)
        if node is None:
            return None
        nodes.append(node)
    curves: list[LocalCurve] = []
    for index in range(curve_count):
        curve = _curve_at(blob, curve_at, index)
        if curve is None:
            return None
        curves.append(curve)
    return PrefabDesc(tuple(nodes), tuple(curves))


def _node_at(blob: bytes, base: int, index: int) -> PrefabNode | None:
    at = base + index * _NODE_SIZE
    if at + _NODE_SIZE > len(blob):
        return None
    px, py, pz = struct.unpack_from("<3f", blob, at + 16)
    rx, _ry, rz = struct.unpack_from("<3f", blob, at + 28)
    return PrefabNode(px, py, pz, math.atan2(rz, rx))


def _curve_at(blob: bytes, base: int, index: int) -> LocalCurve | None:
    at = base + index * _CURVE_SIZE
    if at + _CURVE_SIZE > len(blob):
        return None
    if _CURVE.size != _CURVE_SIZE:
        return None
    parts = _CURVE.unpack_from(blob, at)
    # token, flags, 4 leads, start xyz, end xyz, start quat, end quat
    end_node, end_lane, start_node, start_lane = parts[2:6]
    sx, sy, sz = parts[6:9]
    ex, ey, ez = parts[9:12]
    sqw, _sqx, sqy, _sqz = parts[12:16]
    eqw, _eqx, eqy, _eqz = parts[16:20]
    return LocalCurve(
        sx, sy, sz, _yaw(sqw, sqy), ex, ey, ez, _yaw(eqw, eqy),
        start_node, start_lane, end_node, end_lane,
    )


def _find_blob(archives: Sequence[HashFs], key: str) -> bytes | None:
    found: bytes | None = None
    for archive in archives:
        try:
            blob = archive.get(key)
        except ValueError:
            continue
        if blob is not None:
            found = blob
    return found


def _field(body: str, key: str) -> str:
    for raw in body.splitlines():
        line = raw.strip()
        if not line.startswith(key + ":"):
            continue
        return line.split(":", 1)[1].strip().strip('"')
    return ""
