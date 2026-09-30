"""Sector cache for the live lane window.

Looks and the prefab index load once. A 4 km sector is parsed the first time
the truck's view can reach it, then kept until the truck has moved on. Lane
lines are rebuilt only when that set of sectors changes.
"""

from __future__ import annotations

import colorsys
import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from core.road_layout.archive import HashFs
from core.road_layout.grid import (
    map_folder,
    sectors_touching,
)
from core.road_layout.lanes import LaneLine, lane_lines
from core.road_layout.looks import RoadLook, load_road_looks
from core.road_layout.prefab import PrefabLibrary, load_prefab_paths
from core.road_layout.sector import SectorError, SectorMesh, parse_sector
from core.sdk_installer.game_paths import find_game_installations

log = logging.getLogger("road_layout")

# The window draws about 200 m. Roads are at most 1 km, so the sector set has
# to include the neighbour that holds the far node. Obstacles are drawn to 400 m.
_LOAD_M = 1400.0
_KEEP_M = _LOAD_M + 4000.0


@dataclass(frozen=True)
class MapLine:
    """One lane polyline in world XZ, with a box for culling."""

    kind: str
    pts: tuple[tuple[float, float], ...]
    min_x: float
    max_x: float
    min_z: float
    max_z: float
    prefab_id: int | None = None
    label: str = ""
    color: str = ""


@dataclass(frozen=True)
class MapDot:
    """One placed model in world XZ."""

    kind: str
    x: float
    z: float
    label: str = ""
    color: str = ""


@dataclass(frozen=True)
class MapFrame:
    """Lane lines for the sectors currently held. Empty when nothing is loaded."""

    lines: tuple[MapLine, ...]
    sectors: int
    props: tuple[MapLine, ...] = ()
    dots: tuple[MapDot, ...] = ()


class LiveMap:
    """Opens the game archives and serves lane lines around a moving pose."""

    def __init__(
        self,
        map_name: str,
        map_archives: Sequence[HashFs],
        all_archives: Sequence[HashFs],
    ) -> None:
        self.map_name = map_name
        self._map_archives = list(map_archives)
        self._sectors: dict[tuple[int, int], SectorMesh] = {}
        self._caps: dict[tuple[int, int], object] = {}
        self._looks: dict[str, RoadLook] = load_road_looks(all_archives)
        self._prefabs = PrefabLibrary(all_archives, load_prefab_paths(all_archives))
        self._books: dict = _obstacle_books(all_archives)
        self._needed: tuple[tuple[int, int], ...] | None = None
        self._lines: tuple[MapLine, ...] = ()
        self._props: tuple[MapLine, ...] = ()
        self._dots: tuple[MapDot, ...] = ()

    @classmethod
    def open_installed(cls, game: str = "ets2") -> "LiveMap":
        installs = find_game_installations(game)
        if not installs:
            raise FileNotFoundError(f"no {game} install found")
        return cls.open_dir(installs[0])

    @classmethod
    def open_dir(cls, game_dir: Path) -> "LiveMap":
        opened: list[HashFs] = []
        for path in sorted(Path(game_dir).glob("*.scs")):
            try:
                opened.append(HashFs(path))
            except (OSError, ValueError):
                log.warning("skipped archive %s", path.name)
        map_name = next((name for archive in opened if (name := map_folder(archive))), None)
        if map_name is None:
            raise FileNotFoundError("no map folder in the game archives")
        maps = [
            archive for archive in opened
            if map_folder(archive) == map_name or archive.listdir(f"map/{map_name}") is not None
        ]
        if not maps:
            raise FileNotFoundError("map archives have no sectors")
        return cls(map_name, maps, opened)

    def frame_for(self, x: float, z: float) -> MapFrame | None:
        """Lines for the sectors around ``(x, z)``. None when the set is unchanged."""
        needed = tuple(sectors_touching(x, z, _LOAD_M))
        if needed == self._needed:
            return None
        for key in needed:
            self._ensure(key)
        keep = set(sectors_touching(x, z, _KEEP_M))
        for key in list(self._sectors):
            if key not in keep:
                del self._sectors[key]
                self._caps.pop(key, None)
        mesh = SectorMesh()
        nodes: dict = {}
        caps = []
        for key in needed:
            part = self._sectors.get(key)
            if part is None:
                continue
            mesh.nodes.update(part.nodes)
            nodes.update(part.nodes)
            mesh.roads.extend(part.roads)
            mesh.prefabs.extend(part.prefabs)
            mesh.sectors += 1
            cap = self._caps.get(key)
            if cap is not None:
                caps.append(cap)
        self._lines = tuple(_pack(line) for line in lane_lines(
            mesh, self._looks, self._prefabs, steps=8,
        ))
        self._props, self._dots = _prop_geometry(nodes, caps, self._books)
        self._needed = needed
        log.info(
            "lane window %s lines %d props %d",
            " ".join(f"{ix:+d},{iz:+d}" for ix, iz in needed),
            len(self._lines),
            len(self._props),
        )
        return MapFrame(self._lines, mesh.sectors, self._props, self._dots)

    def stale(self, x: float, z: float) -> bool:
        """True when ``frame_for`` would parse sectors."""
        return tuple(sectors_touching(x, z, _LOAD_M)) != self._needed

    def _ensure(self, key: tuple[int, int]) -> None:
        if key in self._sectors:
            return
        ix, iz = key
        try:
            mesh, cap = _load_obstacles(self._map_archives, self.map_name, ix, iz)
        except Exception:
            log.exception("sector %+d %+d failed to parse", ix, iz)
            mesh, cap = SectorMesh(), None
        self._sectors[key] = mesh
        if cap is not None:
            self._caps[key] = cap


def _obstacle_books(archives: Sequence[HashFs]) -> dict:
    try:
        from tools.map_obstacle_probe import load_catalog
        books, _note = load_catalog(archives)
        return books
    except (OSError, ValueError):
        log.warning("obstacle catalog failed")
        return {}


def _load_obstacles(archives, map_name: str, ix: int, iz: int):
    from tools.map_obstacle_probe import _load_sector
    mesh, cap, _times, note = _load_sector(archives, map_name, ix, iz)
    if "failed" in note:
        log.warning("obstacle sector %+d %+d: %s", ix, iz, note)
    return mesh, cap


def _prop_geometry(nodes: dict, caps: list, books: dict) -> tuple[tuple[MapLine, ...], tuple[MapDot, ...]]:
    if not caps or not books:
        return (), ()
    try:
        from tools.map_obstacle_probe import geometry_for, is_clutter
        lines, dots = geometry_for(nodes, caps, books)
    except Exception:
        log.exception("obstacle geometry failed")
        return (), ()
    kept_lines: list[tuple[str, list, str]] = []
    kept_dots: list[tuple[str, float, float, str]] = []
    labels: list[str] = []
    for row in lines:
        kind, pts, label = row[0], row[1], row[2] if len(row) > 2 else ""
        if kind == "road" or len(pts) < 2 or is_clutter(label):
            continue
        kept_lines.append((kind, pts, label))
        labels.append(label or kind)
    for row in dots:
        kind, x, z = row[0], row[1], row[2]
        label = row[4] if len(row) > 4 else ""
        if is_clutter(label):
            continue
        kept_dots.append((kind, x, z, label))
        labels.append(label or kind)
    colors = _assign_colors(labels)
    props: list[MapLine] = []
    marks: list[MapDot] = []
    for kind, pts, label in kept_lines:
        key = label or kind
        packed = _pack_xz(kind, pts, key, colors[key])
        if packed is not None:
            props.append(packed)
    for kind, x, z, label in kept_dots:
        key = label or kind
        marks.append(MapDot(kind, x, z, key, colors[key]))
    return tuple(props), tuple(marks)


def _assign_colors(labels: list[str]) -> dict[str, str]:
    """One bright hex per model path. The same path keeps its color."""
    taken: set[str] = set()
    out: dict[str, str] = {}
    for label in sorted(set(labels)):
        digest = hashlib.sha1(label.encode("utf-8")).digest()
        hue = int.from_bytes(digest[:2], "big") / 65535.0
        sat = 0.55 + 0.4 * (digest[2] / 255.0)
        val = 0.62 + 0.35 * (digest[3] / 255.0)
        red, green, blue = colorsys.hsv_to_rgb(hue, sat, val)
        number = (int(red * 255) << 16) | (int(green * 255) << 8) | int(blue * 255)
        color = f"#{number:06X}"
        while color in taken or (number & 255) + ((number >> 8) & 255) + ((number >> 16) & 255) < 280:
            number = (number + 0x1F3A5B) & 0xFFFFFF
            color = f"#{number:06X}"
        taken.add(color)
        out[label] = color
    return out


def parse_or_empty(blob: bytes | None) -> SectorMesh:
    if not blob:
        return SectorMesh()
    return parse_sector(blob)


def _pack_xz(kind: str, pts: list[tuple[float, float]], label: str, color: str) -> MapLine | None:
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    return MapLine(kind, tuple(pts), min(xs), max(xs), min(zs), max(zs), None, label, color)


def _pack(line: LaneLine) -> MapLine:
    pts = tuple((p[0], p[2]) for p in line.points)
    if not pts:
        return MapLine(line.kind, (), 0.0, 0.0, 0.0, 0.0, line.prefab_id)
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    return MapLine(line.kind, pts, min(xs), max(xs), min(zs), max(zs), line.prefab_id)
