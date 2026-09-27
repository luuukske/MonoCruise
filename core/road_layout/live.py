"""Sector cache for the live lane window.

Looks and the prefab index load once. A 4 km sector is parsed the first time
the truck's view can reach it, then kept until the truck has moved on. Lane
lines are rebuilt only when that set of sectors changes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from core.road_layout.archive import HashFs
from core.road_layout.grid import (
    map_folder,
    read_sector_blob,
    sectors_touching,
)
from core.road_layout.lanes import LaneLine, lane_lines
from core.road_layout.looks import RoadLook, load_road_looks
from core.road_layout.prefab import PrefabLibrary, load_prefab_paths
from core.road_layout.sector import SectorError, SectorMesh, parse_sector
from core.sdk_installer.game_paths import find_game_installations

log = logging.getLogger("road_layout")

# The window draws about 200 m. Roads are at most 1 km, so the sector set has
# to include the neighbour that holds the far node.
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


@dataclass(frozen=True)
class MapFrame:
    """Lane lines for the sectors currently held. Empty when nothing is loaded."""

    lines: tuple[MapLine, ...]
    sectors: int


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
        self._looks: dict[str, RoadLook] = load_road_looks(all_archives)
        self._prefabs = PrefabLibrary(all_archives, load_prefab_paths(all_archives))
        self._needed: tuple[tuple[int, int], ...] | None = None
        self._lines: tuple[MapLine, ...] = ()

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
        mesh = SectorMesh()
        for key in needed:
            part = self._sectors.get(key)
            if part is None:
                continue
            mesh.nodes.update(part.nodes)
            mesh.roads.extend(part.roads)
            mesh.prefabs.extend(part.prefabs)
            mesh.sectors += 1
        self._lines = tuple(_pack(line) for line in lane_lines(
            mesh, self._looks, self._prefabs, steps=8,
        ))
        self._needed = needed
        log.info(
            "lane window %s lines %d",
            " ".join(f"{ix:+d},{iz:+d}" for ix, iz in needed),
            len(self._lines),
        )
        return MapFrame(self._lines, mesh.sectors)

    def stale(self, x: float, z: float) -> bool:
        """True when ``frame_for`` would parse sectors."""
        return tuple(sectors_touching(x, z, _LOAD_M)) != self._needed

    def _ensure(self, key: tuple[int, int]) -> None:
        if key in self._sectors:
            return
        ix, iz = key
        try:
            blob = read_sector_blob(self._map_archives, self.map_name, ix, iz)
            self._sectors[key] = parse_or_empty(blob)
        except (OSError, ValueError, SectorError):
            log.warning("sector %+d %+d failed to parse", ix, iz)
            self._sectors[key] = SectorMesh()


def parse_or_empty(blob: bytes | None) -> SectorMesh:
    if not blob:
        return SectorMesh()
    return parse_sector(blob)


def _pack(line: LaneLine) -> MapLine:
    pts = tuple((p[0], p[2]) for p in line.points)
    if not pts:
        return MapLine(line.kind, (), 0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    return MapLine(line.kind, pts, min(xs), max(xs), min(zs), max(zs))
