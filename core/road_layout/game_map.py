"""One installed game's map: archives, road looks, prefab curves, sectors on demand.

Shared by the runtime map worker and the offline tools. Opening reads every
``.scs`` index (about 1.5 s on ETS2 with its DLCs), so ``GameMapOpener`` does
it one archive per step for a caller that must not block.
"""

from __future__ import annotations

import logging
import math
import struct
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from core.road_layout.archive import Archive, HashFs
from core.road_layout.ego_lane import LineIndex
from core.road_layout.buried import drop_buried
from core.road_layout.grid import map_folder, read_sector_blob, sector_index, sector_key
from core.road_layout.lanes import LaneLine, lane_lines
from core.road_layout.looks import RoadLook, load_road_looks
from core.road_layout.prefab import PrefabLibrary, load_prefab_paths
from core.road_layout.sector import SectorError, SectorMesh, parse_sector
from core.sdk_installer.game_paths import find_game_installations

log = logging.getLogger("road_layout")

# A road is at most about 1 km, so its far node can sit one sector over.
_ROAD_REACH_M = 1100.0
_SECTOR_CACHE = 24


@dataclass(frozen=True)
class MapLane:
    """One lane in world XZ with its heights and a box for culling."""

    kind: str
    pts: tuple[tuple[float, float], ...]
    ys: tuple[float, ...]
    min_x: float
    max_x: float
    min_z: float
    max_z: float
    prefab_id: int | None
    uid: int
    lane_index: int
    lane_count: int
    lateral_m: float
    # +1 traffic runs in point order, -1 against, 0 unknown (LaneLine.travel).
    travel: int = 0


@dataclass(frozen=True)
class LaneWindow:
    """Lanes whose geometry reaches a world box. ``sectors`` were parsed for it."""

    lines: tuple[LaneLine, ...]
    sectors: tuple[tuple[int, int], ...]
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    lanes: tuple[MapLane, ...] = ()
    index: LineIndex | None = None


class GameMap:
    """Archives of one game, with parsed sectors kept in a small LRU."""

    def __init__(self, map_name: str, map_archives: Sequence[Archive], all_archives: Sequence[Archive]) -> None:
        self.map_name = map_name
        self._map_archives = list(map_archives)
        self._all_archives = list(all_archives)
        self._looks: dict[str, RoadLook] = load_road_looks(all_archives)
        self._prefab_paths = load_prefab_paths(all_archives)
        self._prefabs = PrefabLibrary(all_archives, self._prefab_paths)
        self._sectors: OrderedDict[tuple[int, int], SectorMesh] = OrderedDict()
        self.failed_sectors: set[tuple[int, int]] = set()
        self._base_map: GameMap | None = None
        self._overlay_archives: list[Archive] = []
        self._added: set[int] = set()

    @classmethod
    def open_installed(cls, game: str = "ets2") -> "GameMap":
        installs = find_game_installations(game)
        if not installs:
            raise FileNotFoundError(f"no {game} install found")
        return cls.open_dir(installs[0])

    @classmethod
    def open_dir(cls, game_dir: Path) -> "GameMap":
        opener = GameMapOpener(game_dir)
        while not opener.step():
            pass
        if opener.result is None:
            raise FileNotFoundError(opener.error or "map did not open")
        return opener.result

    def with_overlay(self, archives: Sequence[Archive]) -> "GameMap":
        """This map with *archives* layered on top: later archives replace sectors, looks and prefabs."""
        overlay = object.__new__(GameMap)
        overlay.map_name = self.map_name
        overlay._map_archives = self._map_archives + [
            a for a in archives if a.listdir(f"map/{self.map_name}") is not None
        ]
        overlay._all_archives = self._all_archives + list(archives)
        overlay._looks = {**self._looks, **load_road_looks(archives)}
        overlay._prefab_paths = {**self._prefab_paths, **load_prefab_paths(archives)}
        overlay._prefabs = PrefabLibrary(overlay._all_archives, overlay._prefab_paths)
        overlay._sectors = OrderedDict()
        overlay.failed_sectors = set()
        overlay._base_map = self
        overlay._overlay_archives = list(archives)
        overlay._added = set()
        return overlay

    def sector(self, key: tuple[int, int]) -> SectorMesh:
        """Parsed sector, empty when absent or unreadable. Failures are logged once."""
        cached = self._sectors.get(key)
        if cached is not None:
            self._sectors.move_to_end(key)
            return cached
        ix, iz = key
        mesh = SectorMesh()
        try:
            blob = read_sector_blob(self._map_archives, self.map_name, ix, iz)
            if blob:
                mesh = parse_sector(blob)
        except (SectorError, ValueError, OSError, struct.error, IndexError, OverflowError):
            if key not in self.failed_sectors:
                log.warning("map sector %+d %+d unreadable, no lanes there", ix, iz)
            self.failed_sectors.add(key)
            mesh = SectorMesh()
        if self._base_map is not None and mesh.sectors:
            self._note_added(key, mesh)
        self._sectors[key] = mesh
        while len(self._sectors) > _SECTOR_CACHE:
            self._sectors.popitem(last=False)
        return mesh

    def _note_added(self, key: tuple[int, int], mesh: SectorMesh) -> None:
        """Remember the items a TruckersMP sector has that the base game's version of it lacks."""
        sector = sector_key(self.map_name, *key)
        if not any(a.has(sector) for a in self._overlay_archives):
            return
        base = self._base_map.sector(key)
        known = {r.uid for r in base.roads} | {p.uid for p in base.prefabs}
        self._added |= ({r.uid for r in mesh.roads} | {p.uid for p in mesh.prefabs}) - known

    def sectors_needed(self, min_x: float, max_x: float, min_z: float, max_z: float) -> list[tuple[int, int]]:
        """Sectors ``lanes_in_box`` would parse for this box that are not cached yet."""
        keys = _sectors_for_box(min_x - _ROAD_REACH_M, max_x + _ROAD_REACH_M,
                                min_z - _ROAD_REACH_M, max_z + _ROAD_REACH_M)
        return [key for key in keys if key not in self._sectors]

    def lanes_in_box(
        self,
        min_x: float,
        max_x: float,
        min_z: float,
        max_z: float,
        *,
        steps: int = 8,
        road_tangent: str = "stored",
        prefab_tangent: str = "chord",
    ) -> LaneWindow:
        """Lanes of every road or prefab that can reach the box."""
        keys = _sectors_for_box(min_x - _ROAD_REACH_M, max_x + _ROAD_REACH_M,
                                min_z - _ROAD_REACH_M, max_z + _ROAD_REACH_M)
        merged = SectorMesh()
        for key in keys:
            part = self.sector(key)
            merged.nodes.update(part.nodes)
            merged.sectors += 1
        for key in keys:
            part = self.sector(key)
            for road in part.roads:
                if _road_reaches(merged.nodes, road, min_x, max_x, min_z, max_z):
                    merged.roads.append(road)
            for item in part.prefabs:
                if _prefab_reaches(merged.nodes, item, min_x, max_x, min_z, max_z):
                    merged.prefabs.append(item)
        lines = lane_lines(
            merged, self._looks, self._prefabs, steps,
            road_tangent=road_tangent, prefab_tangent=prefab_tangent,
        )
        lanes = tuple(lane for line in lines if (lane := pack_lane(line)) is not None)
        if self._added:
            lanes = drop_buried(lanes, self._added)
        return LaneWindow(
            tuple(lines), tuple(keys), (min_x, max_x, min_z, max_z), lanes, LineIndex(lanes),
        )


class GameMapOpener:
    """Opens a game's archives one per ``step``. ``result`` is set when done."""

    def __init__(self, game_dir: Path) -> None:
        self._pending = sorted(Path(game_dir).glob("*.scs"))
        self._opened: list[HashFs] = []
        self._folders: list[str | None] = []
        self.result: GameMap | None = None
        self.error: str = ""
        self.done = False

    def step(self) -> bool:
        """Open one archive, or finish. True once nothing is left to do."""
        if self.done:
            return True
        if self._pending:
            path = self._pending.pop(0)
            try:
                archive = HashFs(path)
                self._opened.append(archive)
                self._folders.append(map_folder(archive))
            except (OSError, ValueError):
                log.warning("map archive %s skipped", path.name)
            return False
        self.done = True
        map_name = next((name for name in self._folders if name), None)
        if map_name is None:
            self.error = "no map folder in the game archives"
            return True
        maps = [
            a for a, name in zip(self._opened, self._folders)
            if name == map_name or a.listdir(f"map/{map_name}") is not None
        ]
        if not maps:
            self.error = "map archives have no sectors"
            return True
        self.result = GameMap(map_name, maps, self._opened)
        return True


def pack_lane(line: LaneLine) -> MapLane | None:
    if len(line.points) < 2:
        return None
    pts = tuple((p[0], p[2]) for p in line.points)
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    return MapLane(
        line.kind, pts, tuple(p[1] for p in line.points),
        min(xs), max(xs), min(zs), max(zs),
        line.prefab_id, line.uid, line.lane_index, line.lane_count, line.lateral_m,
        line.travel,
    )


def _sectors_for_box(min_x: float, max_x: float, min_z: float, max_z: float) -> list[tuple[int, int]]:
    ix0, iz0 = sector_index(min_x, min_z)
    ix1, iz1 = sector_index(max_x, max_z)
    return [(ix, iz) for iz in range(iz0, iz1 + 1) for ix in range(ix0, ix1 + 1)]


def _gap_to_box(x: float, z: float, min_x: float, max_x: float, min_z: float, max_z: float) -> float:
    dx = max(min_x - x, 0.0, x - max_x)
    dz = max(min_z - z, 0.0, z - max_z)
    return math.hypot(dx, dz)


def _road_reaches(nodes, road, min_x, max_x, min_z, max_z) -> bool:
    start = nodes.get(road.start_uid)
    end = nodes.get(road.end_uid)
    if start is None or end is None:
        return False
    reach = 0.5 * max(road.length, math.hypot(end.x - start.x, end.z - start.z)) + 20.0
    gap = min(
        _gap_to_box(start.x, start.z, min_x, max_x, min_z, max_z),
        _gap_to_box(end.x, end.z, min_x, max_x, min_z, max_z),
    )
    return gap <= reach


def _prefab_reaches(nodes, item, min_x, max_x, min_z, max_z) -> bool:
    for uid in item.node_uids:
        node = nodes.get(uid)
        if node is not None and _gap_to_box(node.x, node.z, min_x, max_x, min_z, max_z) <= 250.0:
            return True
    return False
