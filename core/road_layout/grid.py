"""Which 4 km sector a world position falls in, and how to load it.

Telemetry X and Z select the square with ``floor(coord / 4000)``. The file
is ``map/<map>/sec+0000-0002.base`` inside a HashFS archive. A later archive
replaces the same path, which is how a map DLC overrides the base sector.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from core.road_layout.archive import HashFs
from core.road_layout.sector import SectorMesh, parse_sector

SECTOR_M = 4000


def sector_index(x: float, z: float) -> tuple[int, int]:
    """Sector column and row for a telemetry position. Y is not a sector axis."""
    return math.floor(x / SECTOR_M), math.floor(z / SECTOR_M)


def sector_filename(ix: int, iz: int) -> str:
    """``sec+0003-0001.base``. Signs are part of the four-digit field."""
    return f"sec{ix:+05d}{iz:+05d}.base"


def sector_key(map_name: str, ix: int, iz: int) -> str:
    return f"map/{map_name}/{sector_filename(ix, iz)}"


def neighborhood(ix: int, iz: int, radius: int = 1) -> list[tuple[int, int]]:
    """The sector and the squares around it, near to far in row-major order."""
    span = range(-radius, radius + 1)
    return [(ix + dx, iz + dz) for dz in span for dx in span]


def sectors_touching(x: float, z: float, radius_m: float) -> list[tuple[int, int]]:
    """Sectors whose square meets the circle of ``radius_m`` around ``(x, z)``."""
    reach = max(0.0, radius_m)
    ix0, iz0 = sector_index(x - reach, z - reach)
    ix1, iz1 = sector_index(x + reach, z + reach)
    return [(ix, iz) for iz in range(iz0, iz1 + 1) for ix in range(ix0, ix1 + 1)]


def map_folder(archive: HashFs) -> str | None:
    """``europe`` or ``usa`` from the single ``.mbd`` under ``map/``, if present."""
    listing = archive.listdir("map")
    if listing is None:
        return None
    _subs, files = listing
    names = [name[:-4] for name in files if name.endswith(".mbd")]
    if len(names) != 1:
        return None
    return names[0]


def read_sector_blob(
    archives: Sequence[HashFs],
    map_name: str,
    ix: int,
    iz: int,
) -> bytes | None:
    """Sector bytes. The last archive that has the path wins."""
    key = sector_key(map_name, ix, iz)
    found: bytes | None = None
    for archive in archives:
        blob = archive.get(key)
        if blob is not None:
            found = blob
    return found


def load_sectors(
    archives: Sequence[HashFs],
    map_name: str,
    indices: Iterable[tuple[int, int]],
) -> SectorMesh:
    """Merge the given sectors. Nodes from a later sector replace the same uid."""
    mesh = SectorMesh()
    for ix, iz in indices:
        blob = read_sector_blob(archives, map_name, ix, iz)
        if blob is None:
            continue
        part = parse_sector(blob)
        mesh.version = part.version
        mesh.nodes.update(part.nodes)
        mesh.roads.extend(part.roads)
        mesh.prefabs.extend(part.prefabs)
        mesh.sectors += 1
    return mesh
