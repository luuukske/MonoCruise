"""How much non-vehicle map geometry is sitting in the sector files.

The lane reader already walks every item in a `.base` sector and skips the
ones that are not a road or a prefab. Models, signs, curves and buildings
live in the matching `.aux`. Roadside models, railings and vegetation bands
are on the road itself: railings in the `.base` record, the rest in `.data`.

Vegetation is a density band, not a list of trees. This probe times a sector
the way the live lane window loads one, and can draw one window to a PNG.

    python tools/map_obstacle_probe.py
    python tools/map_obstacle_probe.py --x 3296 --z 19626 --scan 4
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.road_layout.sector as sector
from core.road_layout.archive import HashFs
from core.road_layout.centerline import node_forward_xz, sample_span
from core.road_layout.grid import map_folder, sector_index
from core.road_layout.looks import _units
from core.road_layout.sector import (
    MapNode,
    MapPrefab,
    MapRoad,
    SectorError,
    SectorMesh,
    _Cur,
    _quad,
    parse_sector,
)
from core.sdk_installer.game_paths import find_game_installations

_EOF = 0xFFFFFFFFFFFFFFFF
_CELL = 40.0

_TYPE_NAMES = {
    1: "terrain",
    2: "building",
    3: "road",
    4: "prefab",
    5: "model",
    6: "company",
    9: "mover",
    12: "city",
    34: "trigger",
    36: "sign",
    39: "bezier",
    40: "compound",
    43: "far_model",
    44: "curve",
}

_DEF_STEMS = (
    "model", "sign", "railing", "veget", "curve", "building", "edge", "barrier", "fence",
)

_DRAW = {
    "rail": (255, 150, 40),
    "tree": (70, 190, 90),
    "hedge": (40, 140, 70),
    "sign": (240, 220, 70),
    "pillar": (230, 70, 70),
    "lamp": (170, 170, 240),
    "building": (150, 150, 170),
    "edge": (200, 120, 80),
    "other": (210, 80, 190),
    "road": (90, 90, 110),
}


@dataclass
class DefInfo:
    path: str
    collision: bool | None


# Same short token can exist as a sign and as a model. The item type picks the book.
_BOOKS = {
    "model": ("model_def", "far_model_def"),
    "sign": ("sign_model", "sign_board_model", "sign_stand_model"),
    "curve": ("curve_model",),
    "building": ("building_scheme", "building_model"),
    "railing": ("railing_model",),
    "roadside": ("sign_model", "model_def"),
}


@dataclass
class Railing:
    side: str
    name: str
    offset_m: float


@dataclass
class RoadModel:
    side: str
    name: str
    offset_m: float
    distance_m: float


@dataclass
class VegBand:
    side: str
    name: str
    density: float
    from_m: float
    to_m: float
    collision: bool


@dataclass
class RoadExtra:
    uid: int
    start_uid: int
    end_uid: int
    length: float
    nested: bool
    right_veg_collision: bool
    left_veg_collision: bool
    railings: list[Railing] = field(default_factory=list)
    models: list[RoadModel] = field(default_factory=list)
    veg: list[VegBand] = field(default_factory=list)
    center_veg: str = ""
    center_density: float = 0.0


@dataclass(frozen=True)
class CurvePart:
    """One model repeated along a curve. X offsets are sideways, in metres."""

    name: str
    off0: float
    off1: float


@dataclass
class PointItem:
    kind: str
    name: str
    node_uid: int
    node_uid2: int
    collision: bool | None
    x: float = 0.0
    z: float = 0.0
    x2: float = 0.0
    z2: float = 0.0
    placed: bool = False
    segment: bool = False
    locators: tuple[int, ...] = ()
    parts: tuple[CurvePart, ...] = ()


@dataclass
class Capture:
    order: list[tuple[str, int, int]]
    roads: dict[int, RoadExtra]
    points: list[PointItem]
    counts: Counter
    nested: bool = False
    uid: int = 0
    flags: int = 0
    record_order: bool = False


_CAP: Capture | None = None
_ORIG: dict = {}


def _install(cap: Capture) -> None:
    global _CAP
    _CAP = cap
    if _ORIG:
        return
    _ORIG["header"] = sector._header
    _ORIG["road"] = sector._read_road
    _ORIG["prefab"] = sector._read_prefab
    _ORIG["item"] = sector._read_item
    _ORIG["skip"] = dict(sector._SKIP)
    sector._header = _header
    sector._read_road = _read_road
    sector._read_prefab = _read_prefab
    sector._read_item = _read_item
    sector._SKIP[sector._MODEL] = _read_model
    sector._SKIP[sector._SIGN] = _read_sign
    sector._SKIP[sector._CURVE] = _read_curve
    sector._SKIP[sector._BUILDING] = _read_building


def _restore() -> None:
    global _CAP
    _CAP = None
    if not _ORIG:
        return
    sector._header = _ORIG["header"]
    sector._read_road = _ORIG["road"]
    sector._read_prefab = _ORIG["prefab"]
    sector._read_item = _ORIG["item"]
    sector._SKIP.clear()
    sector._SKIP.update(_ORIG["skip"])
    _ORIG.clear()


def _header(c: _Cur) -> int:
    uid = c.u64()
    c.skip(40)
    flags = c.u32()
    c.u8()
    cap = _CAP
    if cap is not None:
        cap.uid = uid
        cap.flags = flags
    return uid


def _read_item(c: _Cur, mesh: SectorMesh, nested: bool) -> None:
    cap = _CAP
    if cap is not None and c.i + 4 <= c.n:
        typ = int.from_bytes(c.b[c.i:c.i + 4], "little")
        cap.counts[typ] += 1
    prev = False if cap is None else cap.nested
    if cap is not None:
        cap.nested = nested or prev
    try:
        _ORIG["item"](c, mesh, nested)
    finally:
        if cap is not None:
            cap.nested = prev


def _read_road(c: _Cur, uid: int) -> MapRoad:
    cap = _CAP
    flags = 0 if cap is None else cap.flags
    c.u8()
    c.u8()
    c.u16()
    look = c.token()
    for _ in range(8):
        c.token()
    c.token()
    c.f32()
    c.token()
    c.f32()
    c.token()
    c.token()
    c.token()
    railings: list[Railing] = []
    for _ in range(3):
        right_name = c.token()
        right_off = c.i16() / 100.0
        left_name = c.token()
        left_off = c.i16() / 100.0
        if right_name:
            railings.append(Railing("right", right_name, right_off))
        if left_name:
            railings.append(Railing("left", left_name, left_off))
    c.i32()
    c.i32()
    start = c.u64()
    end = c.u64()
    length = c.f32()
    if cap is not None:
        kflag2 = (flags >> 8) & 0xFF
        extra = RoadExtra(
            uid, start, end, length, cap.nested,
            bool(kflag2 & (1 << 4)), bool(kflag2 & (1 << 5)), railings,
        )
        cap.roads[uid] = extra
        if cap.record_order and not cap.nested:
            cap.order.append(("road", uid, 0))
    return MapRoad(uid, look, start, end, length)


def _read_prefab(c: _Cur, uid: int) -> MapPrefab:
    item = _ORIG["prefab"](c, uid)
    cap = _CAP
    if cap is not None and cap.record_order and not cap.nested:
        cap.order.append(("prefab", uid, len(item.node_uids)))
    return item


def _read_model(c: _Cur) -> None:
    cap = _CAP
    name = c.token()
    c.token()
    c.token()
    for _ in range(c.u32()):
        c.token()
    node = c.u64()
    c.skip(12)
    c.token()
    c.u32()
    c.f32()
    if cap is None:
        return
    # Model collision is stored inverted: bit 13 set means collision off.
    collision = not bool(cap.flags & (1 << 13))
    cap.points.append(PointItem("model", name, node, 0, collision))


def _read_sign(c: _Cur) -> None:
    cap = _CAP
    name = c.token()
    node = c.u64()
    c.token()
    c.token()
    for _ in range(c.u8()):
        c.token()
        c.token()
        c.token()
    template = c.u64_str()
    if template:
        _skip_sign_tail(c)
    if cap is None:
        return
    collision = not bool(cap.flags & (1 << 30))
    cap.points.append(PointItem("sign", name, node, 0, collision))


def _skip_sign_tail(c: _Cur) -> None:
    for _ in range(c.u32()):
        c.token()
        flags = c.u8()
        if flags & 0x01:
            c.i8()
            c.i8()
        if flags & 0x02:
            c.token()
    for _ in range(c.u32()):
        c.u32()
        c.token()
        for _attr in range(c.u32()):
            version = c.u16()
            c.u32()
            if version == 1:
                c.i8()
            elif version == 2:
                c.i32()
            elif version == 3:
                c.u32()
            elif version == 4:
                c.f32()
            elif version == 5:
                c.u64_str()
            elif version == 6:
                c.u64()
            else:
                raise SectorError(f"sign attribute {version} at {c.i}")


def _read_curve(c: _Cur) -> None:
    cap = _CAP
    start = c.u64()
    end = c.u64()
    locators: list[int] = []
    for _ in range(2):
        uid = c.u64()
        if uid:
            locators.append(uid)
    c.f32()
    parts: list[CurvePart] = []
    for _ in range(c.u32().bit_count()):
        name = c.token()
        c.u32()
        c.u32()
        c.f32()
        c.f32()
        c.f32()
        c.token()
        c.u32()
        c.f32()
        c.token()
        c.token()
        c.token()
        c.token()
        c.skip(4 * c.u32())
        c.f32()
        off0 = c.f32()
        c.f32()
        off1 = c.f32()
        c.f32()
        if name:
            parts.append(CurvePart(name, off0, off1))
    if cap is None or not parts:
        return
    cap.points.append(PointItem(
        "curve", parts[0].name, start, end, None,
        segment=True, locators=tuple(locators), parts=tuple(parts),
    ))


def _read_building(c: _Cur) -> None:
    cap = _CAP
    name = c.token()
    c.token()
    start = c.u64()
    end = c.u64()
    c.f32()
    c.u32()
    c.f32()
    c.skip(4 * c.u32())
    if cap is not None and name:
        cap.points.append(PointItem("building", name, start, end, None, segment=True))


def _road_veg(c: _Cur) -> tuple[str, float, float, float]:
    name = c.token()
    density = c.u16() / 10.0
    c.u8()
    c.u8()
    start = c.u16() / 10.0
    end = c.u16() / 10.0
    return name, density, start, end


def _read_road_payload(c: _Cur, road: RoadExtra) -> None:
    c.token()
    for side, collides in (
        ("right", road.right_veg_collision),
        ("left", road.left_veg_collision),
    ):
        for _slot in range(2):
            name = c.token()
            offset = c.i16() / 100.0
            distance = c.u16() / 100.0
            if name:
                road.models.append(RoadModel(side, name, offset, distance))
        c.u16()
        for _slot in range(3):
            name, density, start, end = _road_veg(c)
            if name and density > 0 and end > start:
                road.veg.append(VegBand(side, name, density, start, end, collides))
        c.token()
        _quad(c)
    c.token()
    c.u32()
    c.u16()
    c.u32()
    c.f32()
    center = c.token()
    density = c.u16() / 10.0
    c.u8()
    c.u8()
    if center and density > 0:
        road.center_veg = center
        road.center_density = density
    for _ in range(4):
        c.u16()
    c.skip(20 * c.u32())
    for _side in range(2):
        c.skip(8 * c.u32())
    for _side in range(2):
        for _item in range(c.u32()):
            c.u16()
            c.u16()
            c.token()
            c.token()
    for _side in range(2):
        for _item in range(c.u32()):
            c.u16()
            c.u16()
            c.token()
    for _ in range(4):
        c.token()


def _skip_prefab_payload(c: _Cur, nodes: int) -> None:
    # Every prefab stores six corners in the data file, not one per connected node.
    del nodes
    c.token()
    for _ in range(6):
        c.u16()
        c.u16()
        c.u16()
        for _veg in range(2):
            _road_veg(c)
    for _part in range(c.u32()):
        c.token()
        for _veg in range(2):
            c.token()
            c.u16()
            c.u16()
    c.u32()
    c.skip(20 * c.u32())
    for _ in range(6):
        c.token()
        c.token()
        c.token()
    for _ in range(6):
        _quad(c)


def _read_data(blob: bytes, cap: Capture) -> str:
    c = _Cur(blob)
    c.u32()
    c.token()
    c.u32()
    index = 0
    while c.i + 8 <= c.n:
        uid = c.u64()
        if uid == _EOF:
            break
        if index >= len(cap.order):
            return f"data has extra uid at payload {index}, {c.n - c.i} bytes left"
        kind, expect, nodes = cap.order[index]
        if uid != expect:
            return (
                f"data payload {index} uid {uid} != {kind} {expect} "
                f"({len(cap.order)} expected)"
            )
        started_at = c.i
        try:
            if kind == "road":
                _read_road_payload(c, cap.roads[uid])
            else:
                _skip_prefab_payload(c, nodes)
        except SectorError as exc:
            return (
                f"data failed on {kind} {index}/{len(cap.order)} "
                f"uid {uid} nodes {nodes} at {started_at}: {exc} "
                f"file {c.n}"
            )
        index += 1
    trailing = c.n - c.i
    if index != len(cap.order):
        return f"data ended at payload {index} of {len(cap.order)}, {trailing} trailing"
    if trailing:
        return f"payloads {index} ok, {trailing} trailing bytes"
    return f"payloads {index} ok"


def _open_map(game: Path) -> tuple[str, list[HashFs], list[HashFs]]:
    opened: list[HashFs] = []
    for path in sorted(game.glob("*.scs")):
        try:
            opened.append(HashFs(path))
        except (OSError, ValueError):
            continue
    map_name = next((name for archive in opened if (name := map_folder(archive))), None)
    if map_name is None:
        raise SystemExit("no map folder in the game archives")
    maps = [
        archive for archive in opened
        if map_folder(archive) == map_name or archive.listdir(f"map/{map_name}") is not None
    ]
    if not maps:
        raise SystemExit("map archives have no sectors")
    return map_name, maps, opened


def _blob(archives: list[HashFs], map_name: str, ix: int, iz: int, ext: str) -> bytes | None:
    key = f"map/{map_name}/sec{ix:+05d}{iz:+05d}.{ext}"
    found: bytes | None = None
    for archive in archives:
        blob = archive.get(key)
        if blob is not None:
            found = blob
    return found


def _parse_file(blob: bytes, cap: Capture, *, order: bool) -> tuple[SectorMesh, float]:
    cap.record_order = order
    cap.counts = Counter()
    started = time.perf_counter()
    mesh = parse_sector(blob)
    return mesh, (time.perf_counter() - started) * 1000.0


def _load_sector(
    archives: list[HashFs], map_name: str, ix: int, iz: int,
) -> tuple[SectorMesh, Capture, dict[str, float], str]:
    cap = Capture([], {}, [], Counter())
    times: dict[str, float] = {}
    notes: list[str] = []
    mesh = SectorMesh()
    _install(cap)
    try:
        base = _blob(archives, map_name, ix, iz, "base")
        if base:
            try:
                mesh, times["base"] = _parse_file(base, cap, order=True)
                notes.append(_fmt_counts("base", cap.counts))
            except SectorError as exc:
                notes.append(f"base failed: {exc}")
        else:
            notes.append("base missing")
        data = _blob(archives, map_name, ix, iz, "data")
        if data and cap.order:
            started = time.perf_counter()
            try:
                notes.append(_read_data(data, cap))
            except SectorError as exc:
                notes.append(f"data failed: {exc}")
            times["data"] = (time.perf_counter() - started) * 1000.0
        elif data:
            notes.append("data present but no road or prefab to attach it to")
        else:
            notes.append("data missing")
        aux = _blob(archives, map_name, ix, iz, "aux")
        if aux:
            try:
                aux_mesh, times["aux"] = _parse_file(aux, cap, order=False)
                notes.append(_fmt_counts("aux", cap.counts))
                # Nodes only. Aux roads must not become extra lane lines.
                mesh.nodes.update(aux_mesh.nodes)
            except SectorError as exc:
                notes.append(f"aux failed: {exc}")
        else:
            notes.append("aux missing")
    finally:
        _restore()
    return mesh, cap, times, "; ".join(notes)


def _fmt_counts(label: str, counts: Counter) -> str:
    parts = []
    for typ, count in sorted(counts.items(), key=lambda item: -item[1]):
        if count <= 0:
            continue
        parts.append(f"{_TYPE_NAMES.get(typ, str(typ))} {count}")
    return f"{label}: " + ", ".join(parts[:12])


def _sii_text(blob: bytes) -> str | None:
    if blob.startswith(b"ScsC") or b"\x00" in blob[:120]:
        return None
    return blob.decode("utf-8", "replace")


def _path_field(body: str) -> str:
    for raw in body.splitlines():
        line = raw.strip()
        if "/" not in line or ":" not in line:
            continue
        value = line.split(":", 1)[1].strip().strip('"')
        if "/" in value:
            return value
    return ""


def _collision_field(body: str) -> bool | None:
    for raw in body.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        if line.lower().startswith("use_collision"):
            return "true" in line.lower()
    return None


def load_catalog(archives: list[HashFs]) -> tuple[dict[str, dict[str, DefInfo]], str]:
    books: dict[str, dict[str, DefInfo]] = defaultdict(dict)
    files = 0
    encrypted = 0
    started = time.perf_counter()
    for archive in archives:
        listing = archive.listdir("def/world")
        if listing is None:
            continue
        _subs, names = listing
        for name in names:
            if not name.endswith(".sii"):
                continue
            lower = name.lower()
            if not any(stem in lower for stem in _DEF_STEMS):
                continue
            blob = archive.get(f"def/world/{name}")
            if not blob:
                continue
            files += 1
            text = _sii_text(blob)
            if text is None:
                encrypted += 1
                continue
            for kind, unit, body in _units(text):
                info = DefInfo(_path_field(body), _collision_field(body))
                books[kind][unit] = info
                alias = unit.split(".", 1)[1] if "." in unit else ""
                if alias:
                    books[kind][alias] = info
    elapsed = (time.perf_counter() - started) * 1000.0
    tokens = sum(len(book) for book in books.values())
    note = f"defs {tokens} tokens from {files} files, {encrypted} encrypted, {elapsed:.0f} ms"
    return books, note


def _lookup(books: dict[str, dict[str, DefInfo]], token: str, kinds: tuple[str, ...]) -> DefInfo | None:
    for kind in kinds:
        info = books.get(kind, {}).get(token)
        if info is not None:
            return info
    return None


def _category(path: str, token: str) -> str:
    text = path.lower()
    if "decal" in text:
        return "other"
    if "/railing/" in text or "/barrier/" in text or "/fences/" in text or "fence_" in text:
        return "rail"
    if "/pillar/" in text or "pillar" in text or "bollard" in text or "trolley_column" in text:
        return "pillar"
    if "hedge" in text or "shrub" in text:
        return "hedge"
    if any(bit in text for bit in ("/tree", "tree_", "treewall")):
        return "tree"
    if "/sign/" in text:
        return "sign"
    if "/lamp/" in text or "streetlight" in text or "street_light" in text:
        return "lamp"
    if "/building/" in text or "building_" in text or token.lower().startswith("scheme"):
        return "building"
    if "bridge" in text:
        return "bridge"
    if not text and token.lower().startswith("scheme"):
        return "building"
    return "other" if text else "unknown"


def classify(token: str, books: dict[str, dict[str, DefInfo]], item_kind: str, *, forced: str = "") -> str:
    if forced:
        return forced
    info = _lookup(books, token, _BOOKS.get(item_kind, ()))
    if info is None:
        return "unknown"
    return _category(info.path, token)


def _is_solid(item_flag: bool | None, info: DefInfo | None) -> bool:
    if info is not None and info.collision is False:
        return False
    if item_flag is False:
        return False
    return item_flag is True or (info is not None and info.collision is True)


def _resolve(points: list[PointItem], nodes: dict[int, MapNode]) -> tuple[int, int]:
    placed = 0
    missing = 0
    for item in points:
        start = nodes.get(item.node_uid)
        if start is None:
            missing += 1
            continue
        item.x = start.x
        item.z = start.z
        item.placed = True
        placed += 1
        if item.segment:
            end = nodes.get(item.node_uid2)
            if end is None:
                item.segment = False
            else:
                item.x2 = end.x
                item.z2 = end.z
    return placed, missing


def _span(start: MapNode, end: MapNode, length: float):
    f0x, f0z = node_forward_xz(start)
    f1x, f1z = node_forward_xz(end)
    steps = max(1, min(80, math.ceil(max(length, 1.0) / 12.0)))
    return sample_span(
        start.x, start.y, start.z, f0x, f0z,
        end.x, end.y, end.z, f1x, f1z,
        steps,
    )


def _shift(x: float, z: float, fx: float, fz: float, lateral: float) -> tuple[float, float]:
    return (x + lateral * (-fz), z + lateral * fx)


def bend_points(nodes: Sequence[MapNode]) -> list[tuple[float, float, float, float, float]]:
    """Hermite samples. Each node's ahead is the tangent, so a corner stays bent."""
    if len(nodes) < 2:
        return []
    span: list[tuple[float, float, float, float, float]] = []
    for start, end in zip(nodes, nodes[1:]):
        chord = math.hypot(end.x - start.x, end.z - start.z)
        steps = max(8, min(80, math.ceil(max(chord, 1.0) / 8.0)))
        f0x, f0z = node_forward_xz(start)
        f1x, f1z = node_forward_xz(end)
        piece = sample_span(
            start.x, start.y, start.z, f0x, f0z,
            end.x, end.y, end.z, f1x, f1z,
            steps,
        )
        if span:
            piece = piece[1:]
        span.extend(piece)
    return span


def _bent_item(item: PointItem, nodes: dict[int, MapNode]):
    chain = [nodes.get(item.node_uid)]
    chain.extend(nodes.get(uid) for uid in item.locators)
    chain.append(nodes.get(item.node_uid2))
    if any(node is None for node in chain):
        return None
    return bend_points(chain)


def _offset_line(span, off0: float, off1: float) -> list[tuple[float, float]]:
    last = max(len(span) - 1, 1)
    points: list[tuple[float, float]] = []
    for index, (x, _y, z, fx, fz) in enumerate(span):
        lateral = off0 + (off1 - off0) * (index / last)
        points.append(_shift(x, z, fx, fz, lateral))
    return points


def _convention(roads: list[RoadExtra]) -> str:
    right = [item.offset_m for road in roads for item in road.railings if item.side == "right"]
    left = [item.offset_m for road in roads for item in road.railings if item.side == "left"]
    right = [value for value in right if abs(value) > 0.2]
    left = [value for value in left if abs(value) > 0.2]
    if not right or not left:
        return "signed"
    if statistics.median(right) > 0.3 and statistics.median(left) > 0.3:
        return "outward"
    return "signed"


def _lateral(side: str, offset: float, convention: str) -> float:
    if convention == "outward":
        magnitude = abs(offset)
        return magnitude if side == "right" else -magnitude
    return offset


def _road_index(roads: list[RoadExtra], nodes: dict[int, MapNode]):
    grid: dict[tuple[int, int], list[tuple[float, float, float, float]]] = defaultdict(list)
    resolved = 0
    for road in roads:
        start = nodes.get(road.start_uid)
        end = nodes.get(road.end_uid)
        if start is None or end is None:
            continue
        resolved += 1
        span = _span(start, end, road.length)
        for i in range(len(span) - 1):
            ax, _ay, az = span[i][0], span[i][1], span[i][2]
            bx, bz = span[i + 1][0], span[i + 1][2]
            min_x, max_x = (ax, bx) if ax <= bx else (bx, ax)
            min_z, max_z = (az, bz) if az <= bz else (bz, az)
            ix0 = math.floor(min_x / _CELL)
            ix1 = math.floor(max_x / _CELL)
            iz0 = math.floor(min_z / _CELL)
            iz1 = math.floor(max_z / _CELL)
            for ix in range(ix0, ix1 + 1):
                for iz in range(iz0, iz1 + 1):
                    grid[(ix, iz)].append((ax, az, bx, bz))
    return grid, resolved


def _near(x: float, z: float, grid) -> float:
    ix = math.floor(x / _CELL)
    iz = math.floor(z / _CELL)
    best = 1e9
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            for ax, az, bx, bz in grid.get((ix + dx, iz + dz), ()):
                dist = _segment(x, z, ax, az, bx, bz)
                if dist < best:
                    best = dist
    return best


def _segment(px: float, pz: float, ax: float, az: float, bx: float, bz: float) -> float:
    dx = bx - ax
    dz = bz - az
    length2 = dx * dx + dz * dz
    if length2 < 1e-8:
        return math.hypot(px - ax, pz - az)
    t = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / length2))
    return math.hypot(px - (ax + t * dx), pz - (az + t * dz))


def _top(counter: Counter, limit: int = 8) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{name} {count}" for name, count in counter.most_common(limit))


def report_sector(
    label: str,
    mesh: SectorMesh,
    cap: Capture,
    times: dict[str, float],
    note: str,
    books: dict[str, dict[str, DefInfo]],
    focus: tuple[float, float] | None,
) -> dict:
    placed, missing = _resolve(cap.points, mesh.nodes)
    roads = list(cap.roads.values())
    grid, road_n = _road_index(roads, mesh.nodes)
    convention = _convention(roads)
    cats: Counter = Counter()
    near_cats: Counter = Counter()
    solid_near: Counter = Counter()
    near_names: dict[str, Counter] = defaultdict(Counter)
    near_paths: dict[str, Counter] = defaultdict(Counter)
    collision = Counter()
    near_n = 0
    for item in cap.points:
        if not item.placed:
            continue
        kind = classify(item.name, books, item.kind)
        cats[kind] += 1
        info = _lookup(books, item.name, _BOOKS.get(item.kind, ()))
        solid = _is_solid(item.collision, info)
        if item.collision is not None:
            collision[f"{item.kind} {'on' if item.collision else 'off'}"] += 1
        dist = _near(item.x, item.z, grid)
        if item.segment:
            dist = min(dist, _near(item.x2, item.z2, grid), _near(
                (item.x + item.x2) * 0.5, (item.z + item.z2) * 0.5, grid,
            ))
        if dist <= 15.0:
            near_n += 1
            near_cats[kind] += 1
            if solid:
                solid_near[kind] += 1
            near_names[kind][item.name] += 1
            if info is not None and info.path:
                near_paths[kind][info.path] += 1
    rail_n = sum(len(road.railings) for road in roads)
    model_defs = sum(len(road.models) for road in roads)
    instances = 0
    roadside_near = Counter()
    for road in roads:
        start = mesh.nodes.get(road.start_uid)
        end = mesh.nodes.get(road.end_uid)
        if start is None or end is None or road.length <= 0:
            continue
        for model in road.models:
            if model.distance_m < 1.0:
                continue
            count = int(road.length / model.distance_m)
            instances += count
            roadside_near[classify(model.name, books, "roadside")] += count
    veg_n = sum(len(road.veg) for road in roads)
    veg_hit = sum(1 for road in roads for band in road.veg if band.collision)
    veg_names = Counter(band.name for road in roads for band in road.veg)
    focus_n = ""
    if focus is not None:
        fx, fz = focus
        window = 0
        for item in cap.points:
            if item.placed and math.hypot(item.x - fx, item.z - fz) <= 200.0:
                window += 1
        focus_n = f"placed items inside 200 m of the pose: {window}"
    print(f"\n{label}")
    print(f"  {note}")
    print(
        "  parse ms: "
        + ", ".join(f"{key} {value:.1f}" for key, value in times.items())
        + f"  roads {len(roads)} ({road_n} with both nodes)  points {len(cap.points)}"
        f" placed {placed} missing-node {missing}"
    )
    print(f"  railing convention {convention}, railing slots {rail_n}")
    print(f"  roadside model slots {model_defs}, repeated instances {instances}")
    print(f"  vegetation bands {veg_n}, bands with collision flag {veg_hit}")
    if veg_names:
        print(f"  vegetation names: {_top(veg_names)}")
    print(f"  placed categories: {_top(cats, 12)}")
    print(f"  within 15 m of a road centre: {near_n}  {_top(near_cats, 12)}")
    print(f"  of those, tagged solid: {_top(solid_near, 12)}")
    print(f"  collision flags: {_top(collision, 8)}")
    if focus_n:
        print(f"  {focus_n}")
    for kind in ("tree", "sign", "pillar", "rail", "lamp", "building", "other"):
        if near_names[kind]:
            print(f"  near {kind}: {_top(near_names[kind])}")
    for label in ("tree", "hedge", "pillar", "rail", "sign", "lamp", "bridge", "other"):
        if near_paths[label]:
            print(f"    paths {label}: {_top(near_paths[label], 5)}")
    rail_paths = Counter()
    for road in roads:
        for rail in road.railings:
            info = _lookup(books, rail.name, _BOOKS["railing"])
            rail_paths[info.path if info and info.path else rail.name] += 1
    if rail_paths:
        print(f"  road-railing paths: {_top(rail_paths, 5)}")
    if roadside_near:
        print(f"  roadside repeats by category: {_top(roadside_near)}")
        shown = 0
        for road in roads:
            for model in road.models:
                if shown >= 6:
                    break
                kind = classify(model.name, books, "roadside")
                info = _lookup(books, model.name, _BOOKS["roadside"])
                path = info.path if info and info.path else model.name
                print(
                    f"    {model.side} {path} offset {model.offset_m:.2f} m"
                    f" every {model.distance_m:.1f} m ({kind})"
                )
                shown += 1
    offsets = [item.offset_m for road in roads for item in road.railings]
    if offsets:
        print(
            "  railing offset m: "
            f"median {statistics.median(offsets):.2f} "
            f"min {min(offsets):.2f} max {max(offsets):.2f}"
        )
        rail_names = Counter(item.name for road in roads for item in road.railings)
        print(f"  railing names: {_top(rail_names)}")
    return {
        "roads": roads,
        "points": cap.points,
        "nodes": mesh.nodes,
        "convention": convention,
        "near": near_n,
        "times": times,
    }


def _polylines(summary: dict, books: dict[str, dict[str, DefInfo]]):
    lines: list[tuple[str, list[tuple[float, float]]]] = []
    dots: list[tuple[str, float, float, bool | None]] = []
    nodes: dict[int, MapNode] = summary["nodes"]
    convention = summary["convention"]
    for road in summary["roads"]:
        start = nodes.get(road.start_uid)
        end = nodes.get(road.end_uid)
        if start is None or end is None:
            continue
        span = _span(start, end, road.length)
        centre = [(x, z) for x, _y, z, _fx, _fz in span]
        if len(centre) >= 2:
            lines.append(("road", centre, ""))
        for rail in road.railings:
            lateral = _lateral(rail.side, rail.offset_m, convention)
            shifted = [
                _shift(x, z, fx, fz, lateral) for x, _y, z, fx, fz in span
            ]
            if len(shifted) >= 2:
                lines.append(("rail", shifted, _model_label(rail.name, books, "railing")))
        if road.length <= 0:
            continue
        for model in road.models:
            if model.distance_m < 1.0:
                continue
            lateral = _lateral(model.side, model.offset_m, convention)
            kind = classify(model.name, books, "roadside")
            label = _model_label(model.name, books, "roadside")
            pos = model.distance_m * 0.5
            made = 0
            while pos < road.length and made < 80:
                t = min(0.999, pos / road.length)
                index = min(len(span) - 1, int(t * (len(span) - 1)))
                x, _y, z, fx, fz = span[index]
                dots.append((kind, *_shift(x, z, fx, fz, lateral), True, label))
                pos += model.distance_m
                made += 1
    for item in summary["points"]:
        if not item.placed:
            continue
        if item.segment and item.kind in {"curve", "building"}:
            span = _bent_item(item, nodes)
            if not span:
                continue
            if item.kind == "curve" and item.parts:
                for part in item.parts:
                    kind = classify(part.name, books, "curve")
                    label = _model_label(part.name, books, "curve")
                    lines.append((kind, _offset_line(span, part.off0, part.off1), label))
                continue
            kind = classify(item.name, books, item.kind)
            label = _model_label(item.name, books, item.kind)
            lines.append((kind, [(x, z) for x, _y, z, _fx, _fz in span], label))
            continue
        kind = classify(item.name, books, item.kind)
        dots.append((kind, item.x, item.z, item.collision, _model_label(item.name, books, item.kind)))
    return lines, dots


def _model_label(token: str, books: dict[str, dict[str, DefInfo]], item_kind: str) -> str:
    info = _lookup(books, token, _BOOKS.get(item_kind, ()))
    if info is not None and info.path:
        return info.path
    return f"{item_kind}:{token}"


def is_clutter(label: str) -> bool:
    """Road paint, power lines, low curbs, street signs, decals, and the ground."""
    text = label.lower().replace("\\", "/")
    if text.startswith("sign:") or "/props/street/" in text or "/sign/" in text:
        return True
    if any(bit in text for bit in ("power_line", "power_wire", "powerline")):
        return True
    if any(bit in text for bit in (
        "curb", "stripe", "at_lines", "overlay", "decal",
        "/terrain/", "terrain_", "road_mark", "lane_mark", "roadmark",
    )):
        if any(keep in text for keep in ("barrier", "railing", "fence", "guardrail", "wall")):
            return False
        return True
    return False


def geometry_for(nodes: dict[int, MapNode], caps: Sequence[Capture], books: dict[str, dict[str, DefInfo]]):
    """Polylines and dots for the lane window. Roads stay on their own layer."""
    roads: dict[int, RoadExtra] = {}
    points: list[PointItem] = []
    for cap in caps:
        roads.update(cap.roads)
        points.extend(cap.points)
    _resolve(points, nodes)
    road_list = list(roads.values())
    summary = {
        "roads": road_list,
        "points": points,
        "nodes": nodes,
        "convention": _convention(road_list),
    }
    return _polylines(summary, books)


def draw_window(
    path: Path,
    summary: dict,
    books: dict[str, dict[str, DefInfo]],
    origin: tuple[float, float],
    radius_m: float,
) -> None:
    from PySide6.QtGui import QColor, QImage, QPainter, QPen
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    lines, dots = _polylines(summary, books)
    ox, oz = origin
    size = 1200
    scale = size / (radius_m * 2.0)
    image = QImage(size, size, QImage.Format.Format_RGB32)
    image.fill(QColor(16, 16, 22))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    def to_px(x: float, z: float) -> QPointF | None:
        px = (x - ox + radius_m) * scale
        py = (z - oz + radius_m) * scale
        if px < -20 or py < -20 or px > size + 20 or py > size + 20:
            return None
        return QPointF(px, py)

    for row in lines:
        kind, pts = row[0], row[1]
        if kind in {"other", "unknown", "road"}:
            continue
        screen = [p for p in (to_px(x, z) for x, z in pts) if p is not None]
        if len(screen) < 2:
            continue
        color = _DRAW.get(kind, _DRAW["other"])
        painter.setPen(QPen(QColor(*color), 2.4 if kind == "rail" else 1.2))
        for a, b in zip(screen, screen[1:]):
            painter.drawLine(a, b)
    for row in dots:
        kind, x, z, collision = row[0], row[1], row[2], row[3]
        if kind in {"other", "unknown"}:
            continue
        point = to_px(x, z)
        if point is None:
            continue
        color = _DRAW.get(kind, _DRAW["other"])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(*color))
        radius = 3.5 if kind in {"pillar", "sign", "tree"} else 2.2
        painter.drawEllipse(point, radius, radius)
    painter.setPen(QPen(QColor(242, 242, 242), 2))
    center = to_px(ox, oz)
    if center is not None:
        painter.drawEllipse(center, 5, 5)
    painter.end()
    image.save(str(path))


def _sector_names(archives: list[HashFs], map_name: str) -> list[tuple[int, int]]:
    names: set[str] = set()
    for archive in archives:
        listing = archive.listdir(f"map/{map_name}")
        if listing is None:
            continue
        _subs, files = listing
        for name in files:
            if name.endswith(".base"):
                names.add(name)
    out: list[tuple[int, int]] = []
    for name in names:
        stem = name[:-5]
        if len(stem) < 13:
            continue
        try:
            out.append((int(stem[3:8]), int(stem[8:13])))
        except ValueError:
            continue
    out.sort()
    return out


def _quantile_sectors(sectors: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    if not sectors or count <= 0:
        return []
    picked: list[tuple[int, int]] = []
    for i in range(count):
        index = int((i + 0.5) * len(sectors) / count)
        index = min(index, len(sectors) - 1)
        item = sectors[index]
        if item not in picked:
            picked.append(item)
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract roadside map objects from one ETS2 sector.")
    parser.add_argument("--x", type=float, default=3296.0)
    parser.add_argument("--z", type=float, default=19626.0)
    parser.add_argument("--scan", type=int, default=0, help="Also parse this many sectors spread over the map")
    parser.add_argument("--no-draw", action="store_true")
    parser.add_argument("--radius", type=float, default=350.0)
    args = parser.parse_args()

    installs = find_game_installations("ets2")
    if not installs:
        raise SystemExit("no ETS2 install found")
    print(f"game {installs[0].name}")
    started = time.perf_counter()
    map_name, maps, everything = _open_map(installs[0])
    print(f"archives {len(everything)} map {map_name} open {(time.perf_counter() - started) * 1000:.0f} ms")
    catalog, catalog_note = load_catalog(everything)
    print(catalog_note)

    ix, iz = sector_index(args.x, args.z)
    targets = [(ix, iz, f"pose sector {ix:+d},{iz:+d} at ({args.x:.0f}, {args.z:.0f})")]
    if args.scan:
        sectors = _sector_names(maps, map_name)
        print(f"sectors on the map: {len(sectors)}")
        for sx, sz in _quantile_sectors(sectors, args.scan):
            if (sx, sz) != (ix, iz):
                targets.append((sx, sz, f"sample sector {sx:+d},{sz:+d}"))

    drawn = False
    draw_path = Path(tempfile.gettempdir()) / "monocruise-map-obstacles.png"
    for sx, sz, label in targets:
        mesh, cap, times, note = _load_sector(maps, map_name, sx, sz)
        summary = report_sector(
            label, mesh, cap, times, note, catalog,
            (args.x, args.z) if (sx, sz) == (ix, iz) else None,
        )
        if not args.no_draw and not drawn and (sx, sz) == (ix, iz):
            draw_window(draw_path, summary, catalog, (args.x, args.z), args.radius)
            print(f"  drew {args.radius:.0f} m window to {draw_path.name}")
            drawn = True
    if drawn:
        print(f"png {draw_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
