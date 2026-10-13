"""Sector capture for the obstacle probe: models, signs, curves, railings, vegetation.

Patches the lane reader's item hooks in ``core.road_layout.sector`` while one
sector loads, then restores them. Tools only: the runtime map reads lanes and
never installs these hooks.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import core.road_layout.sector as sector
from core.road_layout.archive import HashFs
from core.road_layout.grid import map_folder
from core.road_layout.sector import (
    MapPrefab,
    MapRoad,
    SectorError,
    SectorMesh,
    _Cur,
    _quad,
    parse_sector,
)

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
