"""Binary `.base` sector: roads and the nodes they hang on.

Layout follows the version-907 sector template (item type, then a fixed
header, then a per-type payload). Anything that is not a road is skipped so
the node table at the end stays aligned. A parse that does not consume the
file is a layout miss, not a short read to ignore.

TruckersMP mod sectors come in older versions. Three item layouts differ and are
branched on ``_Cur.version``: terrain (one field fewer before 902), sign (no
board list before 906) and curve (one entry, no mask, before 907). Only the
versions in ``_OLD_VERSIONS`` are read, because each was checked to consume its
files exactly and to agree with the base game on shared roads; any other older
version is refused. See core/road_layout/README.md.

Node positions are stored in 1/256 m. The ground plane is X and Z, Y up,
matching telemetry `coordinateX/Y/Z`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

_TOKEN_LETTERS = "\x000123456789abcdefghijklmnopqrstuvwxyz_"

# Item type ids in the sector file. Gaps are types the template does not carry.
ROAD = 3
_TERRAIN = 1
_BUILDING = 2
_PREFAB = 4
_MODEL = 5
_COMPANY = 6
_SERVICE = 7
_CUT_PLANE = 8
_MOVER = 9
_NO_WEATHER = 11
_CITY = 12
_HINGE = 13
_MAP_OVERLAY = 18
_FERRY = 19
_GARAGE = 22
_CAMERA_POINT = 23
_TRIGGER = 34
_FUEL_PUMP = 35
_SIGN = 36
_BUS_STOP = 37
_TRAFFIC_RULE = 38
_BEZIER = 39
_COMPOUND = 40
_TRAJECTORY = 41
_MAP_AREA = 42
_FAR_MODEL = 43
_CURVE = 44
_CAMERA_PATH = 45
_CUTSCENE = 46
_HOOKUP = 47
_VISIBILITY = 48
_GATE = 49


@dataclass(frozen=True)
class MapNode:
    """One map node in the telemetry world frame."""

    uid: int
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float


@dataclass(frozen=True)
class MapRoad:
    """One road item: two node ids and the road-look token."""

    uid: int
    look: str
    start_uid: int
    end_uid: int
    length: float


@dataclass(frozen=True)
class MapPrefab:
    """One prefab item. ``node_uids[0]`` is the world node the local origin sits on."""

    uid: int
    model: str
    node_uids: tuple[int, ...]
    origin_index: int


@dataclass
class SectorMesh:
    """Roads, prefabs and nodes from one or more `.base` sectors."""

    version: int = 0
    nodes: dict[int, MapNode] = field(default_factory=dict)
    roads: list[MapRoad] = field(default_factory=list)
    prefabs: list[MapPrefab] = field(default_factory=list)
    sectors: int = 0


class SectorError(ValueError):
    """The sector bytes do not match the template this reader knows."""


# Older sector versions seen in TruckersMP archives and verified against the base game.
_OLD_VERSIONS = frozenset({896, 898, 900, 901, 903, 905, 906})


class _Cur:
    def __init__(self, data: bytes, version: int = 907) -> None:
        self.b = data
        self.i = 0
        self.n = len(data)
        self.version = version

    def _take(self, k: int) -> memoryview:
        if self.i + k > self.n:
            raise SectorError(f"truncated at {self.i}, need {k} more bytes")
        sl = memoryview(self.b)[self.i:self.i + k]
        self.i += k
        return sl

    def u8(self) -> int:
        return self._take(1)[0]

    def u16(self) -> int:
        return int.from_bytes(self._take(2), "little")

    def i16(self) -> int:
        return int.from_bytes(self._take(2), "little", signed=True)

    def u32(self) -> int:
        return int.from_bytes(self._take(4), "little")

    def i32(self) -> int:
        return int.from_bytes(self._take(4), "little", signed=True)

    def i8(self) -> int:
        return int.from_bytes(self._take(1), "little", signed=True)

    def u64(self) -> int:
        return int.from_bytes(self._take(8), "little")

    def f32(self) -> float:
        return struct.unpack("<f", self._take(4))[0]

    def skip(self, k: int) -> None:
        self._take(k)

    def token(self) -> str:
        return _token_str(self.u64())

    def u64_str(self) -> str:
        length = self.u64()
        if length == 0:
            return ""
        if length > 1_000_000:
            raise SectorError(f"string length {length} at {self.i}")
        return bytes(self._take(int(length))).decode("utf-8", "replace")


def _token_str(n: int) -> str:
    chars: list[str] = []
    while True:
        chars.append(_TOKEN_LETTERS[n % 38])
        n //= 38
        if n == 0:
            break
    return "".join(chars).replace("\x00", "")


def _tokens(c: _Cur) -> None:
    for _ in range(c.u32()):
        c.token()


def _u64s(c: _Cur) -> int:
    count = c.u32()
    c.skip(8 * count)
    return count


def _veg_spheres(c: _Cur) -> None:
    # center float3, radius, type: 20 bytes.
    c.skip(20 * c.u32())


def _quad(c: _Cur) -> None:
    mats = c.u16()
    c.skip(10 * mats)  # token + u16
    colors = c.u16()
    c.skip(4 * colors)
    c.u16()
    c.u16()
    c.skip(4 * c.u32())
    # x u16, y u16, normal float3: 16 bytes.
    c.skip(16 * c.u32())
    c.skip(16 * c.u32())


def _header(c: _Cur) -> int:
    uid = c.u64()
    c.skip(12 + 12 + 12 + 4)
    c.u32()
    c.u8()
    return uid


def _skip_terrain(c: _Cur) -> None:
    c.u64()
    c.u64()
    c.skip(12)
    c.skip(12)
    c.f32()
    c.f32()
    if c.version >= 902:
        c.u32()
    c.skip((8 + 2) * 3)
    for _ in range(2):
        c.u16()
        c.token()
        c.f32()
        c.token()
        c.f32()
        for _v in range(3):
            c.token()
            c.u16()
            c.u8()
            c.u8()
            c.u16()
            c.u16()
        c.u16()
        c.u16()
    _veg_spheres(c)
    _quad(c)
    _quad(c)
    c.token()
    c.token()
    c.token()
    c.token()


def _skip_building(c: _Cur) -> None:
    c.token()
    c.token()
    c.u64()
    c.u64()
    c.f32()
    c.u32()
    c.f32()
    c.skip(4 * c.u32())


def _read_road(c: _Cur, uid: int) -> MapRoad:
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
    for _ in range(3):
        c.token()
        c.i16()
        c.token()
        c.i16()
    c.i32()
    c.i32()
    start = c.u64()
    end = c.u64()
    length = c.f32()
    return MapRoad(uid, look, start, end, length)


def _read_prefab(c: _Cur, uid: int) -> MapPrefab:
    model = c.token()
    c.token()
    for _ in range(c.u32()):
        c.token()
    count = c.u32()
    node_uids = tuple(c.u64() for _ in range(count))
    _u64s(c)
    c.u64()
    origin = c.u16()
    for _ in range(count):
        c.token()
        c.f32()
    c.token()
    return MapPrefab(uid, model, node_uids, origin)


def _skip_model(c: _Cur) -> None:
    c.token()
    c.token()
    c.token()
    for _ in range(c.u32()):
        c.token()
    c.u64()
    c.skip(12)
    c.token()
    c.u32()
    c.f32()


def _skip_company(c: _Cur) -> None:
    c.token()
    c.u64()
    c.token()
    c.u64()
    c.skip(12 * c.u32())


def _skip_service(c: _Cur) -> None:
    c.u64()
    c.u64()
    _u64s(c)


def _skip_mover(c: _Cur) -> None:
    _tokens(c)
    c.token()
    c.token()
    c.token()
    c.f32()
    c.f32()
    c.f32()
    c.u32()
    c.skip(4 * c.u32())
    _u64s(c)


def _skip_no_weather(c: _Cur) -> None:
    c.f32()
    c.f32()
    c.i32()
    c.skip(16)
    c.u64()


def _skip_city(c: _Cur) -> None:
    c.token()
    c.f32()
    c.f32()
    c.u64()


def _skip_hinge(c: _Cur) -> None:
    c.token()
    c.token()
    c.u64()
    c.f32()
    c.f32()


def _skip_ferry(c: _Cur) -> None:
    c.token()
    c.u64()
    c.u64()
    c.skip(12)


def _skip_garage(c: _Cur) -> None:
    c.token()
    c.u32()
    c.u64()
    c.u64()
    _u64s(c)


def _skip_trigger(c: _Cur) -> None:
    _tokens(c)
    node_count = _u64s(c)
    for _ in range(c.u32()):
        c.token()
        has = c.i32()
        if has < 0:
            continue
        c.skip(4 * has)
        for _p in range(c.u32()):
            c.u64_str()
        _u64s(c)
        c.f32()
        c.u32()
    if node_count == 1:
        c.f32()


def _sign_boards(c: _Cur) -> None:
    for _ in range(c.u32()):
        c.token()
        flags = c.u8()
        if flags & 0x01:
            c.i8()
            c.i8()
        if flags & 0x02:
            c.token()


def _sign_attributes(c: _Cur) -> None:
    for _ in range(c.u32()):
        c.u32()
        c.token()
        for _a in range(c.u32()):
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


def _skip_sign(c: _Cur) -> None:
    c.token()
    c.u64()
    c.token()
    c.token()
    for _ in range(c.u8()):
        c.token()
        c.token()
        c.token()
    template = c.u64_str()
    if template == "":
        return
    if c.version >= 906:
        _sign_boards(c)
    _sign_attributes(c)


def _skip_traffic_rule(c: _Cur) -> None:
    _tokens(c)
    _u64s(c)
    c.token()
    c.f32()


def _skip_bezier(c: _Cur) -> None:
    c.skip(12 * 16)
    c.u16()
    c.u16()
    c.u64()
    c.u32()
    for _ in range(3):
        c.token()
        c.u16()
        c.u8()
    _veg_spheres(c)
    _quad(c)


def _skip_trajectory(c: _Cur) -> None:
    _u64s(c)
    c.u64()
    for _ in range(c.u32()):
        c.u32()
        c.token()
        c.u32()
        c.skip(12)
    for _ in range(c.u32()):
        c.token()
        c.token()
    _tokens(c)


def _skip_far_model(c: _Cur) -> None:
    c.f32()
    c.f32()
    for _ in range(c.u32()):
        c.token()
        c.skip(12)
    _u64s(c)
    _u64s(c)


def _curve_entry(c: _Cur, current: bool) -> None:
    if current:
        c.token()
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
    if current:
        for _ in range(5):
            c.f32()


def _skip_curve(c: _Cur) -> None:
    if c.version >= 907:
        c.u64()
        c.u64()
        c.skip(16)
        c.f32()
        for _ in range(c.u32().bit_count()):
            _curve_entry(c, True)
        return
    c.token()
    c.u64()
    c.u64()
    c.skip(16)
    c.f32()
    _curve_entry(c, False)


def _skip_camera_path(c: _Cur) -> None:
    _tokens(c)
    _u64s(c)
    _u64s(c)
    _u64s(c)
    for _ in range(c.u32()):
        c.i32()
        c.i32()
        c.f32()
        c.f32()
        c.skip(12)
        c.skip(12)
    c.f32()


def _skip_cutscene(c: _Cur) -> None:
    _tokens(c)
    c.u64()
    for _ in range(c.u32()):
        c.skip(4 * c.u32())
        for _s in range(c.u32()):
            c.u64_str()
        _u64s(c)
        c.f32()
        c.u32()


def _skip_gate(c: _Cur) -> None:
    c.token()
    _u64s(c)
    for _ in range(2):
        c.u64_str()
        c.i32()


def _read_node(c: _Cur) -> MapNode:
    uid = c.u64()
    px = c.i32() / 256.0
    py = c.i32() / 256.0
    pz = c.i32() / 256.0
    qw, qx, qy, qz = c.f32(), c.f32(), c.f32(), c.f32()
    c.u64()
    c.u64()
    c.u32()
    return MapNode(uid, px, py, pz, qw, qx, qy, qz)


_SKIP = {
    _TERRAIN: _skip_terrain,
    _BUILDING: _skip_building,
    _MODEL: _skip_model,
    _COMPANY: _skip_company,
    _SERVICE: _skip_service,
    _CUT_PLANE: _u64s,
    _MOVER: _skip_mover,
    _NO_WEATHER: _skip_no_weather,
    _CITY: _skip_city,
    _HINGE: _skip_hinge,
    _MAP_OVERLAY: lambda c: (c.token(), c.u64()),
    _FERRY: _skip_ferry,
    _GARAGE: _skip_garage,
    _CAMERA_POINT: lambda c: (_tokens(c), c.u64()),
    _TRIGGER: _skip_trigger,
    _FUEL_PUMP: lambda c: (c.u64(), c.u64(), _u64s(c)),
    _SIGN: _skip_sign,
    _BUS_STOP: lambda c: (c.token(), c.u64(), c.u64()),
    _TRAFFIC_RULE: _skip_traffic_rule,
    _BEZIER: _skip_bezier,
    _TRAJECTORY: _skip_trajectory,
    _MAP_AREA: lambda c: (_u64s(c), c.u32()),
    _FAR_MODEL: _skip_far_model,
    _CURVE: _skip_curve,
    _CAMERA_PATH: _skip_camera_path,
    _CUTSCENE: _skip_cutscene,
    _HOOKUP: lambda c: (c.u64_str(), c.u64()),
    _VISIBILITY: lambda c: (c.u64(), c.f32(), c.f32(), _u64s(c)),
    _GATE: _skip_gate,
}


def _read_item(c: _Cur, mesh: SectorMesh, nested: bool) -> None:
    typ = c.u32()
    uid = _header(c)
    if typ == ROAD:
        mesh.roads.append(_read_road(c, uid))
        return
    if typ == _PREFAB:
        mesh.prefabs.append(_read_prefab(c, uid))
        return
    if typ == _COMPOUND:
        if nested:
            raise SectorError(f"nested compound at {c.i}")
        c.u64()
        for _ in range(c.u32()):
            _read_item(c, mesh, nested=True)
        for _ in range(c.u32()):
            node = _read_node(c)
            mesh.nodes.setdefault(node.uid, node)
        return
    skip = _SKIP.get(typ)
    if skip is None:
        raise SectorError(f"unknown item type {typ} at {c.i}")
    skip(c)


def parse_sector(data: bytes) -> SectorMesh:
    """Roads and nodes from one sector `.base` blob."""
    c = _Cur(data)
    version = c.u32()
    if version < 907 and version not in _OLD_VERSIONS:
        raise SectorError(f"version {version}: layout not verified")
    c.version = version
    c.token()
    c.u32()
    mesh = SectorMesh(version=version)
    try:
        for _ in range(c.u32()):
            _read_item(c, mesh, nested=False)
        for _ in range(c.u32()):
            node = _read_node(c)
            mesh.nodes[node.uid] = node
        c.skip(8 * c.u32())
    except SectorError as exc:
        raise SectorError(f"version {version}: {exc}") from exc
    if c.i != c.n:
        raise SectorError(
            f"version {version}: {c.n - c.i} trailing bytes at {c.i} "
            f"({len(mesh.roads)} roads, {len(mesh.nodes)} nodes)"
        )
    mesh.sectors = 1
    return mesh
