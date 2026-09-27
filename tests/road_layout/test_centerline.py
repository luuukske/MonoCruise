"""Centerline geometry and a synthetic sector, no game files."""

from __future__ import annotations

import struct

from core.road_layout.centerline import nearest_centerline, sample_centerline
from core.road_layout.cityhash import city_hash64_path
from core.road_layout.grid import sector_filename, sector_index
from core.road_layout.sector import MapNode, parse_sector

# Directory hashes from base_map.scs. CityHash 1.1 does not produce these.
_EMPTY = 0x9AE16A3B2F90404F
_MAP = 0x3543EC1D70156653
_EUROPE = 0x23D3841ECE787493


def test_cityhash_matches_the_archive_directories():
    assert city_hash64_path("") == _EMPTY
    assert city_hash64_path("map") == _MAP
    assert city_hash64_path("map/europe") == _EUROPE


def test_sector_index_and_filename():
    assert sector_index(0.0, 0.0) == (0, 0)
    assert sector_index(3999.9, -0.1) == (0, -1)
    assert sector_index(4000.0, -4000.0) == (1, -1)
    assert sector_filename(0, -2) == "sec+0000-0002.base"
    assert sector_filename(-3, 1) == "sec-0003+0001.base"


def test_straight_centerline_is_the_chord():
    start = MapNode(1, 0.0, 5.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    end = MapNode(2, 0.0, 5.0, -100.0, 1.0, 0.0, 0.0, 0.0)
    pts = sample_centerline(start, end, steps=4)
    assert pts[0][0] == 0.0 and pts[0][2] == 0.0
    assert pts[-1][2] == -100.0
    # t = 0.25. A zero tangent would land at 15.625 m, not 25.
    assert abs(pts[1][0]) < 1e-6
    assert abs(pts[1][2] + 25.0) < 1e-4
    assert abs(pts[1][1] - 5.0) < 1e-6


def test_synthetic_sector_roundtrip_and_offset():
    blob = _sector_with_one_road()
    mesh = parse_sector(blob)
    assert mesh.version == 907
    assert len(mesh.roads) == 1
    assert mesh.roads[0].start_uid == 1
    assert mesh.roads[0].end_uid == 2
    assert abs(mesh.nodes[2].z + 100.0) < 1e-6
    assert abs(mesh.nodes[2].y - 5.0) < 1e-6
    on_line = nearest_centerline(mesh, 0.0, -25.0)
    assert on_line is not None
    assert on_line.distance_m < 0.05
    assert abs(on_line.road_y - 5.0) < 0.05
    beside = nearest_centerline(mesh, 4.0, -25.0)
    assert beside is not None
    assert abs(beside.distance_m - 4.0) < 0.05


def _u16(n: int) -> bytes:
    return int(n).to_bytes(2, "little")


def _u32(n: int) -> bytes:
    return int(n).to_bytes(4, "little")


def _i16(n: int) -> bytes:
    return int(n).to_bytes(2, "little", signed=True)


def _i32(n: int) -> bytes:
    return int(n).to_bytes(4, "little", signed=True)


def _u64(n: int) -> bytes:
    return int(n).to_bytes(8, "little")


def _f32(n: float) -> bytes:
    return struct.pack("<f", n)


def _sector_with_one_road() -> bytes:
    header = _u64(10) + _f32(0.0) * 10 + _u32(0) + bytes([0])
    road = bytes([0, 0]) + _u16(0) + _u64(0) * 9
    road += _u64(0) + _f32(0.0) + _u64(0) + _f32(0.0)
    road += _u64(0) * 3
    for _ in range(3):
        road += _u64(0) + _i16(0) + _u64(0) + _i16(0)
    road += _i32(0) + _i32(0) + _u64(1) + _u64(2) + _f32(100.0)
    item = _u32(3) + header + road
    nodes = _node(1, 0.0, 5.0, 0.0) + _node(2, 0.0, 5.0, -100.0)
    return _u32(907) + _u64(0) + _u32(0) + _u32(1) + item + _u32(2) + nodes + _u32(0)


def _node(uid: int, x: float, y: float, z: float) -> bytes:
    return (
        _u64(uid)
        + _i32(round(x * 256))
        + _i32(round(y * 256))
        + _i32(round(z * 256))
        + _f32(1.0) + _f32(0.0) + _f32(0.0) + _f32(0.0)
        + _u64(0) + _u64(0) + _u32(0)
    )
