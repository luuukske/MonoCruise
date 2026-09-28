"""Centerline geometry and a synthetic sector, no game files."""

from __future__ import annotations

import struct

from core.road_layout.centerline import nearest_centerline, sample_centerline
from core.road_layout.cityhash import city_hash64_path
from core.road_layout.grid import sector_filename, sector_index, sectors_touching
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


def test_sectors_touching_grows_only_at_the_edge():
    assert sectors_touching(2000.0, 2000.0, 100.0) == [(0, 0)]
    assert sectors_touching(3990.0, 2000.0, 50.0) == [(0, 0), (1, 0)]
    corner = sectors_touching(3990.0, 10.0, 50.0)
    assert (0, 0) in corner and (1, 0) in corner and (0, -1) in corner and (1, -1) in corner


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


def test_one_plus_one_lanes_sit_half_a_lane_off_the_spline():
    from core.road_layout.lanes import lane_lines, nearest_lane
    from core.road_layout.looks import RoadLook
    from core.road_layout.sector import MapNode, MapRoad, SectorMesh

    look = RoadLook("road.look0", ["traffic_lane.road.local"], ["traffic_lane.road.local"])
    assert look.laterals() == [2.25, -2.25]
    wide = RoadLook(
        "ger31",
        ["a", "a"],
        ["a", "a"],
        offset_m=5.75,
        shift_left=[-4.75, -4.75],
        shift_right=[-4.75, -4.75],
    )
    assert wide.laterals() == [3.25, 7.75, -3.25, -7.75]
    gapped = RoadLook("gap", ["a"], ["a"], offset_m=1.0)
    assert gapped.laterals() == [3.25, -3.25]
    start = MapNode(1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    end = MapNode(2, 0.0, 0.0, -100.0, 1.0, 0.0, 0.0, 0.0)
    mesh = SectorMesh(nodes={1: start, 2: end}, roads=[
        MapRoad(10, "road.look0", 1, 2, 100.0),
    ])
    lines = lane_lines(mesh, {"road.look0": look}, None, steps=4)
    assert len(lines) == 2
    hit = nearest_lane(lines, 2.25, -25.0)
    assert hit is not None
    assert hit.distance_m < 0.05
    assert hit.kind == "road"


def test_a_long_road_has_more_points_than_a_short_one():
    from core.road_layout.lanes import lane_lines
    from core.road_layout.sector import MapNode, MapRoad, SectorMesh

    def _points(length: float) -> int:
        start = MapNode(1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        end = MapNode(2, 0.0, 0.0, -length, 1.0, 0.0, 0.0, 0.0)
        mesh = SectorMesh(nodes={1: start, 2: end}, roads=[
            MapRoad(10, "look", 1, 2, length),
        ])
        lines = lane_lines(mesh, {}, None, steps=8)
        return len(lines[0].points)

    assert _points(400.0) > _points(40.0)


def test_one_way_odd_counts_sit_half_a_lane_left():
    from core.road_layout.looks import RoadLook

    assert RoadLook("ow1", [], ["a"]).laterals() == [-2.25]
    assert RoadLook("ow2", [], ["a", "b"]).laterals() == [-2.25, 2.25]
    assert RoadLook("ow3", [], ["a", "b", "c"]).laterals() == [-6.75, -2.25, 2.25]
    assert RoadLook("ow4", [], ["a"] * 4).laterals() == [-6.75, -2.25, 2.25, 6.75]
    assert RoadLook("ow5", [], ["a"] * 5).laterals() == [-11.25, -6.75, -2.25, 2.25, 6.75]
    assert RoadLook("shift", [], ["a"], shift_right=[1.75]).laterals() == [-0.5]
    assert RoadLook("left", ["a"], []).laterals() == [2.25]


def test_prefab_curve_bytes_roundtrip():
    import struct

    from core.road_layout.prefab import _read_ppd

    node = bytearray(104)
    struct.pack_into("<3f", node, 16, 10.0, 1.0, 5.0)
    struct.pack_into("<3f", node, 28, 1.0, 0.0, 0.0)
    curve = bytearray(132)
    struct.pack_into("<3f", curve, 16, 1.0, 2.0, 3.0)
    struct.pack_into("<3f", curve, 28, 4.0, 5.0, 6.0)
    struct.pack_into("<4f", curve, 40, 1.0, 0.0, 0.0, 0.0)
    struct.pack_into("<4f", curve, 56, 1.0, 0.0, 0.0, 0.0)
    blob = struct.pack("<I 11I 12I", 25, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 96, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    blob += bytes(node) + bytes(curve)
    desc = _read_ppd(blob)
    assert desc is not None
    assert abs(desc.nodes[0].x - 10.0) < 1e-5
    assert abs(desc.nodes[0].z - 5.0) < 1e-5
    assert abs(desc.curves[0].x0 - 1.0) < 1e-5
    assert abs(desc.curves[0].z0 - 3.0) < 1e-5
    assert abs(desc.curves[0].y0 - 2.0) < 1e-5


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
