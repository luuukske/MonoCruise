"""Lanes a TruckersMP road was built on top of: dropped only under something the mod added."""

from __future__ import annotations

import struct

from core.road_layout.buried import drop_buried
from core.road_layout.game_map import GameMap, MapLane


def _lane(uid: int, y: float, x0: float = 0.0, x1: float = 150.0, z: float = 0.0, steps: int = 15) -> MapLane:
    pts = tuple((x0 + (x1 - x0) * i / steps, z) for i in range(steps + 1))
    return MapLane("road", pts, tuple(y for _ in pts), x0, x1, z, z, None, uid, 0, 1, 0.0)


def test_a_lane_under_an_added_one_is_dropped():
    old, new = _lane(1, 40.2), _lane(2, 42.2, z=1.0)
    assert drop_buried([old, new], {2}) == (new,)


def test_nothing_is_dropped_when_the_mod_added_nothing():
    old, new = _lane(1, 40.2), _lane(2, 42.2, z=1.0)
    assert drop_buried([old, new], set()) == (old, new)


def test_only_added_lanes_count_as_the_upper_level():
    base_upper, base_lower = _lane(1, 42.2), _lane(2, 40.2, z=1.0)
    assert drop_buried([base_upper, base_lower], {99}) == (base_upper, base_lower)


def test_a_lane_a_truck_could_pass_under_is_kept():
    deck, road = _lane(2, 47.0, z=1.0), _lane(1, 40.2)
    assert drop_buried([road, deck], {2}) == (road, deck)


def test_a_lane_beside_the_new_road_is_kept():
    old, new = _lane(1, 40.2, z=6.0), _lane(2, 42.2)
    assert drop_buried([old, new], {2}) == (old, new)


def test_a_crossing_is_not_buried():
    ew, ns = _lane(1, 40.2), MapLane("road", ((75.0, -75.0), (75.0, 75.0)), (42.2, 42.2), 75.0, 75.0, -75.0, 75.0, None, 2, 0, 1, 0.0)
    assert drop_buried([ew, ns], {2}) == (ew, ns)


def test_a_short_stub_is_kept():
    stub, new = _lane(1, 40.2, 0.0, 12.0, steps=2), _lane(2, 42.2, z=1.0)
    assert drop_buried([stub, new], {2}) == (stub, new)


def test_a_lane_never_buries_itself():
    new = _lane(2, 42.2)
    assert drop_buried([new], {2}) == (new,)


def _road(uid: int) -> bytes:
    tok = struct.pack("<Q", 0)
    payload = struct.pack("<BBH", 0, 0, 0) + tok * 10 + struct.pack("<f", 0.0) + tok + struct.pack("<f", 0.0) + tok * 3
    payload += (tok + struct.pack("<h", 0) + tok + struct.pack("<h", 0)) * 3
    payload += struct.pack("<ii", 0, 0) + struct.pack("<QQf", 1, 2, 10.0)
    return struct.pack("<IQ", 3, uid) + bytes(40) + struct.pack("<IB", 0, 0) + payload


def _sector(*uids: int) -> bytes:
    head = struct.pack("<IQII", 907, 0, 0, len(uids))
    return head + b"".join(_road(u) for u in uids) + struct.pack("<II", 0, 0)


class _Archive:
    def __init__(self, sectors: dict[str, bytes]):
        self._sectors = sectors

    def get(self, key):
        return self._sectors.get(key)

    def has(self, key):
        return key in self._sectors

    def listdir(self, key):
        return ([], []) if key == "map/europe" else None


def test_an_overlay_sector_notes_only_the_roads_the_base_game_lacks():
    key = "map/europe/sec+0000+0000.base"
    other = "map/europe/sec+0001+0000.base"
    base = GameMap("europe", [_Archive({key: _sector(10, 11), other: _sector(30)})], [_Archive({})])
    overlay = base.with_overlay([_Archive({key: _sector(10, 11, 12)})])
    overlay.sector((0, 0))
    overlay.sector((1, 0))
    assert overlay._added == {12}
    assert base._added == set()


def test_a_map_without_an_overlay_never_drops_lanes():
    key = "map/europe/sec+0000+0000.base"
    base = GameMap("europe", [_Archive({key: _sector(10)})], [_Archive({})])
    base.sector((0, 0))
    assert base._added == set()
