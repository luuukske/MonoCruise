"""Older sector versions from TruckersMP mods: the three layouts that differ, built byte by byte."""

from __future__ import annotations

import struct

import pytest

from core.road_layout.sector import SectorError, parse_sector

_HEADER_BODY = bytes(40) + struct.pack("<IB", 0, 0)


def _item(typ: int, uid: int, payload: bytes) -> bytes:
    return struct.pack("<IQ", typ, uid) + _HEADER_BODY + payload


def _sector(version: int, *items: bytes) -> bytes:
    head = struct.pack("<IQII", version, 0, 0, len(items))
    return head + b"".join(items) + struct.pack("<II", 0, 0)


def _sign(version: int) -> bytes:
    body = struct.pack("<QQQQB", 1, 2, 0, 0, 0) + struct.pack("<Q", 3) + b"abc"
    boards = struct.pack("<I", 0) if version >= 906 else b""
    return _item(36, 7, body + boards + struct.pack("<I", 0))


def _curve(version: int) -> bytes:
    head = struct.pack("<QQ", 11, 12) + bytes(16) + struct.pack("<f", 50.0)
    tail = struct.pack("<I3f", 5, 1.0, 1.0, 0.0) + bytes(8) + struct.pack("<If", 0, 0.0) + bytes(32)
    tail += struct.pack("<I", 2) + bytes(8)
    if version >= 907:
        entry = bytes(8) + struct.pack("<I", 0) + tail + bytes(20)
        return _item(44, 8, head + struct.pack("<I", 1) + entry)
    return _item(44, 8, struct.pack("<Q", 9) + head + tail)


def _terrain(version: int) -> bytes:
    body = struct.pack("<QQ", 1, 2) + bytes(24) + struct.pack("<ff", 0.0, 0.0)
    if version >= 902:
        body += struct.pack("<I", 0)
    body += bytes(30)
    for _ in range(2):
        body += struct.pack("<H", 0) + bytes(8) + struct.pack("<f", 0.0) + bytes(8) + struct.pack("<f", 0.0)
        body += (bytes(8) + struct.pack("<HBBHH", 0, 0, 0, 0, 0)) * 3
        body += struct.pack("<HH", 0, 0)
    quad = struct.pack("<HHHHIII", 0, 0, 0, 0, 0, 0, 0)
    body += struct.pack("<I", 0) + quad + quad + bytes(32)
    return _item(1, 6, body)


@pytest.mark.parametrize("version", [896, 898, 900, 901, 903, 905, 906, 907])
def test_each_layout_reads_its_own_items_exactly(version):
    blob = _sector(version, _sign(version), _curve(version), _terrain(version), _sign(version))
    assert parse_sector(blob).version == version


@pytest.mark.parametrize("maker, wrong, right", [
    (_sign, 906, 905),
    (_curve, 907, 906),
    (_terrain, 903, 901),
])
def test_a_layout_read_as_the_wrong_version_is_a_layout_miss(maker, wrong, right):
    with pytest.raises(SectorError):
        parse_sector(_sector(right, maker(wrong), maker(wrong)))
    parse_sector(_sector(wrong, maker(wrong), maker(wrong)))


@pytest.mark.parametrize("version", [895, 897, 899, 902, 904])
def test_versions_nobody_checked_are_refused(version):
    with pytest.raises(SectorError, match="not verified"):
        parse_sector(_sector(version))


def test_a_newer_version_still_uses_the_current_layout():
    assert parse_sector(_sector(908, _sign(907))).version == 908
