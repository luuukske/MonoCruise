"""HashFS v2 reader for SCS `.scs` archives.

Enough to open a path and list a directory. Image metadata and gdeflate are
not handled: map sectors in current ETS2 archives are zlib or raw.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

from core.road_layout.cityhash import city_hash64_path

_HEADER = struct.Struct("<4sHH4sIIIIQQQB")
_ENTRY = struct.Struct("<QIHBB")
_PLAIN = struct.Struct("<IIII")

_COMP_NONE = 0
_COMP_ZLIB = 1


@dataclass(frozen=True)
class _Entry:
    is_dir: bool
    compression: int
    compressed_size: int
    size: int
    offset: int


class HashFs:
    """One `.scs` archive, indexed by CityHash64 path."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[int, _Entry] = {}
        self._load()

    def _load(self) -> None:
        with self.path.open("rb") as fh:
            magic, version, _salt, method, count, ent_c, meta_size, meta_c, ent_off, meta_off, _sec, _plat = (
                _HEADER.unpack(fh.read(_HEADER.size))
            )
            if magic != b"SCS#" or version != 2 or method != b"CITY":
                raise ValueError(f"not a HashFS v2 CITY archive: {self.path.name}")
            ent = _inflate(fh, ent_off, ent_c, _ENTRY.size * count)
            meta = _inflate(fh, meta_off, meta_c, meta_size)
        for i in range(count):
            hash_, meta_index, meta_count, flags, _some = _ENTRY.unpack_from(ent, i * _ENTRY.size)
            if meta_count != 1:
                continue
            hdr = int.from_bytes(meta[4 * meta_index:4 * meta_index + 4], "little")
            index = hdr & 0xFFFFFF
            packed, size, _unk, rel = _PLAIN.unpack_from(meta, 4 * index)
            self._entries[hash_] = _Entry(
                is_dir=bool(flags & 1),
                compression=(packed >> 28) & 0xF,
                compressed_size=packed & 0x0FFFFFFF,
                size=size & 0x0FFFFFFF,
                offset=(rel & 0xFFFFFFFF) * 16,
            )

    def get(self, key: str) -> bytes | None:
        """File bytes for *key*, or None when the archive has no such path."""
        entry = self._entries.get(city_hash64_path(key))
        if entry is None or entry.is_dir:
            return None
        with self.path.open("rb") as fh:
            fh.seek(entry.offset)
            blob = fh.read(entry.compressed_size)
        return _decompress(blob, entry.compression, entry.size)

    def listdir(self, key: str) -> tuple[list[str], list[str]] | None:
        """`(subdirectories, files)` for *key*. Root is the empty string."""
        entry = self._entries.get(city_hash64_path(key))
        if entry is None or not entry.is_dir:
            return None
        with self.path.open("rb") as fh:
            fh.seek(entry.offset)
            blob = fh.read(entry.compressed_size)
        data = _decompress(blob, entry.compression, entry.size)
        count = int.from_bytes(data[:4], "little")
        lengths = data[4:4 + count]
        pos = 4 + count
        subdirs: list[str] = []
        files: list[str] = []
        for length in lengths:
            name = data[pos:pos + length].decode("utf-8", "replace")
            pos += length
            if name.startswith("/"):
                subdirs.append(name[1:])
            else:
                files.append(name)
        return subdirs, files


def _inflate(fh, offset: int, compressed: int, size: int) -> bytes:
    fh.seek(offset)
    blob = fh.read(compressed)
    if len(blob) != compressed:
        raise ValueError("short archive table read")
    return _decompress(blob, _COMP_ZLIB if compressed != size else _COMP_NONE, size)


def _decompress(blob: bytes, compression: int, size: int) -> bytes:
    if compression == _COMP_NONE or len(blob) == size:
        return blob
    if compression == _COMP_ZLIB:
        return zlib.decompress(blob)
    raise ValueError(f"unsupported SCS compression {compression}")
