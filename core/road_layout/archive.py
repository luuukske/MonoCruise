"""Readers for SCS `.scs` archives (HashFS v2) and TruckersMP `.mp` archives.

Enough to open a path and list a directory. Image metadata and gdeflate are
not handled: map sectors in current ETS2 archives are zlib or raw. A `.mp` file
is HashFS v1 (plain zlib entries, text directories) or an ordinary zip. Neither
is encrypted; nothing here decrypts anything, and a file is never held open
between reads so the TruckersMP launcher can replace it.
"""

from __future__ import annotations

import struct
import zlib
from bisect import bisect_left
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
        self._hashes: list[int] = []
        self._ent = b""
        self._meta = b""
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
        # Entries are sorted by hash; decode one only when it is looked up.
        self._hashes = memoryview(ent).cast("Q")[::2].tolist() if count else []
        self._ent = ent
        self._meta = meta

    def _entry(self, key: str) -> _Entry | None:
        hash_ = city_hash64_path(key)
        i = bisect_left(self._hashes, hash_)
        if i >= len(self._hashes) or self._hashes[i] != hash_:
            return None
        _h, meta_index, meta_count, flags, _some = _ENTRY.unpack_from(self._ent, i * _ENTRY.size)
        if meta_count != 1:
            return None
        hdr = int.from_bytes(self._meta[4 * meta_index:4 * meta_index + 4], "little")
        index = hdr & 0xFFFFFF
        packed, size, _unk, rel = _PLAIN.unpack_from(self._meta, 4 * index)
        return _Entry(
            is_dir=bool(flags & 1),
            compression=(packed >> 28) & 0xF,
            compressed_size=packed & 0x0FFFFFFF,
            size=size & 0x0FFFFFFF,
            offset=(rel & 0xFFFFFFFF) * 16,
        )

    def has(self, key: str) -> bool:
        entry = self._entry(key)
        return entry is not None and not entry.is_dir

    def get(self, key: str) -> bytes | None:
        """File bytes for *key*, or None when the archive has no such path."""
        entry = self._entry(key)
        if entry is None or entry.is_dir:
            return None
        with self.path.open("rb") as fh:
            fh.seek(entry.offset)
            blob = fh.read(entry.compressed_size)
        return _decompress(blob, entry.compression, entry.size)

    def listdir(self, key: str) -> tuple[list[str], list[str]] | None:
        """`(subdirectories, files)` for *key*. Root is the empty string."""
        entry = self._entry(key)
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


_V1_HEADER = struct.Struct("<4sI4sII")
_V1_ENTRY = struct.Struct("<QQIIII")
_V1_DIR = 0x1
_V1_ZLIB = 0x2
_MAX_ENTRIES = 1_000_000
_MAX_ENTRY_BYTES = 64 * 1024 * 1024


class HashFsV1:
    """One TruckersMP `.mp` archive in HashFS version 1, indexed by CityHash64 path."""

    def __init__(self, path: Path) -> None:
        self.path = path
        with path.open("rb") as fh:
            total = fh.seek(0, 2)
            fh.seek(0)
            magic, version, method, count, start = _V1_HEADER.unpack(fh.read(_V1_HEADER.size))
            if magic != b"SCS#" or version != 1 or method != b"CITY":
                raise ValueError(f"not a HashFS v1 CITY archive: {path.name}")
            if count > _MAX_ENTRIES or start + _V1_ENTRY.size * count > total:
                raise ValueError(f"HashFS v1 table out of range: {path.name}")
            fh.seek(start)
            table = fh.read(_V1_ENTRY.size * count)
        self._size = total
        self._entries = sorted(_V1_ENTRY.iter_unpack(table))
        self._hashes = [entry[0] for entry in self._entries]

    def _find(self, key: str):
        hash_ = city_hash64_path(key)
        i = bisect_left(self._hashes, hash_)
        if i >= len(self._hashes) or self._hashes[i] != hash_:
            return None
        return self._entries[i]

    def _read(self, entry) -> bytes:
        _hash, offset, flags, _crc, size, packed = entry
        if size > _MAX_ENTRY_BYTES or offset + packed > self._size:
            raise ValueError("HashFS v1 entry out of range")
        with self.path.open("rb") as fh:
            fh.seek(offset)
            blob = fh.read(packed)
        data = _bounded_inflate(blob, size) if flags & _V1_ZLIB else blob
        if len(data) != size:
            raise ValueError("HashFS v1 entry size mismatch")
        return data

    def has(self, key: str) -> bool:
        entry = self._find(key)
        return entry is not None and not entry[2] & _V1_DIR

    def get(self, key: str) -> bytes | None:
        """File bytes for *key*, or None when the archive has no such path."""
        entry = self._find(key)
        if entry is None or entry[2] & _V1_DIR:
            return None
        return self._read(entry)

    def listdir(self, key: str) -> tuple[list[str], list[str]] | None:
        """`(subdirectories, files)` for *key*; a v1 directory is lines of text, `*` marks a subdirectory."""
        entry = self._find(key)
        if entry is None or not entry[2] & _V1_DIR:
            return None
        subdirs: list[str] = []
        files: list[str] = []
        for line in self._read(entry).decode("utf-8", "replace").splitlines():
            if line.startswith("*"):
                subdirs.append(line[1:])
            elif line:
                files.append(line)
        return subdirs, files


_ZIP_LOCAL = struct.Struct("<4s5H3I2H")
_ZIP_MAX_ENTRIES = 200_000


class ZipArchive:
    """One TruckersMP `.mp` archive that is a plain zip, read without keeping the file open."""

    def __init__(self, path: Path) -> None:
        import zipfile

        self.path = path
        try:
            with zipfile.ZipFile(path) as zf:
                infos = zf.infolist()
        except zipfile.BadZipFile as exc:
            raise ValueError(f"not a zip archive: {path.name}") from exc
        if len(infos) > _ZIP_MAX_ENTRIES:
            raise ValueError(f"zip has too many entries: {path.name}")
        self._files: dict[str, tuple[int, int, int, int]] = {}
        self._dirs: dict[str, tuple[set[str], set[str]]] = {"": (set(), set())}
        for info in infos:
            name = info.filename.strip("/").lower()
            if info.is_dir() or not name or info.flag_bits & 0x1:
                continue
            self._files[name] = (info.header_offset, info.compress_size, info.file_size, info.compress_type)
            parent = ""
            parts = name.split("/")
            for part in parts[:-1]:
                self._dirs.setdefault(parent, (set(), set()))[0].add(part)
                parent = f"{parent}/{part}" if parent else part
            self._dirs.setdefault(parent, (set(), set()))[1].add(parts[-1])

    def has(self, key: str) -> bool:
        return key.strip("/").lower() in self._files

    def get(self, key: str) -> bytes | None:
        entry = self._files.get(key.strip("/").lower())
        if entry is None:
            return None
        offset, packed, size, method = entry
        if size > _MAX_ENTRY_BYTES:
            raise ValueError("zip entry too large")
        with self.path.open("rb") as fh:
            fh.seek(offset)
            head = _ZIP_LOCAL.unpack(fh.read(_ZIP_LOCAL.size))
            if head[0] != b"PK":
                raise ValueError("zip entry header missing")
            fh.seek(head[9] + head[10], 1)
            blob = fh.read(packed)
        if method == 0:
            data = blob
        elif method == 8:
            data = _bounded_inflate(blob, size, raw=True)
        else:
            raise ValueError(f"unsupported zip method {method}")
        if len(data) != size:
            raise ValueError("zip entry size mismatch")
        return data

    def listdir(self, key: str) -> tuple[list[str], list[str]] | None:
        listing = self._dirs.get(key.strip("/").lower())
        if listing is None:
            return None
        return sorted(listing[0]), sorted(listing[1])


Archive = HashFs | HashFsV1 | ZipArchive


def open_archive(path: Path) -> Archive:
    """The reader for a `.scs` or `.mp` file, chosen by its first bytes. Raises ``ValueError``."""
    with path.open("rb") as fh:
        head = fh.read(8)
    if head[:4] == b"PK":
        return ZipArchive(path)
    if head[:4] == b"SCS#" and int.from_bytes(head[4:8], "little") == 1:
        return HashFsV1(path)
    return HashFs(path)


def _bounded_inflate(blob: bytes, size: int, raw: bool = False) -> bytes:
    """Inflate *blob*, never producing more than the declared *size* (a corrupt file cannot balloon)."""
    try:
        inflater = zlib.decompressobj(-15 if raw else 15)
        data = inflater.decompress(blob, size + 1)
    except zlib.error as exc:
        raise ValueError("corrupt compressed entry") from exc
    if len(data) > size:
        raise ValueError("compressed entry larger than declared")
    return data
