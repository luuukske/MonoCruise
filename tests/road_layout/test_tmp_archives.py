"""TruckersMP `.mp` archives: both containers, the file lookup, and the layering over the base map."""

from __future__ import annotations

import json
import os
import struct
import zipfile
import zlib

import pytest

from core.road_layout import tmp_mods
from core.road_layout.archive import HashFsV1, ZipArchive, open_archive
from core.road_layout.cityhash import city_hash64_path
from core.road_layout.game_map import GameMap


def _write_v1(path, files: dict[str, bytes], dirs: dict[str, str] | None = None) -> None:
    """HashFS v1 as TruckersMP ships it: zlib files, text directories, a 32 byte entry table."""
    entries = []
    blobs = b""
    base = 0x20 + 32 * (len(files) + len(dirs or {}))
    for key, data in files.items():
        packed = zlib.compress(data)
        entries.append((city_hash64_path(key), base + len(blobs), 0x6, 0, len(data), len(packed)))
        blobs += packed
    for key, text in (dirs or {}).items():
        raw = text.encode()
        entries.append((city_hash64_path(key), base + len(blobs), 0x1, 0, len(raw), len(raw)))
        blobs += raw
    entries.sort()
    table = b"".join(struct.pack("<QQIIII", *e) for e in entries)
    head = struct.pack("<4sI4sII", b"SCS#", 1, b"CITY", len(entries), 0x20)
    path.write_bytes(head + b"\0" * (0x20 - len(head)) + table + blobs)


def _sector(node_uid: int) -> bytes:
    """The smallest valid version-907 sector: no items, one node, no trailing ids."""
    node = struct.pack("<Q3i4fQQI", node_uid, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0)
    return struct.pack("<IQIII", 907, 0, 0, 0, 1) + node + struct.pack("<I", 0)


def test_hashfs_v1_reads_files_and_directories(tmp_path):
    path = tmp_path / "a.mp"
    _write_v1(path, {"map/europe/sec+0001-0004.base": b"sector"}, {"map/europe": "sec+0001-0004.base\n*sub\n"})
    archive = HashFsV1(path)
    assert archive.get("map/europe/sec+0001-0004.base") == b"sector"
    assert archive.get("map/europe/missing") is None
    assert archive.listdir("map/europe") == (["sub"], ["sec+0001-0004.base"])
    assert archive.listdir("map/europe/sec+0001-0004.base") is None


def test_hashfs_v1_is_not_held_open(tmp_path):
    path = tmp_path / "a.mp"
    _write_v1(path, {"x/y": b"data"})
    archive = HashFsV1(path)
    assert archive.get("x/y") == b"data"
    path.unlink()


def test_hashfs_v1_rejects_corrupt_entries(tmp_path):
    path = tmp_path / "a.mp"
    _write_v1(path, {"x/y": b"data"})
    raw = bytearray(path.read_bytes())
    raw[-3:] = b"\xff\xff\xff"
    path.write_bytes(bytes(raw))
    with pytest.raises(ValueError):
        HashFsV1(path).get("x/y")


def test_hashfs_v1_refuses_a_table_past_the_file(tmp_path):
    path = tmp_path / "a.mp"
    path.write_bytes(struct.pack("<4sI4sII", b"SCS#", 1, b"CITY", 500, 0x20))
    with pytest.raises(ValueError):
        HashFsV1(path)


def test_archives_can_say_whether_a_path_exists_without_reading_it(tmp_path):
    v1 = tmp_path / "a.mp"
    _write_v1(v1, {"x/y": b"1"}, {"x": "y\n"})
    z = tmp_path / "b.mp"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("x/y", b"1")
    for archive in (HashFsV1(v1), ZipArchive(z)):
        assert archive.has("x/y") and not archive.has("x/missing") and not archive.has("x")


def test_zip_archive_reads_and_lists_case_insensitively(tmp_path):
    path = tmp_path / "z.mp"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("map/europe/sec+0001-0004.base", b"deflated" * 50, zipfile.ZIP_DEFLATED)
        zf.writestr("def/world/road_look.tmp.sii", b"stored", zipfile.ZIP_STORED)
    archive = ZipArchive(path)
    assert archive.get("map/europe/sec+0001-0004.base") == b"deflated" * 50
    assert archive.get("DEF/world/road_look.tmp.sii") == b"stored"
    assert archive.get("nothing") is None
    assert archive.listdir("map") == (["europe"], [])
    assert archive.listdir("map/europe") == ([], ["sec+0001-0004.base"])
    assert archive.listdir("absent") is None
    path.unlink()


def test_zip_archive_skips_encrypted_entries(tmp_path):
    path = tmp_path / "z.mp"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("a/b", b"x")
    raw = bytearray(path.read_bytes())
    flag_at = raw.index(b"PK\x01\x02") + 8
    raw[flag_at] |= 1
    path.write_bytes(bytes(raw))
    assert ZipArchive(path).get("a/b") is None


def test_open_archive_picks_the_container(tmp_path):
    v1 = tmp_path / "a.mp"
    _write_v1(v1, {"x": b"1"})
    z = tmp_path / "b.mp"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("x", b"1")
    assert isinstance(open_archive(v1), HashFsV1)
    assert isinstance(open_archive(z), ZipArchive)
    junk = tmp_path / "c.mp"
    junk.write_bytes(b"not an archive at all, just text")
    with pytest.raises((ValueError, struct.error)):
        open_archive(junk)


def _install(tmp_path, monkeypatch, names_and_ages: dict[str, int], listed: list[str] | None = None):
    root = tmp_path / "TruckersMP" / "installation"
    mods = root / "data" / "ets2" / "mods"
    mods.mkdir(parents=True)
    for name, age in names_and_ages.items():
        (mods / name).write_bytes(b"x")
        os.utime(mods / name, (1_700_000_000 - age, 1_700_000_000 - age))
    cache = {f"/data/ets2/mods/{n}": {"Md5": "0", "Type": 0} for n in (names_and_ages if listed is None else listed)}
    (root / "filecache.dat").write_text(json.dumps(cache), encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    return mods


def test_mod_paths_run_oldest_first(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, {"new.mp": 10, "old.mp": 5000, "mid.mp": 700, "notes.txt": 1})
    assert [p.name for p in tmp_mods.tmp_mod_paths("ets2")] == ["old.mp", "mid.mp", "new.mp"]
    assert tmp_mods.tmp_mod_paths("ats") == []


def test_only_mods_the_launcher_lists_count(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, {"current.mp": 10, "old_event.mp": 5000, "gone.mp": 20},
             listed=["current.mp", "gone.mp", "missing_on_disk.mp"])
    (tmp_path / "TruckersMP" / "installation" / "data" / "ets2" / "mods" / "gone.mp").unlink()
    assert [p.name for p in tmp_mods.tmp_mod_paths("ets2")] == ["current.mp"]


def test_shared_mods_are_included(tmp_path, monkeypatch):
    mods = _install(tmp_path, monkeypatch, {"a.mp": 10})
    shared = mods.parent.parent / "shared" / "mods"
    shared.mkdir(parents=True)
    (shared / "s.mp").write_bytes(b"x")
    cache = tmp_path / "TruckersMP" / "installation" / "filecache.dat"
    cache.write_text(json.dumps({"/data/ets2/mods/a.mp": {}, "/data/shared/mods/s.mp": {}}), encoding="utf-8")
    assert {p.name for p in tmp_mods.tmp_mod_paths("ets2")} == {"a.mp", "s.mp"}


def test_an_unreadable_file_list_means_no_archives(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, {"a.mp": 10})
    cache = tmp_path / "TruckersMP" / "installation" / "filecache.dat"
    for text in ("not json", "[1, 2]", ""):
        cache.write_text(text, encoding="utf-8")
        assert tmp_mods.tmp_mod_paths("ets2") == []
    cache.unlink()
    assert tmp_mods.tmp_mod_paths("ets2") == []


def test_list_entries_cannot_climb_out_of_the_install(tmp_path, monkeypatch):
    mods = _install(tmp_path, monkeypatch, {"a.mp": 10}, listed=[])
    (mods.parent.parent.parent / "outside.mp").write_bytes(b"x")
    cache = tmp_path / "TruckersMP" / "installation" / "filecache.dat"
    cache.write_text(json.dumps({"/data/ets2/mods/../../../outside.mp": {}}), encoding="utf-8")
    assert tmp_mods.tmp_mod_paths("ets2") == []


def test_mod_paths_follow_the_launcher_install_path(tmp_path, monkeypatch):
    _install(tmp_path, monkeypatch, {"default.mp": 1})
    other = tmp_path / "elsewhere"
    (other / "data" / "ets2" / "mods").mkdir(parents=True)
    (other / "data" / "ets2" / "mods" / "moved.mp").write_bytes(b"x")
    (other / "filecache.dat").write_text(json.dumps({"/data/ets2/mods/moved.mp": {}}), encoding="utf-8")
    (tmp_path / "TruckersMP" / "launcher-options.json").write_text(
        json.dumps({"installPath": str(other)}), encoding="utf-8")
    assert [p.name for p in tmp_mods.tmp_mod_paths("ets2")] == ["moved.mp"]


def test_no_launcher_means_no_archives(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path))
    assert tmp_mods.tmp_mod_paths("ets2") == []
    monkeypatch.delenv("APPDATA")
    assert tmp_mods.tmp_mod_paths("ets2") == []


def test_overlay_opener_skips_unreadable_archives(tmp_path, monkeypatch):
    mods = _install(tmp_path, monkeypatch, {"junk.mp": 100, "good.mp": 50})
    _write_v1(mods / "good.mp", {"x": b"1"})
    opener = tmp_mods.TmpOverlayOpener("ets2")
    while not opener.step():
        pass
    assert len(opener.archives) == 1


class _Fake:
    def __init__(self, sectors: dict[str, bytes]):
        self._sectors = sectors

    def get(self, key):
        return self._sectors.get(key)

    def has(self, key):
        return key in self._sectors

    def listdir(self, key):
        return ([], []) if key == "map/europe" else None


def test_sector_blob_stays_valid_for_the_fixture():
    from core.road_layout.sector import parse_sector

    assert set(parse_sector(_sector(7)).nodes) == {7}


def test_overlay_sector_replaces_the_base_sector():
    key = "map/europe/sec+0000+0000.base"
    base = GameMap("europe", [_Fake({key: _sector(1)})], [_Fake({})])
    overlay = base.with_overlay([_Fake({key: _sector(2)})])
    assert set(base.sector((0, 0)).nodes) == {1}
    assert set(overlay.sector((0, 0)).nodes) == {2}


def test_unreadable_overlay_sector_leaves_no_lanes_not_the_base_roads():
    key = "map/europe/sec+0000+0000.base"
    base = GameMap("europe", [_Fake({key: _sector(1)})], [_Fake({})])
    overlay = base.with_overlay([_Fake({key: b"\x8b\x03\x00\x00garbage"})])
    assert overlay.sector((0, 0)).nodes == {}
    assert (0, 0) in overlay.failed_sectors
    assert set(base.sector((0, 0)).nodes) == {1}


def test_lane_map_loader_swaps_to_the_overlay_and_back():
    from core.road_layout.live import LiveMap

    key = "map/europe/sec+0000+0000.base"
    live = LiveMap("europe", [_Fake({key: _sector(1)})], [_Fake({})])
    live._ensure((0, 0))
    assert set(live._sectors[(0, 0)].nodes) == {1} and not live.overlaid
    live.set_overlay([_Fake({key: _sector(2)})])
    live._ensure((0, 0))
    assert set(live._sectors[(0, 0)].nodes) == {2} and live.overlaid
    live.set_overlay([])
    live._ensure((0, 0))
    assert set(live._sectors[(0, 0)].nodes) == {1} and not live.overlaid
