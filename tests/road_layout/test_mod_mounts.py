"""Mounted map mods: the game log, the directory package, and the base/ fallback."""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

from core.road_layout.archive import DirArchive, open_mounted
from core.road_layout.game_map import GameMap
from core.road_layout.mod_mounts import ModLogScan, ModOverlayOpener, mounts_in
from core.sdk_installer.game_paths import GAME_CONFIG
from tools.map_replay import mod_map_fits


def _sector(node_uid: int) -> bytes:
    node = struct.pack("<Q3i4fQQI", node_uid, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0, 0, 0)
    return struct.pack("<IQIII", 907, 0, 0, 0, 1) + node + struct.pack("<I", 0)


def _user(tmp_path: Path) -> tuple[Path, Path]:
    user = tmp_path / "Euro Truck Simulator 2"
    mod = user / "mod"
    mod.mkdir(parents=True)
    return user, mod


def test_the_log_keeps_mount_order_and_drops_the_previous_profile(tmp_path):
    user, mod = _user(tmp_path)
    first = mod / "first.scs"
    second = mod / "second.scs"
    first.write_bytes(b"PK")
    second.write_bytes(b"PK")
    text = "\n".join([
        f"[fs] device {first.as_posix()} mounted to mod pool.",
        "New profile selected: 'Driver'",
        f"[fs] device {second.as_posix()} mounted to mod pool.",
        "[fs] device somewhere/effect.scs mounted to scs pool.",
    ])
    assert mounts_in(text, user) == [second]


def test_a_path_outside_the_mod_folder_is_ignored(tmp_path):
    user, mod = _user(tmp_path)
    kept = mod / "kept.scs"
    kept.write_bytes(b"x")
    outside = tmp_path / "secret.scs"
    outside.write_bytes(b"x")
    escaped = mod / ".." / "secret.scs"
    text = "\n".join([
        f"[fs] device {outside.as_posix()} mounted to mod pool.",
        f"[fs] device {escaped.as_posix()} mounted to mod pool.",
        f"[fs] device {kept.as_posix()} mounted to mod pool.",
    ])
    assert mounts_in(text, user) == [kept]


def test_a_workshop_package_for_this_game_is_allowed(tmp_path):
    user, _mod = _user(tmp_path)
    app = GAME_CONFIG["ets2"]["steam_id"]
    pack = tmp_path / "steam" / "workshop" / "content" / app / "9" / "universal"
    pack.mkdir(parents=True)
    text = f"[fs] device {pack.as_posix()} mounted to mod pool."
    assert mounts_in(text, user) == [pack]


def test_a_mount_split_across_reads_is_kept(tmp_path):
    user, mod = _user(tmp_path)
    pack = mod / "map.scs"
    pack.write_bytes(b"PK")
    line = f"[fs] device {pack.as_posix()} mounted to mod pool.\n"
    from core.road_layout import mod_mounts

    pad = b"#.\n" * ((mod_mounts._SLICE // 3) + 10)
    log_path = user / "game.log.txt"
    log_path.write_bytes(pad + line.encode())
    scan = ModLogScan(log_path, user)
    assert scan.step() is False
    while not scan.step():
        pass
    assert scan.mounts == [pack]


def test_a_directory_package_rejects_a_path_that_leaves_it(tmp_path):
    root = tmp_path / "pkg"
    (root / "map" / "europe").mkdir(parents=True)
    sector = root / "map" / "europe" / "sec+0000+0000.base"
    sector.write_bytes(b"sector")
    archive = DirArchive(root)
    assert archive.get("map/europe/sec+0000+0000.base") == b"sector"
    assert archive.get("../secret") is None
    assert archive.listdir("map/europe")[1] == ["sec+0000+0000.base"]


def test_a_package_with_the_map_under_base_replaces_the_sector(tmp_path):
    key = "map/europe/sec+0000+0000.base"
    path = tmp_path / "map.scs"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("base/" + key, _sector(4))
    mounted = open_mounted(path)
    assert mounted.get(key) == _sector(4)

    class _Fake:
        def __init__(self, sectors):
            self._sectors = sectors

        def get(self, name):
            return self._sectors.get(name)

        def has(self, name):
            return name in self._sectors

        def listdir(self, name):
            return ([], []) if name == "map/europe" else None

    base = GameMap("europe", [_Fake({key: _sector(1)})], [_Fake({})])
    noted = []
    overlay = base.with_overlay([mounted], mark_added=False)
    overlay._note_added = lambda *args: noted.append(args)
    assert set(overlay.sector((0, 0)).nodes) == {4}
    assert noted == []
    assert set(base.sector((0, 0)).nodes) == {1}


def test_the_opener_skips_a_package_it_cannot_read(tmp_path):
    junk = tmp_path / "junk.scs"
    junk.write_bytes(b"not an archive")
    opener = ModOverlayOpener("ets2", [junk])
    while not opener.step():
        pass
    assert opener.archives == []


def test_mod_map_is_used_only_when_the_stock_map_misses():
    assert mod_map_fits(10, 20, 30)
    assert not mod_map_fits(20, 30, 30)
    assert not mod_map_fits(10, 15, 30)
    assert not mod_map_fits(0, 1, 0)
