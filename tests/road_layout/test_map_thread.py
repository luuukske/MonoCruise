"""Map worker states with a stub map: no archives, no game, no registry siblings."""

from __future__ import annotations

import time

import core.map_thread.thread as mt
from core.road_layout.ego_lane import LineIndex
from core.road_layout.game_map import LaneWindow, pack_lane
from core.road_layout.lanes import LaneLine


class _StubMap:
    def __init__(self) -> None:
        line = LaneLine("road", "t", tuple((float(x), 0.0, 0.0) for x in range(-300, 701, 10)), None, 1, 0, 1, 0.0)
        lanes = (pack_lane(line),)
        self.window = LaneWindow((line,), (), (-600.0, 600.0, -600.0, 600.0), lanes, LineIndex(lanes))
        self.builds = 0

    def sectors_needed(self, *box):
        return []

    def sector(self, key):
        return None

    def lanes_in_box(self, *box):
        self.builds += 1
        return self.window


def _thread(monkeypatch, *, enabled=True, game="ets2", pose=(0.0, 0.0, 0.4, 1.0, 0.0)):
    monkeypatch.setattr(mt, "_enabled", lambda: enabled)
    monkeypatch.setattr(mt, "_running_game", lambda: game)
    monkeypatch.setattr(mt, "_ego_pose", lambda: pose)
    thread = mt.MapThread()
    thread.running = True
    thread._game = game
    thread._map = _StubMap()
    return thread


def test_ready_snapshot_carries_ego_route(monkeypatch):
    thread = _thread(monkeypatch)
    thread.loop()
    snap = thread.data.snapshot
    assert snap.status == "ready"
    assert snap.route is not None and abs(snap.route.offset_m - 0.4) < 1e-6
    assert snap.fresh(time.monotonic())


def test_window_is_reused_while_ego_stays_inside(monkeypatch):
    thread = _thread(monkeypatch)
    thread.loop()
    thread.loop()
    assert thread._map.builds == 1


def test_disabled_publishes_off(monkeypatch):
    thread = _thread(monkeypatch, enabled=False)
    thread.loop()
    assert thread.data.snapshot.status == "off"
    assert thread.data.snapshot.route is None


def test_no_game_waits(monkeypatch):
    thread = _thread(monkeypatch, game="")
    thread.loop()
    assert thread.data.snapshot.status == "waiting"


def test_stale_or_missing_snapshot_reads_as_no_map(monkeypatch):
    assert mt.map_snapshot() is None or mt.map_snapshot().status == "ready"
    old = mt.MapSnapshot("ready", t_mono=time.monotonic() - 5.0)
    assert not old.fresh(time.monotonic())


class _OverlayMap(_StubMap):
    def with_overlay(self, archives):
        self.overlaid = list(archives)
        overlay = _OverlayMap()
        overlay.is_overlay = True
        return overlay


def _tmp_thread(monkeypatch, *, archives=1, enabled=True):
    thread = _thread(monkeypatch)
    thread._map = thread._base_map = _OverlayMap()
    state = {"signal": True, "opened": 0}
    monkeypatch.setattr(mt, "_tmp_signal", lambda: state["signal"])
    monkeypatch.setattr(mt, "_tmp_enabled", lambda: enabled)

    class _Opener:
        def __init__(self, game):
            state["opened"] += 1
            self.archives = [object()] * archives
            self.done = False

        def step(self):
            self.done = True
            return True

    monkeypatch.setattr(mt, "TmpOverlayOpener", _Opener)
    return thread, state


def test_connecting_to_truckersmp_layers_its_map_additions(monkeypatch):
    thread, _ = _tmp_thread(monkeypatch)
    thread.loop()
    assert thread._map is thread._overlay_map and thread._map is not thread._base_map
    assert getattr(thread._map, "is_overlay", False)
    thread.loop()
    snap = thread.data.snapshot
    assert snap.status == "ready" and snap.tmp


def test_no_truckersmp_signal_keeps_the_base_map(monkeypatch):
    thread, state = _tmp_thread(monkeypatch)
    state["signal"] = False
    thread.loop()
    assert thread._map is thread._base_map and not thread.data.snapshot.tmp
    assert state["opened"] == 0


def test_the_overlay_outlives_a_short_gap_then_drops(monkeypatch):
    thread, state = _tmp_thread(monkeypatch)
    thread.loop()
    state["signal"] = False
    thread.loop()
    assert thread._map is thread._overlay_map
    thread._tmp_seen = time.monotonic() - mt._TMP_HOLD_S - 1.0
    thread.loop()
    assert thread._map is thread._base_map and not thread.data.snapshot.tmp


def test_switch_off_in_config_keeps_the_base_map(monkeypatch):
    thread, state = _tmp_thread(monkeypatch, enabled=False)
    thread.loop()
    assert thread._map is thread._base_map and state["opened"] == 0


def test_no_archives_is_tried_once_and_the_base_map_stays(monkeypatch):
    thread, state = _tmp_thread(monkeypatch, archives=0)
    for _ in range(3):
        thread.loop()
    assert thread._map is thread._base_map and state["opened"] == 1
    assert thread.data.snapshot.status == "ready" and not thread.data.snapshot.tmp


def test_a_broken_overlay_build_keeps_the_base_map(monkeypatch):
    thread, state = _tmp_thread(monkeypatch)

    def boom(archives):
        raise ValueError("bad archive")

    thread._base_map.with_overlay = boom
    for _ in range(3):
        thread.loop()
    assert thread._map is thread._base_map and state["opened"] == 1
    assert thread.data.snapshot.status == "ready"
