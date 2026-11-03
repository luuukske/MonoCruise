"""LaneDepartureThread wired to fake siblings: settings, missing threads and the command it sends."""

from __future__ import annotations

import struct
import threading
import time
from types import SimpleNamespace

import pytest

from core.lane_departure import command as cmd
from core.lane_departure.thread import LaneDepartureThread
from core.settings import Settings
from core.thread_management.registry import registry
from tests.lane_departure.test_detector import DT, DRIFT_START, SPEED, frame_at, make_route


class World:
    def __init__(self, monkeypatch, route=None):
        self.route = route or make_route()
        self.radar = SimpleNamespace(
            is_alive=lambda: True,
            data=SimpleNamespace(
                _lock=threading.Lock(), ego_x=0.0, ego_z=0.0, ego_yaw_rad=0.0, ego_speed=SPEED,
                paused=False, t_mono=0.0, ego_geometry=None,
            ),
        )
        self.tel = SimpleNamespace(
            is_alive=lambda: True,
            data=SimpleNamespace(
                _lock=threading.Lock(), userSteer=0.0, gear=1, userBrake=0.0,
                blinkerLeft=False, blinkerRight=False, hazardsActive=False,
            ),
        )
        self.aeb = SimpleNamespace(is_alive=lambda: True, data=SimpleNamespace(_lock=threading.Lock(), AEB_cue=False))
        self.threads = {"radar_thread": self.radar, "telemetry_thread": self.tel, "aeb_thread": self.aeb}
        self.base = time.monotonic()
        monkeypatch.setattr(registry, "get_thread", self._get)
        monkeypatch.setattr("core.map_thread.thread.map_snapshot", lambda *a, **k: SimpleNamespace(route=self.route, tmp=False))
        self.buf = bytearray(cmd.SIZE)
        struct.pack_into("=I", self.buf, 0, cmd.VERSION)

    def _get(self, name):
        if name not in self.threads:
            raise KeyError(name)
        return self.threads[name]

    def thread(self):
        thread = LaneDepartureThread()
        thread.running = True
        thread._channel._buf = self.buf
        return thread

    def step(self, thread, i, lateral=0.0, psi=0.0):
        t = i * DT
        f = frame_at(t, self.route, None, lateral, psi)
        d = self.radar.data
        d.ego_x, d.ego_z, d.ego_yaw_rad, d.t_mono = f.x, f.z, f.yaw_rad, self.base + t
        thread.loop()

    def command(self):
        return struct.unpack_from("=BBH", self.buf, 8)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(Settings.instance(), "ldw_enabled", True, raising=False)


def drift_run(world, thread, steps=200, rate=0.5):
    seen = []
    psi = rate / SPEED
    for i in range(steps):
        t = i * DT
        lateral = 0.0 if t < DRIFT_START else rate * (t - DRIFT_START)
        world.step(thread, i, lateral, psi if t >= DRIFT_START else 0.0)
        seen.append(world.command())
    return seen


def test_a_drift_sends_a_rumble_request(monkeypatch, enabled):
    world = World(monkeypatch)
    thread = world.thread()
    seen = drift_run(world, thread)
    assert (cmd.CMD_RUMBLE, 2, cmd.PEAK_UNITS) in seen
    assert thread.data.plugin.present
    assert struct.unpack_from("=I", world.buf, 4)[0] >= len(seen)


def test_the_cue_is_softer_than_the_warning(monkeypatch, enabled):
    world = World(monkeypatch)
    seen = drift_run(world, world.thread())
    first = next(c for c in seen if c[0] == cmd.CMD_RUMBLE)
    assert first == (cmd.CMD_RUMBLE, 2, cmd.intensity_units(0.6))
    assert seen.index((cmd.CMD_RUMBLE, 2, cmd.PEAK_UNITS)) > seen.index(first)


def test_switched_off_the_thread_writes_nothing(monkeypatch):
    monkeypatch.setattr(Settings.instance(), "ldw_enabled", False, raising=False)
    world = World(monkeypatch)
    thread = world.thread()
    drift_run(world, thread, steps=120)
    assert bytes(world.buf[4:]) == bytes(cmd.SIZE - 4)


def test_switching_off_mid_rumble_asks_for_silence(monkeypatch):
    settings = Settings.instance()
    monkeypatch.setattr(settings, "ldw_enabled", True, raising=False)
    world = World(monkeypatch)
    thread = world.thread()
    seen = drift_run(world, thread, steps=110)
    assert world.command()[0] == cmd.CMD_RUMBLE, seen[-3:]
    monkeypatch.setattr(settings, "ldw_enabled", False, raising=False)
    thread.loop()
    assert world.command() == (cmd.CMD_NONE, 0, 0)


def test_a_blinker_keeps_the_wheel_quiet(monkeypatch, enabled):
    world = World(monkeypatch)
    world.tel.data.blinkerRight = True
    seen = drift_run(world, world.thread())
    assert all(c[0] == cmd.CMD_NONE for c in seen)


def test_test_mode_rumbles_even_with_a_blinker(monkeypatch, enabled):
    monkeypatch.setattr(Settings.instance(), "ldw_test_mode", True, raising=False)
    world = World(monkeypatch)
    world.tel.data.blinkerRight = True
    assert (cmd.CMD_RUMBLE, 2, cmd.PEAK_UNITS) in drift_run(world, world.thread())


def test_the_gate_summary_says_where_the_time_went(monkeypatch, enabled, caplog):
    monkeypatch.setattr("core.lane_departure.thread._SUMMARY_S", 0.0)
    world = World(monkeypatch)
    with caplog.at_level("INFO", logger="core.lane_departure.thread"):
        drift_run(world, world.thread(), steps=60)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("lane departure last")]
    assert lines and "closest edge" in lines[-1]


def test_aeb_cue_keeps_the_wheel_quiet(monkeypatch, enabled):
    world = World(monkeypatch)
    world.aeb.data.AEB_cue = True
    assert all(c[0] == cmd.CMD_NONE for c in drift_run(world, world.thread()))


@pytest.mark.parametrize("missing", ["radar_thread", "telemetry_thread"])
def test_a_missing_sibling_is_silent_not_fatal(monkeypatch, enabled, missing):
    world = World(monkeypatch)
    del world.threads[missing]
    assert all(c[0] == cmd.CMD_NONE for c in drift_run(world, world.thread(), steps=120))


def test_a_missing_aeb_thread_does_not_matter(monkeypatch, enabled):
    world = World(monkeypatch)
    del world.threads["aeb_thread"]
    assert (cmd.CMD_RUMBLE, 2, cmd.PEAK_UNITS) in drift_run(world, world.thread())


def test_a_stalled_radar_stops_the_rumble(monkeypatch, enabled):
    world = World(monkeypatch)
    thread = world.thread()
    drift_run(world, thread, steps=110)
    assert world.command()[0] == cmd.CMD_RUMBLE
    world.radar.data.t_mono = time.monotonic() - 5.0
    thread.loop()
    assert world.command()[0] == cmd.CMD_NONE


def test_teardown_asks_for_silence(monkeypatch, enabled):
    world = World(monkeypatch)
    thread = world.thread()
    drift_run(world, thread, steps=110)
    thread.teardown()
    assert world.command() == (cmd.CMD_NONE, 0, 0)
    assert not thread.data.active


def test_the_panel_sees_the_side_acted_on(monkeypatch, enabled):
    world = World(monkeypatch)
    thread = world.thread()
    drift_run(world, thread, steps=110)
    assert thread.data.active
    assert thread.data.firing_side == thread.data.side != 0
    assert thread.data.watch_reason in ("warn", "approach")


def test_switched_off_the_panel_sees_no_lane(monkeypatch):
    world = World(monkeypatch)
    thread = world.thread()
    drift_run(world, thread, steps=110)
    assert thread.data.watch_reason == "off"
    assert thread.data.firing_side == 0


def test_lka_alone_shows_the_lane_and_the_side_its_pulse_steers_from(monkeypatch):
    monkeypatch.setattr(Settings.instance(), "lka_enabled", True, raising=False)
    world = World(monkeypatch)
    thread = world.thread()
    drift_run(world, thread, steps=110)
    assert thread.data.watch_reason != "off"
    assert not thread.data.active
    assert thread._pulse.side != 0
    assert thread.data.firing_side == thread._pulse.side
