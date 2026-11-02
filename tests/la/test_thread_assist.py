"""LA through the lane departure thread: settings, plugin versions and what reaches the command block."""

from __future__ import annotations

import struct
from types import SimpleNamespace

import pytest

from core.lane_departure import command as cmd
from core.settings import Settings
from tests.lane_departure.test_detector import DRIFT_START, DT, SPEED, make_route
from tests.lane_departure.test_thread import World

SPRING = 34570.0
GAIN = 0.19
FORCE = SPRING / (GAIN * SPEED * SPEED)


def hold_in(world) -> int:
    return struct.unpack_from("=h", world.buf, 16)[0]


def servo_in(world) -> tuple:
    if len(world.buf) < cmd.SIZE:
        return (0.0, 0.0, 0.0, 0, 0)
    return struct.unpack_from("=fffHb", world.buf, 32)


def wheel(world, version=cmd.VERSION) -> World:
    """A plugin that is hooked and, from version 3, reads the wheel."""
    struct.pack_into("=I", world.buf, 0, version)
    bits = cmd.STATUS_HOOKED | cmd.STATUS_FORCE_SEEN | (cmd.STATUS_STEER if version >= cmd.VERSION else 0)
    struct.pack_into("=I", world.buf, 20, bits)
    return world


def tick(world, thread, i, lateral):
    """One loop; the plugin's steering channel counts on, the wheel stays centred."""
    if len(world.buf) >= cmd.SIZE:
        struct.pack_into("=I", world.buf, 56, i + 1)
    world.step(thread, i, lateral=lateral)


@pytest.fixture
def assist(monkeypatch):
    settings = Settings.instance()
    monkeypatch.setattr(settings, "ldw_enabled", False, raising=False)
    monkeypatch.setattr(settings, "la_enabled", True, raising=False)
    monkeypatch.setattr(settings, "la_force_per_ms2", FORCE, raising=False)
    monkeypatch.setattr(settings, "la_kappa_per_steer", GAIN, raising=False)
    return settings


def run(world, thread, steps=100, lateral=0.6):
    holds, servos = [], []
    for i in range(steps):
        tick(world, thread, i, 0.0 if i * DT < 1.0 else lateral)
        holds.append(hold_in(world))
        servos.append(servo_in(world))
    return holds, servos


def test_an_offset_truck_gets_a_hold_and_a_servo_request_and_no_rumble(monkeypatch, assist):
    world = wheel(World(monkeypatch))
    thread = world.thread()
    holds, servos = run(world, thread)
    target, kp, ki, cap, direction = servos[-1]
    assert direction == 1 and kp > 0.0 and ki > 0.0 and cap > 0
    assert target > 0.0, "right of centre: the wheel is asked to turn counterclockwise"
    assert holds[-1] == 0, "no hold torque: the servo's integral holds the bend"
    assert struct.unpack_from("=f", world.buf, 72)[0] == 0.0, "the game's own force is kept by default"
    assert world.command()[0] == cmd.CMD_NONE
    assert thread.data.assist_reason == "assist"


def test_a_torque_only_plugin_gets_nothing_and_says_so(monkeypatch, assist, caplog):
    world = World(monkeypatch)
    world.buf = bytearray(cmd.LEGACY_SIZE)
    wheel(world, version=cmd.TORQUE_VERSION)
    thread = world.thread()
    thread._channel._size = cmd.LEGACY_SIZE
    with caplog.at_level("INFO", logger="core.la.runner"):
        holds, _ = run(world, thread)
    assert set(holds) == {0}
    assert thread.data.assist_reason == "plugin"
    assert any("version 4 FFB plugin" in r.getMessage() for r in caplog.records)


def test_truckersmp_keeps_it_off_unless_allowed(monkeypatch, assist):
    world = wheel(World(monkeypatch))
    monkeypatch.setattr("core.map_thread.thread.map_snapshot", lambda *a, **k: SimpleNamespace(route=world.route, tmp=True))
    thread = world.thread()
    holds, servos = run(world, thread)
    assert set(holds) == {0} and servos[-1][4] == 0
    assert thread.data.assist_reason == "tmp"
    monkeypatch.setattr(assist, "la_tmp_enabled", True, raising=False)
    allowed = wheel(World(monkeypatch))
    monkeypatch.setattr("core.map_thread.thread.map_snapshot", lambda *a, **k: SimpleNamespace(route=allowed.route, tmp=True))
    assert run(allowed, allowed.thread())[1][-1][4] == 1


def test_without_a_calibration_nothing_is_sent(monkeypatch, assist):
    monkeypatch.setattr(assist, "la_force_per_ms2", 0.0, raising=False)
    monkeypatch.setattr(assist, "la_kappa_per_steer", 0.0, raising=False)
    world = wheel(World(monkeypatch))
    thread = world.thread()
    holds, servos = run(world, thread)
    assert set(holds) == {0} and servos[-1] == (0.0, 0.0, 0.0, 0, 0)
    assert thread.data.assist_reason == "calibrating"


def test_switching_it_off_mid_assist_releases_the_wheel(monkeypatch, assist):
    world = wheel(World(monkeypatch))
    thread = world.thread()
    run(world, thread)
    assert servo_in(world)[4] == 1
    monkeypatch.setattr(assist, "la_enabled", False, raising=False)
    thread.loop()
    assert hold_in(world) == 0 and servo_in(world) == (0.0, 0.0, 0.0, 0, 0)
    assert struct.unpack_from("=f", world.buf, 72)[0] == 0.0
    assert thread.data.assist_torque == 0.0


def test_the_rumble_and_the_assist_run_together(monkeypatch, assist):
    monkeypatch.setattr(assist, "ldw_enabled", True, raising=False)
    world = wheel(World(monkeypatch))
    thread = world.thread()
    commands, servos = [], []
    for i in range(200):
        t = i * DT
        if len(world.buf) >= cmd.SIZE:
            struct.pack_into("=I", world.buf, 56, i + 1)
        world.step(thread, i, 0.0 if t < DRIFT_START else 0.5 * (t - DRIFT_START), 0.5 / SPEED if t >= DRIFT_START else 0.0)
        commands.append(world.command())
        servos.append(servo_in(world))
    assert (cmd.CMD_RUMBLE, 2, cmd.PEAK_UNITS) in commands
    assert any(s[4] == 1 for s in servos)


def test_hands_off_brings_the_warning_rumble_then_lets_go(monkeypatch, assist):
    world = wheel(World(monkeypatch))
    thread = world.thread()
    rumbled = []
    for i in range(int(25.0 / DT)):
        tick(world, thread, i, 0.6)
        rumbled.append(world.command()[0] == cmd.CMD_RUMBLE)
    assert any(rumbled), "a wheel that follows without a tremor reads as hands off"
    assert thread.data.assist_reason == "hands_off" and servo_in(world)[4] == 0


def test_teardown_releases_the_wheel(monkeypatch, assist):
    world = wheel(World(monkeypatch))
    thread = world.thread()
    run(world, thread)
    thread.teardown()
    assert hold_in(world) == 0 and servo_in(world)[4] == 0


def test_nothing_is_written_until_it_is_enabled(monkeypatch, assist):
    monkeypatch.setattr(assist, "la_enabled", False, raising=False)
    world = World(monkeypatch)
    thread = world.thread()
    for i in range(60):
        world.step(thread, i)
    assert bytes(world.buf[4:]) == bytes(cmd.SIZE - 4)


def test_the_follow_everywhere_test_switch_keeps_it_on_at_a_fork(monkeypatch, assist):
    fork = make_route(length=2500.0, branch_s=120.0)
    held = wheel(World(monkeypatch, route=fork))
    thread = held.thread()
    run(held, thread)
    assert thread.data.assist_reason == "branch"
    monkeypatch.setattr(assist, "la_follow_all", True, raising=False)
    kept = wheel(World(monkeypatch, route=fork))
    thread = kept.thread()
    run(kept, thread)
    assert thread.data.assist_reason == "assist" and servo_in(kept)[4] == 1
