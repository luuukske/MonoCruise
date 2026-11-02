"""The shared-memory command block: Python writer against the plugin's header, and its safety rules."""

from __future__ import annotations

import re
import struct

import pytest
from pathlib import Path

from core.lane_departure import command as cmd

HEADER = Path(__file__).resolve().parents[2] / "plugins" / "ffb" / "ffb_cmd.h"
SIGNAL = Path(__file__).resolve().parents[2] / "plugins" / "ffb" / "ffb_signal.h"


def _define(text: str, name: str) -> str:
    return re.search(rf"#define\s+{name}\s+(.+)", text).group(1).strip()


def stamped(version: int = cmd.VERSION) -> bytearray:
    buf = bytearray(cmd.SIZE)
    struct.pack_into("=I", buf, 0, version)
    return buf


def channel(buf: bytearray) -> cmd.RumbleChannel:
    ch = cmd.RumbleChannel()
    ch._buf = buf
    return ch


def test_python_layout_matches_the_plugin_header():
    text = HEADER.read_text(encoding="utf-8")
    assert int(_define(text, "FFB_CMD_VERSION")) == cmd.VERSION
    assert int(_define(text, "FFB_CMD_SIZE")) == cmd.SIZE
    assert _define(text, "FFB_CMD_TAG") == 'L"' + cmd.TAG.replace("\\", "\\\\") + '"'
    offsets = dict(re.findall(r"offsetof\(FfbCmd, (\w+)\) == (\d+)", text))
    assert {k: int(v) for k, v in offsets.items()} == {
        "heartbeat": 4, "command": 8, "seq": 12, "torque": 16, "game_force": 18, "status": 20, "frames": 24,
        "applied": 28, "servo_applied": 30, "servo_target": 32, "servo_kp": 36, "servo_ki": 40, "servo_cap": 44,
        "servo_dir": 46, "servo_only": 47, "steer": 48, "steer_rate": 52, "steer_frames": 56, "servo_scale": 60, "servo_rate": 64,
        "servo_kd": 68, "servo_cancel": 72, "servo_stamp": 76, "servo_integ": 80, "frame_hz": 84,
        "cancel_applied": 88,
    }
    assert (cmd._OFF_HEARTBEAT, cmd._OFF_CMD, cmd._OFF_SEQ, cmd._OFF_TORQUE, cmd._OFF_GAME_FORCE, cmd._OFF_STATUS,
            cmd._OFF_FRAMES, cmd._OFF_APPLIED) == (4, 8, 12, 16, 18, 20, 24, 28)
    assert (cmd._OFF_SERVO_APPLIED, cmd._OFF_SERVO, cmd._OFF_STEER, cmd._OFF_STEER_RATE, cmd._OFF_STEER_FRAMES,
            cmd._OFF_SERVO_SCALE) == (30, 32, 48, 52, 56, 60)
    assert cmd._SERVO.size == 16, "target, kp, ki (f32), cap (u16), dir (i8), only (i8) from offset 32"
    assert (cmd._OFF_SERVO4, cmd._OFF_STAMP, cmd._OFF_INTEG, cmd._OFF_FRAME_HZ, cmd._OFF_CANCEL_APPLIED) == (
        64, 76, 80, 84, 88)
    assert cmd._SERVO4.size == 12, "rate, kd, cancel (f32) from offset 64"
    for name, value in (("HOOKED", cmd.STATUS_HOOKED), ("FORCE_SEEN", cmd.STATUS_FORCE_SEEN),
                        ("RUMBLING", cmd.STATUS_RUMBLING), ("DEAD", cmd.STATUS_DEAD), ("TORQUE", cmd.STATUS_TORQUE),
                        ("SERVO", cmd.STATUS_SERVO), ("STEER", cmd.STATUS_STEER),
                        ("ONE_SIDED", cmd.STATUS_ONE_SIDED)):
        assert int(_define(text, "FFB_ST_" + name)) == value
    assert int(_define(text, "FFB_CMD_RUMBLE")) == cmd.CMD_RUMBLE


def test_the_peak_cap_matches_the_plugin():
    text = SIGNAL.read_text(encoding="utf-8")
    assert float(_define(text, "FFB_PEAK_CAP")) == cmd.PEAK_CAP_UNITS
    assert float(_define(text, "FFB_TORQUE_CAP")) == cmd.TORQUE_CAP_UNITS
    assert float(_define(text, "FFB_SERVO_CAP")) == cmd.SERVO_CAP_UNITS
    assert cmd.intensity_units(1.0) == cmd.PEAK_UNITS
    assert cmd.intensity_units(50.0) == cmd.PEAK_CAP_UNITS
    assert cmd.intensity_units(-1.0) == 0


def test_a_rumble_request_reaches_the_block():
    buf = stamped()
    ch = channel(buf)
    status = ch.publish(True, 2, 1600, now=1.0)
    assert status.present
    command, side, units = struct.unpack_from("=BBH", buf, 8)
    assert (command, side, units) == (cmd.CMD_RUMBLE, 2, 1600)
    assert struct.unpack_from("=I", buf, 12)[0] == 1
    assert struct.unpack_from("=I", buf, 4)[0] == 1


def test_the_heartbeat_moves_on_every_call_and_never_reads_zero():
    buf = stamped()
    ch = channel(buf)
    seen = []
    for _ in range(3):
        ch.publish(False, 0, 0, now=1.0)
        seen.append(struct.unpack_from("=I", buf, 4)[0])
    assert seen == [1, 2, 3]
    ch._heartbeat = 0xFFFFFFFF
    ch.publish(False, 0, 0, now=1.0)
    assert struct.unpack_from("=I", buf, 4)[0] == 1


def test_the_event_counter_counts_rising_edges_only():
    buf = stamped()
    ch = channel(buf)
    for active in (True, True, False, True):
        ch.publish(active, 1, 1600, now=1.0)
    assert struct.unpack_from("=I", buf, 12)[0] == 2


def test_repeating_no_rumble_does_not_cancel_another_writers_request():
    buf = stamped()
    ch = channel(buf)
    ch.publish(False, 0, 0, now=1.0)
    beat = struct.unpack_from("=I", buf, 4)[0]
    struct.pack_into("=BBH", buf, 8, cmd.CMD_RUMBLE, 2, 1600)
    ch.publish(False, 0, 0, now=1.1)
    assert struct.unpack_from("=BBH", buf, 8) == (cmd.CMD_RUMBLE, 2, 1600)
    assert struct.unpack_from("=I", buf, 4)[0] == beat + 1


def test_a_plugin_that_goes_away_and_returns_gets_the_request_again():
    buf = stamped()
    ch = channel(buf)
    ch.publish(True, 2, 1600, now=1.0)
    struct.pack_into("=I", buf, 0, 0)
    assert not ch.publish(True, 2, 1600, now=1.1).present
    struct.pack_into("=I", buf, 0, cmd.VERSION)
    struct.pack_into("=BBH", buf, 8, cmd.CMD_NONE, 0, 0)
    ch.publish(True, 2, 1600, now=1.2)
    assert struct.unpack_from("=BBH", buf, 8) == (cmd.CMD_RUMBLE, 2, 1600)


def test_stop_asks_for_silence():
    buf = stamped()
    ch = channel(buf)
    ch.publish(True, 1, 1600, now=1.0)
    ch.stop()
    assert struct.unpack_from("=BBH", buf, 8) == (cmd.CMD_NONE, 0, 0)


def test_nothing_is_written_before_the_plugin_stamps_its_version():
    for version in (0, 7):
        buf = stamped(version)
        status = channel(buf).publish(True, 1, 1600, now=1.0)
        assert not status.present
        assert bytes(buf[4:]) == bytes(cmd.SIZE - 4)


def test_status_bits_decode():
    buf = stamped()
    struct.pack_into("=I", buf, 20, cmd.STATUS_HOOKED | cmd.STATUS_RUMBLING)
    struct.pack_into("=I", buf, 24, 99)
    status = cmd.decode_status(bytes(buf))
    assert (status.present, status.hooked, status.rumbling, status.dead, status.frames) == (True, True, True, False, 99)


def test_an_unavailable_block_reads_as_absent(monkeypatch):
    monkeypatch.setattr(cmd.sys, "platform", "linux")
    ch = cmd.RumbleChannel()
    assert ch.publish(True, 1, 1600, now=1.0) == cmd.PluginStatus()
    assert not ch.is_open


def test_torque_reaches_the_block_capped_and_only_when_it_changes():
    buf = stamped()
    ch = channel(buf)
    ch.publish(False, 0, 0, now=1.0, torque=-812.4)
    assert struct.unpack_from("=h", buf, 16)[0] == -812
    struct.pack_into("=h", buf, 16, 500)
    ch.publish(False, 0, 0, now=1.1, torque=-812.4)
    assert struct.unpack_from("=h", buf, 16)[0] == 500, "a repeat does not cancel another writer"
    ch.publish(False, 0, 0, now=1.2, torque=99999.0)
    assert struct.unpack_from("=h", buf, 16)[0] == cmd.TORQUE_CAP_UNITS
    ch.publish(False, 0, 0, now=1.3, torque=float("nan"))
    assert struct.unpack_from("=h", buf, 16)[0] == 0


def test_stop_also_asks_for_no_torque():
    buf = stamped()
    ch = channel(buf)
    ch.publish(False, 0, 0, now=1.0, torque=900)
    ch.stop()
    assert struct.unpack_from("=h", buf, 16)[0] == 0


def test_a_rumble_only_plugin_still_rumbles_but_gets_no_torque():
    buf = bytearray(cmd.LEGACY_SIZE)
    struct.pack_into("=I", buf, 0, cmd.RUMBLE_ONLY_VERSION)
    struct.pack_into("=I", buf, 20, cmd.STATUS_HOOKED)
    status = channel(buf).publish(True, 2, 1600, now=1.0, torque=900)
    assert status.present and status.version == 1 and not status.torque_capable
    assert struct.unpack_from("=BBH", buf, 8) == (cmd.CMD_RUMBLE, 2, 1600)
    assert struct.unpack_from("=h", buf, 16)[0] == 0


def test_the_plugin_reports_the_game_force_and_the_applied_torque():
    buf = stamped()
    struct.pack_into("=I", buf, 20, cmd.STATUS_HOOKED | cmd.STATUS_TORQUE)
    struct.pack_into("=h", buf, 18, -4321)
    struct.pack_into("=h", buf, 28, 640)
    status = cmd.decode_status(bytes(buf))
    assert (status.game_force, status.applied, status.torquing, status.torque_capable) == (-4321, 640, True, True)
    struct.pack_into("=I", buf, 20, cmd.STATUS_HOOKED | cmd.STATUS_DEAD)
    assert not cmd.decode_status(bytes(buf)).torque_capable


def test_a_torque_plugin_reads_through_the_legacy_block():
    buf = bytearray(cmd.LEGACY_SIZE)
    struct.pack_into("=I", buf, 0, cmd.TORQUE_VERSION)
    struct.pack_into("=I", buf, 20, cmd.STATUS_HOOKED)
    ch = channel(buf)
    ch._size = cmd.LEGACY_SIZE
    status = ch.publish(False, 0, 0, now=1.0, torque=700, servo=cmd.ServoCommand(0.01, 0.0, 1.0, 800, 1))
    assert status.torque_capable and not status.servo_capable
    assert struct.unpack_from("=h", buf, 16)[0] == 700
    assert len(buf) == cmd.LEGACY_SIZE, "nothing written past the old block"


def test_the_servo_request_reaches_the_block_and_off_is_all_zero():
    buf = stamped()
    ch = channel(buf)
    ch.publish(False, 0, 0, now=1.0, servo=cmd.ServoCommand(-0.02, 0.0, 69140.0, 800, -1))
    target, kp, ki, cap, direction = struct.unpack_from("=fffHb", buf, 32)
    assert (round(target, 4), kp, round(ki), cap, direction) == (-0.02, 0.0, 69140, 800, -1)
    ch.publish(False, 0, 0, now=1.1)
    assert struct.unpack_from("=fffHb", buf, 32) == (0.0, 0.0, 0.0, 0, 0)


def test_a_bad_servo_request_is_sent_as_off():
    for bad in (cmd.ServoCommand(0.1, 0.0, -1.0, 800, 1), cmd.ServoCommand(float("nan"), 0.0, 1.0, 800, 1),
                cmd.ServoCommand(0.1, 0.0, 1.0, 800, 2)):
        assert cmd.servo_fields(bad) == (0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, 0)
    assert cmd.servo_fields(cmd.ServoCommand(0.1, 1.0, 1.0, 800, 1, kd=-1.0))[4] == 0, "a negative damping is off"
    assert cmd.servo_fields(cmd.ServoCommand(5.0, 0.0, 1.0, 99999, 1, rate=0.02, kd=3.0, cancel=4.0, only=-5)) == (
        1.0, 0.0, 1.0, cmd.SERVO_CAP_UNITS, 1, 0.02, 3.0, 1.0, -1)


def test_the_plugin_reports_the_wheel():
    buf = stamped()
    struct.pack_into("=I", buf, 20, cmd.STATUS_HOOKED | cmd.STATUS_STEER | cmd.STATUS_SERVO)
    struct.pack_into("=ffIf", buf, 48, -0.031, 0.25, 4242, 0.5)
    struct.pack_into("=h", buf, 30, 321)
    status = cmd.decode_status(bytes(buf))
    assert status.servo_capable and status.servoing
    assert (round(status.steer, 3), round(status.steer_rate, 2), status.steer_frames, status.servo_scale) == (
        -0.031, 0.25, 4242, 0.5)
    assert status.servo_applied == 321


def test_a_version_4_plugin_reports_its_servo_and_steers():
    buf = stamped()
    struct.pack_into("=I", buf, 20, cmd.STATUS_HOOKED | cmd.STATUS_STEER)
    struct.pack_into("=ff", buf, 80, 312.5, 59.8)
    struct.pack_into("=h", buf, 88, -640)
    status = cmd.decode_status(bytes(buf))
    assert status.servo_capable and (status.servo_integ, round(status.frame_hz, 1), status.cancel_applied) == (
        312.5, 59.8, -640)
    old = bytearray(cmd.SERVO3_SIZE)
    struct.pack_into("=I", old, 0, cmd.SERVO3_VERSION)
    struct.pack_into("=I", old, 20, cmd.STATUS_HOOKED | cmd.STATUS_STEER)
    three = cmd.decode_status(bytes(old))
    assert three.present and three.torque_capable and not three.servo_capable, "version 3 rumbles, never steers"


def test_each_servo_request_bumps_the_stamp_the_plugin_moves_the_target_on_from():
    buf = stamped()
    ch = channel(buf)
    ch.publish(False, 0, 0, servo=cmd.ServoCommand(0.01, 1000.0, 10.0, 800, 1, rate=0.02, kd=50.0, cancel=1.0))
    first = struct.unpack_from("=I", buf, 76)[0]
    assert struct.unpack_from("=fff", buf, 64) == pytest.approx((0.02, 50.0, 1.0))
    ch.publish(False, 0, 0, servo=cmd.ServoCommand(0.011, 1000.0, 10.0, 800, 1, rate=0.02, kd=50.0, cancel=1.0))
    ch.publish(False, 0, 0, servo=cmd.ServoCommand(0.011, 1000.0, 10.0, 800, 1, rate=0.02, kd=50.0, cancel=1.0))
    assert struct.unpack_from("=I", buf, 76)[0] == first + 1, "a new target bumps it, a repeat does not"
