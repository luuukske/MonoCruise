"""The shared-memory command block: Python writer against the plugin's header, and its safety rules."""

from __future__ import annotations

import re
import struct
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
        "heartbeat": 4, "command": 8, "seq": 12, "torque": 16, "status": 20, "frames": 24,
    }
    assert (cmd._OFF_HEARTBEAT, cmd._OFF_CMD, cmd._OFF_SEQ, cmd._OFF_STATUS, cmd._OFF_FRAMES) == (4, 8, 12, 20, 24)
    for name, value in (("HOOKED", cmd.STATUS_HOOKED), ("FORCE_SEEN", cmd.STATUS_FORCE_SEEN),
                        ("RUMBLING", cmd.STATUS_RUMBLING), ("DEAD", cmd.STATUS_DEAD)):
        assert int(_define(text, "FFB_ST_" + name)) == value
    assert int(_define(text, "FFB_CMD_RUMBLE")) == cmd.CMD_RUMBLE


def test_the_peak_cap_matches_the_plugin():
    text = SIGNAL.read_text(encoding="utf-8")
    assert float(_define(text, "FFB_PEAK_CAP")) == cmd.PEAK_CAP_UNITS
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
