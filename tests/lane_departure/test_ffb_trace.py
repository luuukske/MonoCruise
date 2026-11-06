"""The plugin's trace block: Python reader against the plugin's header, and the debug CSV it feeds."""

from __future__ import annotations

import csv
import logging
import re
import struct
from pathlib import Path

from core.lane_departure import ffb_trace as tr

HEADER = Path(__file__).resolve().parents[2] / "plugins" / "ffb" / "ffb_trace.h"


def _define(text: str, name: str) -> str:
    return re.search(rf"#define\s+{name}\s+(.+)", text).group(1).strip()


def block(head: int, written: range | None = None, version: int = tr.VERSION, effects: int = 1) -> bytearray:
    """A block whose sample ``i`` has ``t = i`` and ``steer = i / 1000``, for every ``i`` in ``written``."""
    buf = bytearray(tr.SIZE)
    struct.pack_into("=IIIIIII", buf, 0, version, tr.LEN, head, effects, 1, 0, 10000)
    for i in written if written is not None else range(max(0, head - tr.LEN), head):
        struct.pack_into("=dfhhhHHh", buf, 192 + (i % tr.LEN) * 24, float(i), i / 1000, 100 + i % 50, 200, 50, i,
                         tr.FLAG_DRIVING | tr.FLAG_SERVO, 0)
    return buf


def test_python_layout_matches_the_plugin_header():
    text = HEADER.read_text(encoding="utf-8")
    assert _define(text, "FFB_TRACE_TAG") == 'L"' + tr.TAG.replace("\\", "\\\\") + '"'
    assert int(_define(text, "FFB_TRACE_VERSION")) == tr.VERSION
    assert int(_define(text, "FFB_TRACE_LEN")) == tr.LEN
    assert int(_define(text, "FFB_TRACE_SIZE")) == tr.SIZE
    offsets = {k: int(v) for k, v in re.findall(r"offsetof\(FfbTrace, (\w+)\) == (\d+)", text)}
    assert offsets == {"head": 8, "effects": 12, "creates": 16, "other_sets": 20, "const_gain": 24, "cond": 64,
                       "samples": 192}
    sample = {k: int(v) for k, v in re.findall(r"offsetof\(FfbTraceSample, (\w+)\) == (\d+)", text)}
    assert sample == {"steer": 8, "game": 12, "out": 14, "servo": 16, "steer_seq": 18, "flags": 20}
    assert (tr._OFF_HEAD, tr._OFF_COND, tr._OFF_SAMPLES) == (8, 64, 192)
    assert (tr._SAMPLE.size, tr._COND.size, tr._HEAD.size) == (24, 32, 28)
    for name, value in (("SERVO", tr.FLAG_SERVO), ("RUMBLE", tr.FLAG_RUMBLE), ("LIVE", tr.FLAG_LIVE),
                        ("DRIVING", tr.FLAG_DRIVING)):
        assert int(_define(text, "FFB_TF_" + name)) == value
    assert int(_define(text, "FFB_FX_SPRING")) == tr.KINDS.index("spring")
    assert int(_define(text, "FFB_FX_FRICTION")) == tr.KINDS.index("friction")
    assert int(_define(text, "FFB_FX_KINDS")) == len(tr.KINDS)
    assert int(_define(text, "FFB_FX_OTHER")) == tr.KIND_OTHER


def test_each_sample_is_read_once_in_order():
    read, since = tr.decode(bytes(block(10)), 10, None)
    assert read.present and [s.t for s in read.samples] == list(range(10)) and since == 10
    assert read.samples[3].steer == struct.unpack("=f", struct.pack("=f", 0.003))[0]
    read, since = tr.decode(bytes(block(14)), 14, since)
    assert [s.t for s in read.samples] == [10, 11, 12, 13] and read.lost == 0 and since == 14
    read, since = tr.decode(bytes(block(14)), 14, since)
    assert read.samples == () and since == 14


def test_slots_rewritten_during_the_copy_are_dropped_and_counted_lost():
    read, since = tr.decode(bytes(block(300)), 300 + 5, 0)
    assert read.samples[0].t == 300 + 5 - tr.LEN + 2, "the oldest slots may hold newer samples"
    assert read.lost == read.samples[0].t and read.samples[-1].t == 299 and since == 300


def test_a_restarted_plugin_or_a_missing_block_starts_over():
    _, since = tr.decode(bytes(block(500)), 500, None)
    read, since = tr.decode(bytes(block(3)), 3, since)
    assert [s.t for s in read.samples] == [0, 1, 2] and read.lost == 0 and since == 3
    assert tr.decode(bytes(block(3, version=0)), 3, since) == (tr.TraceRead(), None)
    assert tr.decode(b"short", 0, since) == (tr.TraceRead(), None)


def test_the_game_effects_are_named():
    read, _ = tr.decode(bytes(block(0, effects=1 | 1 << 7 | 1 << 8 | 1 << 31)), 0, None)
    assert read.kinds() == ["constant", "spring", "damper", "other"] and read.const_gain == 10000


class FakeReader:
    def __init__(self, reads):
        self.reads = list(reads)

    def read(self, now=None):
        return self.reads.pop(0) if self.reads else tr.TraceRead()

    def close(self):
        pass


def sample(t: float, flags: int = tr.FLAG_DRIVING) -> tr.TraceSample:
    return tr.TraceSample(t, 0.1, -500, -300, 200, 7, flags)


def test_the_log_writes_driving_samples_with_the_tick_context(tmp_path, caplog):
    spring = tr.Condition(pos_coeff=4000, neg_coeff=4000, pos_sat=10000, gain=8000, sets=3)
    log = tr.TraceLog(tmp_path, "la_ffb_debug.csv", "LA")
    log._reader = FakeReader([
        tr.TraceRead(True, (sample(1.0), sample(1.01, 0), sample(1.02, tr.FLAG_DRIVING | tr.FLAG_SERVO)), lost=4,
                     effects=1 | 1 << 7, const_gain=10000, conditions={"spring": spring}),
    ])
    with caplog.at_level(logging.INFO):
        log.tick(5.0, {"speed_kmh": 80.0, "reason": "steer", "kp": 1234.5, "only": 0, "force_per_ms2": None})
    log.close()
    with (tmp_path / "la_ffb_debug.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["t_s"] for r in rows] == ["1.00000", "1.02000"], "a sample from a paused game is skipped"
    assert [r["lost"] for r in rows] == ["4", "0"] and [r["servo_on"] for r in rows] == ["0", "1"]
    assert rows[0]["speed_kmh"] == "80.00000" and rows[0]["reason"] == "steer" and rows[0]["force_per_ms2"] == ""
    assert rows[0]["spring_coeff"] == "4000" and rows[0]["spring_gain"] == "8000" and rows[0]["const_gain"] == "10000"
    assert "the game made constant, spring effects" in caplog.text and "spring coefficient 4000/4000" in caplog.text


def test_the_log_stays_quiet_without_the_plugin(tmp_path):
    log = tr.TraceLog(tmp_path, "la_ffb_debug.csv", "LA")
    log._reader = FakeReader([tr.TraceRead()])
    log.tick(1.0, {})
    assert not (tmp_path / "la_ffb_debug.csv").exists()
