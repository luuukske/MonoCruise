"""Road wetness from the traction plugin and what AEB plans with on a wet road."""

import struct

import pytest

from core.sending_thread.aeb_capacity import FullPedalRun
from core.traction.reader import (
    STALE_AFTER_S,
    WET_GRIP_LOSS,
    TractionReader,
    decode_wetness,
    is_wet,
    wet_grip_factor,
)

_SIZE = 2172


def _block(seq: int, wetness: float, status: int = 0x5F, version: int = 1) -> bytearray:
    buf = bytearray(_SIZE)
    struct.pack_into("=IIIIIIf", buf, 0, version, seq, status, 100, 1, 1, wetness)
    return buf


@pytest.mark.parametrize("wetness", [None, 0.0, -0.3, float("nan")])
def test_unknown_or_dry_road_leaves_the_capacity_exactly_as_before(wetness):
    assert wet_grip_factor(wetness) == 1.0
    assert not is_wet(wetness)


def test_wet_grip_follows_the_fit_and_caps_at_full_wetness():
    assert wet_grip_factor(0.42) == pytest.approx(1.0 - WET_GRIP_LOSS * 0.42)
    assert wet_grip_factor(0.42) == pytest.approx(0.8236, abs=1e-4)
    assert wet_grip_factor(1.6) == pytest.approx(1.0 - WET_GRIP_LOSS)
    assert is_wet(0.2)


def test_decode_needs_the_layout_and_live_weather():
    assert decode_wetness(bytes(_block(4, 0.5))) == (4, 0.5)
    assert decode_wetness(bytes(_block(4, 0.5, status=0x5D)))[1] is None
    assert decode_wetness(bytes(_block(4, 0.5, version=0)))[1] is None


def test_reader_reports_live_wetness_and_forgets_it_when_the_plugin_stops():
    buf = _block(10, 0.42)
    reader = TractionReader(buffer=buf)
    assert reader.wetness(100.0) == pytest.approx(0.42)
    struct.pack_into("=I", buf, 4, 12)
    assert reader.wetness(101.0) == pytest.approx(0.42)
    assert reader.wetness(101.0 + STALE_AFTER_S + 0.1) is None


def test_a_torn_read_keeps_the_last_value():
    buf = _block(10, 0.3)
    reader = TractionReader(buffer=buf)
    assert reader.wetness(0.0) == pytest.approx(0.3)
    struct.pack_into("=I", buf, 4, 11)
    assert reader.wetness(0.1) == pytest.approx(0.3)


def test_no_writer_reads_as_unknown():
    assert TractionReader(buffer=bytearray(_SIZE)).wetness(0.0) is None


def _slam(run: FullPedalRun) -> None:
    for i in range(80):
        run.tick(True, "k", 20000.0, 2.0, 20.0 - 0.1 * i, 0.0, i * 0.05)


def test_an_abandoned_full_pedal_stop_is_never_measured():
    measured = FullPedalRun()
    _slam(measured)
    assert measured.tick(False, "k", 20000.0, 2.0, 5.0, 0.0, 5.0) is not None
    run = FullPedalRun()
    _slam(run)
    run.abandon()
    assert run.tick(False, "k", 20000.0, 2.0, 5.0, 0.0, 5.0) is None
