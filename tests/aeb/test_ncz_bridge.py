"""NCZ state validation and AEB output suppression."""
import struct
from types import SimpleNamespace

import pytest

from core.aeb.ncz_bridge import MAGIC, NCZReader, decode_state
from core.settings import Settings


def packet(zone=1, connected=1, tick=1000, sequence=2, magic=MAGIC):
    return struct.pack("<IIIIQ8x", sequence, magic, zone, connected, tick)


@pytest.mark.parametrize("zone,expected", [(0, False), (1, True), (2, None)])
def test_zone_states(zone, expected):
    assert decode_state(packet(zone=zone), 1200) is expected


@pytest.mark.parametrize("data,now", [
    (packet(sequence=3), 1000), (packet(magic=0), 1000),
    (packet(connected=0), 1000), (packet(), 1501),
    (packet(), 999), (b"", 1000),
])
def test_untrusted_state_does_not_suppress_aeb(data, now):
    assert decode_state(data, now) is None


def test_missing_bridge_does_not_disable_aeb():
    reader = NCZReader()
    reader._api = None
    assert reader.read() is None
    reader.close()


def test_ncz_suppresses_braking_and_reenables_after_exit(monkeypatch):
    from core.aeb import clip_eval
    from tests.aeb.test_engage_sensitivity import _stopped_ahead_clip

    Settings.save({"aeb_skip_tmp_ncz": True})
    original_snapshot = clip_eval._snapshot_tuple
    original_headless = clip_eval._make_headless
    calls = [0]

    def snapshot(*args, **kwargs):
        result = list(original_snapshot(*args, **kwargs))
        result[10] = True
        return tuple(result)

    def state():
        calls[0] += 1
        return calls[0] <= 30

    def headless(cal):
        thread = original_headless(cal)
        thread._ncz_reader = SimpleNamespace(read=state)
        return thread

    monkeypatch.setattr(clip_eval, "_snapshot_tuple", snapshot)
    monkeypatch.setattr(clip_eval, "_make_headless", headless)
    clip = _stopped_ahead_clip(22, 30, capacity=10, n=45)
    result = clip_eval.run_headless(clip)
    assert all(not tick.aeb_brake and tick.target_decel_ms2 == 0 for tick in result[:30])
    assert any(tick.aeb_brake for tick in result[30:])
