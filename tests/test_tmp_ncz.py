"""TruckersMP no-collision zone gate: plugin state, radar gate, AEB skip. See core/radar/README.md §18."""

from __future__ import annotations

import struct
from dataclasses import replace
from types import SimpleNamespace

import pytest

from core.aeb.clip_eval import run_headless
from core.aeb.clip_schema import (
    AEBTickRecord, Clip, ClipMetadata, ConsumedContext, EgoTelemetry, LiveAEB,
    RadarFrameRecord,
)
from core.aeb.thread import AEBThread
from core.radar import tmp_state as TS
from core.radar.elevation import BODY_DATUM_FRAC
from core.radar.reader import _BUF_SIZE, _TOTAL_FORMAT
from core.radar.thread import RadarData
from core.radar.tmp_state import (
    ENTER_CONFIRM_S, STALE_AFTER_S, NoCollisionZoneGate, TmpState, TmpStateReader,
    decode_state, ncz_vehicle_ids,
)
from tests.aeb.harness import make_vehicle


def _state_bytes(version=1, heartbeat=1, connected=1, in_zone=1, streamed=5, collidable=0) -> bytes:
    return struct.pack("=IIBBHHH", version, heartbeat, connected, in_zone, streamed, collidable, 0)


_GHOSTS = TmpState(fresh=True, connected=True, in_no_collision_zone=True,
                   players_streamed=5, players_collidable=0)


def test_decode_reads_the_plugin_layout():
    st = decode_state(_state_bytes(streamed=7, collidable=2), heartbeat_fresh=True)
    assert st == TmpState(fresh=True, connected=True, in_no_collision_zone=True,
                          players_streamed=7, players_collidable=2)


def test_unknown_layout_version_reads_as_no_data():
    assert decode_state(_state_bytes(version=2), heartbeat_fresh=True) == TmpState()
    assert decode_state(_state_bytes(version=0), heartbeat_fresh=True) == TmpState()


@pytest.mark.parametrize("field, value", [
    ("fresh", False),
    ("connected", False),
    ("in_no_collision_zone", False),
    ("players_streamed", 0),
    ("players_collidable", 1),
])
def test_any_disagreeing_signal_means_somebody_can_collide(field, value):
    assert _GHOSTS.nobody_can_collide
    assert not replace(_GHOSTS, **{field: value}).nobody_can_collide


def test_reader_goes_stale_when_the_heartbeat_stops():
    r = TmpStateReader()
    r._buf = bytearray(_state_bytes(heartbeat=10))
    assert r.read(100.0).fresh
    assert r.read(100.0 + STALE_AFTER_S).fresh
    assert not r.read(100.0 + STALE_AFTER_S + 0.01).fresh
    r._buf[:] = _state_bytes(heartbeat=11)
    assert r.read(101.0).fresh


def test_reader_never_trusts_a_zero_heartbeat():
    r = TmpStateReader()
    r._buf = bytearray(_state_bytes(heartbeat=0))
    assert not r.read(5.0).fresh
    assert not r.read(5.0).nobody_can_collide


def test_reader_without_shared_memory_reads_as_no_data():
    r = TmpStateReader()
    r._open_failed = True
    assert r.read(1.0) == TmpState()


def test_gate_needs_agreement_to_hold_and_releases_at_once():
    gate = NoCollisionZoneGate()
    assert not gate.step(_GHOSTS, 10.0)
    assert not gate.step(_GHOSTS, 10.0 + ENTER_CONFIRM_S - 0.01)
    assert gate.step(_GHOSTS, 10.0 + ENTER_CONFIRM_S)
    collidable = replace(_GHOSTS, players_collidable=1)
    assert not gate.step(collidable, 10.5)
    # Re-entry starts the confirmation over.
    assert not gate.step(_GHOSTS, 10.6)
    assert gate.step(_GHOSTS, 10.6 + ENTER_CONFIRM_S)


def test_gate_releases_when_the_writer_goes_stale():
    gate = NoCollisionZoneGate()
    gate.step(_GHOSTS, 0.0)
    assert gate.step(_GHOSTS, 1.0)
    assert not gate.step(replace(_GHOSTS, fresh=False), 1.1)


def test_only_truckersmp_vehicles_become_ghosts():
    tractor = make_vehicle(1, 0.0, -30.0, 0.0, 0.0, is_tmp=True)
    trailer_record = make_vehicle(2, 0.0, -42.0, 0.0, 0.0, is_tmp=True, is_trailer=True)
    ai = make_vehicle(3, 3.5, -30.0, 0.0, 10.0)
    nested = make_vehicle(1_000_008, 0.0, -55.0, 0.0, 0.0, is_tmp=True, is_trailer=True)
    assert ncz_vehicle_ids(True, [tractor, trailer_record, ai], [nested]) == {1, 2, 1_000_008}
    assert ncz_vehicle_ids(False, [tractor, trailer_record, ai], [nested]) == frozenset()


def test_aeb_snapshot_folds_ghosts_into_the_ids_it_skips(monkeypatch):
    data = RadarData(off_surface_ids=frozenset({4}), ncz_ids=frozenset({7, 8}))
    stub = SimpleNamespace(data=data, is_alive=lambda: True)
    monkeypatch.setattr("core.aeb.thread.registry.get_thread", lambda name: stub)
    snap = AEBThread._read_radar_snapshot(None)
    assert snap[13] == {4, 7, 8}


def test_clip_frames_carry_the_flag_and_old_clips_read_false():
    rec = RadarFrameRecord(t_wall=1.0, t_mono=1.0, tmp_ncz=True)
    assert RadarFrameRecord.from_json(rec.to_json()).tmp_ncz
    plain = RadarFrameRecord(t_wall=1.0, t_mono=1.0).to_json()
    assert "tmp_ncz" not in plain
    assert not RadarFrameRecord.from_json(plain).tmp_ncz


_BODY_H = 3.0


def _stopped_tmp_truck_buf(pz: float, vid: int) -> bytes:
    flat: list = []
    flat += [0.0, BODY_DATUM_FRAC * _BODY_H, pz, 1.0, 0.0, 0.0, 0.0, 2.5, _BODY_H, 6.0, 0.0, 0.0]
    flat += [0, vid, 1, 0] + [0.0] * 30
    for _ in range(39):
        flat += [0.0] * 12 + [0, 0, 0, 0] + [0.0] * 30
    buf = struct.pack(_TOTAL_FORMAT, *flat)
    assert len(buf) == _BUF_SIZE
    return buf


def _tmp_collision_clip(ncz: bool, n: int = 90, hz: float = 30.0) -> Clip:
    """Ego at 20 m/s closing on a stopped TruckersMP truck 45 m directly ahead."""
    dt = 1.0 / hz
    meta = ClipMetadata.create(trigger_source="auto_engagement", session_kind="TMP")
    frames, ticks = [], []
    for i in range(n):
        t = i * dt
        frames.append(RadarFrameRecord(
            t_wall=1000.0 + t, t_mono=t,
            ego=EgoTelemetry(coordinateX=0.0, coordinateZ=20.0 * t, rotationX=0.5,
                             rotationY=0.0, speed=20.0),
            traffic_buf=_stopped_tmp_truck_buf(45.0, vid=3),
            tmp_ncz=ncz,
        ))
        ticks.append(AEBTickRecord(
            t_mono=t, radar_t_mono=t,
            consumed=ConsumedContext(max_brake_ms2=10.0, aeb_enabled=True),
            live_aeb=LiveAEB(),
        ))
    return Clip(metadata=meta, radar_frames=frames, aeb_ticks=ticks)


def test_aeb_still_brakes_for_a_truckersmp_truck_outside_a_zone():
    ev = run_headless(_tmp_collision_clip(ncz=False))
    assert any(e.aeb_brake for e in ev)


def test_aeb_ignores_a_ghost_in_a_no_collision_zone():
    ev = run_headless(_tmp_collision_clip(ncz=True))
    assert not any(e.aeb_brake or e.aeb_warn for e in ev)
    assert not any(3 in e.colliding_ids for e in ev)


def test_module_reads_only_the_state_buffer():
    # Player data (Steam ids) lives in ETS2LAMpPlayers; MonoCruise never opens it.
    assert TS._STATE_TAG.endswith("ETS2LAMpState")
    assert TS._STATE_SIZE == struct.calcsize(TS._STATE_FORMAT)
