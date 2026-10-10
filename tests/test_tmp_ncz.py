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
    ENTER_CONFIRM_S, STALE_AFTER_S, TELEPORT_JUMP_M, NoCollisionZoneGate, TmpState,
    TmpStateReader, decode_state, ncz_vehicle_ids,
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
])
def test_the_zone_needs_a_live_connected_plugin_saying_so(field, value):
    assert _GHOSTS.in_zone
    assert not replace(_GHOSTS, **{field: value}).in_zone


def test_per_player_collision_counts_do_not_decide_the_zone():
    # TruckersMP's CanCollideWith read "can collide" for trucks overlapping ego in a zone
    # and "cannot" for a truck closing head-on outside one (2026-10-06): diagnostics only.
    assert replace(_GHOSTS, players_collidable=12).in_zone
    assert replace(_GHOSTS, players_streamed=0, players_collidable=0).in_zone


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
    assert not r.read(5.0).in_zone


def test_reader_without_shared_memory_reads_as_no_data():
    r = TmpStateReader()
    r._open_failed = True
    assert r.read(1.0) == TmpState()


def test_gate_needs_the_zone_to_hold_and_releases_at_once():
    gate = NoCollisionZoneGate()
    assert not gate.step(_GHOSTS, 10.0)
    assert not gate.step(_GHOSTS, 10.0 + ENTER_CONFIRM_S - 0.01)
    assert gate.step(_GHOSTS, 10.0 + ENTER_CONFIRM_S)
    assert not gate.step(replace(_GHOSTS, in_no_collision_zone=False), 10.5)
    # Re-entry starts the confirmation over.
    assert not gate.step(_GHOSTS, 10.6)
    assert gate.step(_GHOSTS, 10.6 + ENTER_CONFIRM_S)


def test_a_teleport_keeps_the_gate_shut_until_the_plugin_reports_no_zone():
    gate = NoCollisionZoneGate()
    gate.step(_GHOSTS, 0.0, (100.0, 100.0))
    assert gate.step(_GHOSTS, 1.0, (101.0, 100.0))
    far = (101.0 + TELEPORT_JUMP_M + 1.0, 100.0)
    # A stuck zone flag after a ferry or respawn must not keep AEB blind.
    assert not gate.step(_GHOSTS, 1.1, far)
    assert not gate.step(_GHOSTS, 5.0, far)
    # A stale or disconnected blip is not the plugin saying "no zone".
    assert not gate.step(replace(_GHOSTS, fresh=False), 5.1, far)
    assert not gate.step(_GHOSTS, 6.0, far)
    assert not gate.step(replace(_GHOSTS, in_no_collision_zone=False), 6.1, far)
    assert not gate.step(_GHOSTS, 6.2, far)
    assert gate.step(_GHOSTS, 6.2 + ENTER_CONFIRM_S + 0.01, far)


def test_driving_never_counts_as_a_teleport():
    gate = NoCollisionZoneGate()
    x = 0.0
    for i in range(300):
        x += 1.5  # 45 m/s at 30 Hz
        gate.step(_GHOSTS, i / 30.0, (x, 0.0))
    assert gate.active


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
    # The state comes from MonoCruise's own TruckersMP plugin; it carries no player identity.
    assert TS._STATE_TAG.endswith("MonoCruiseTmpState")
    assert TS._STATE_SIZE == struct.calcsize(TS._STATE_FORMAT)


def _left(**kw) -> TmpState:
    return replace(_GHOSTS, in_no_collision_zone=False, **kw)


def _open_gate() -> NoCollisionZoneGate:
    gate = NoCollisionZoneGate()
    gate.step(_GHOSTS, 0.0)
    assert gate.step(_GHOSTS, 1.0)
    return gate


def test_gate_flags_only_a_reported_exit():
    gate = _open_gate()
    assert not gate.step(_left(), 1.1)
    assert gate.exited_zone
    assert not gate.step(_left(), 1.2)
    assert not gate.exited_zone
    # A stale writer or a dropped connection is not TruckersMP saying the zone ended.
    for lost in (_left(fresh=False), _left(connected=False), replace(_GHOSTS, fresh=False)):
        gate = _open_gate()
        gate.step(lost, 1.1)
        assert not gate.exited_zone
    # Never opened, nothing to exit.
    closed = NoCollisionZoneGate()
    closed.step(_left(), 0.0)
    assert not closed.exited_zone


def test_a_teleport_is_not_a_zone_exit():
    gate = NoCollisionZoneGate()
    gate.step(_GHOSTS, 0.0, (0.0, 0.0))
    assert gate.step(_GHOSTS, 1.0, (0.0, 0.0))
    gate.step(_left(), 1.1, (TELEPORT_JUMP_M + 10.0, 0.0))
    assert not gate.exited_zone


_EGO = TS.ego_box(0.0, 0.0, 0.0, None)   # reference rig, centre at the origin, facing -z


def _ghost(vid, x, z, **kw):
    return make_vehicle(vid, x, z, 0.0, 0.0, is_tmp=True, **kw)


def test_exit_hold_keeps_players_inside_ego_until_they_separate():
    hold = TS.ExitGhostHold()
    inside = _ghost(5, 0.0, -7.333)                # rear 1.0 m into ego's front
    assert hold.step(False, True, _EGO, [inside]) == {5}
    assert hold.step(False, False, _EGO, [inside]) == {5}
    clear = _ghost(5, 0.0, -9.0)
    assert hold.step(False, False, _EGO, [clear]) == frozenset()
    # Separated once means collidable for good, as in TruckersMP.
    assert hold.step(False, False, _EGO, [inside]) == frozenset()


def test_exit_hold_never_takes_a_body_ego_is_not_clearly_inside():
    hold = TS.ExitGhostHold()
    neighbour = _ghost(6, 3.6, 0.0)                # next lane
    grazing = _ghost(7, 0.0, -8.333 + 0.1)         # 0.1 m of box contact, under the arm inset
    ai = make_vehicle(8, 0.0, -5.0, 0.0, 0.0)      # AI traffic is never a ghost
    assert hold.step(False, True, _EGO, [neighbour, grazing, ai]) == frozenset()
    # Only the exit frame arms; a ghost that comes inside later is a real truck.
    assert hold.step(False, False, _EGO, [_ghost(9, 0.0, -5.0)]) == frozenset()


def test_exit_hold_takes_the_whole_rig():
    hold = TS.ExitGhostHold()
    trailer = _ghost(11, 0.0, -6.0, is_trailer=True)
    tractor = _ghost(10, 0.0, -17.0, length=6.0)    # ahead of its trailer, clear of ego
    nested = _ghost(1_000_000 + 11 * 4, 0.0, -1.0e3, is_trailer=True)
    other = _ghost(12, 0.0, -40.0)
    assert hold.step(False, True, _EGO, [trailer, tractor, other], [nested]) == {
        10, 11, 1_000_000 + 11 * 4,
    }
    # The tractor stays a ghost while any part of the rig is still inside ego.
    assert hold.step(False, False, _EGO, [trailer, tractor]) == {10, 11, 1_000_000 + 11 * 4}
    assert hold.step(False, False, _EGO, [_ghost(11, 0.0, -30.0, is_trailer=True), tractor]) == frozenset()


def test_exit_hold_clears_when_the_zone_returns_or_the_pose_is_lost():
    inside = _ghost(5, 0.0, -6.0)
    for args in ((True, False, _EGO), (False, False, None)):
        hold = TS.ExitGhostHold()
        hold.step(False, True, _EGO, [inside])
        assert hold.step(*args, [inside]) == frozenset()


def test_ego_box_follows_the_wheel_geometry():
    from core.radar.ego_geometry import EgoGeometry
    g = EgoGeometry(half_width_m=1.3, front_m=5.0, rear_m=3.0, path_origin_m=-2.0)
    box = TS.ego_box(10.0, 20.0, 0.0, g)
    assert box.cx == pytest.approx(10.0) and box.cz == pytest.approx(19.0)
    assert box.half_length == pytest.approx(4.0) and box.half_width == pytest.approx(1.3)


def test_clip_frames_carry_the_exit_flag():
    rec = RadarFrameRecord(t_wall=1.0, t_mono=1.0, tmp_ncz_exit=True)
    assert RadarFrameRecord.from_json(rec.to_json()).tmp_ncz_exit
    assert "tmp_ncz_exit" not in RadarFrameRecord(t_wall=1.0, t_mono=1.0).to_json()


def _zone_exit_clip(offset_m: float, exit_reported: bool, v: float = 20.0) -> Clip:
    """Zone for 3 s, then the exit, with a stopped ghost ``offset_m`` ahead of ego's origin."""
    n, k, hz = 150, 90, 30.0
    meta = ClipMetadata.create(trigger_source="auto_engagement", session_kind="TMP")
    frames, ticks = [], []
    for i in range(n):
        t = i / hz
        frames.append(RadarFrameRecord(
            t_wall=1000.0 + t, t_mono=t,
            ego=EgoTelemetry(coordinateX=0.0, coordinateZ=v * t, rotationX=0.5,
                             rotationY=0.0, speed=v),
            traffic_buf=_stopped_tmp_truck_buf(v * k / hz + offset_m, vid=3),
            tmp_ncz=i < k, tmp_ncz_exit=exit_reported and i == k,
        ))
        ticks.append(AEBTickRecord(
            t_mono=t, radar_t_mono=t,
            consumed=ConsumedContext(max_brake_ms2=10.0, aeb_enabled=True),
            live_aeb=LiveAEB(),
        ))
    return Clip(metadata=meta, radar_frames=frames, aeb_ticks=ticks)


def test_aeb_ignores_a_ghost_still_inside_ego_after_the_zone():
    # Without the hold this braked 3 frames after the exit (2026-10-10).
    assert any(e.aeb_brake for e in run_headless(_zone_exit_clip(4.0, exit_reported=False)))
    ev = run_headless(_zone_exit_clip(4.0, exit_reported=True))
    assert not any(e.aeb_brake or e.aeb_warn for e in ev)


def test_aeb_brakes_for_a_truck_that_was_clear_of_ego_at_the_exit():
    # 1.7 m ahead of ego's bumper when the zone ended: collidable in TruckersMP.
    assert any(e.aeb_brake for e in run_headless(_zone_exit_clip(8.0, exit_reported=True)))


def _aeb_frame(vehicles, ncz_ids, latched=frozenset()):
    from core.aeb.calibration import DEFAULT
    from core.aeb.clip_eval import _make_headless, _snapshot_tuple

    t = _make_headless(DEFAULT)
    t._latched_threat_ids = set(latched)
    ego = EgoTelemetry(coordinateX=0.0, coordinateZ=0.0, rotationX=0.0, speed=20.0)
    snap = _snapshot_tuple(ego, vehicles, 10.0, frozenset(ncz_ids), 10.0, frozenset(ncz_ids))
    t._read_radar_snapshot = lambda: snap
    t._read_ego_geometry = lambda: None
    t._read_max_brake_ms2 = lambda: 7.8
    t._read_user_braking = lambda: False
    t._read_addressing_brake = lambda: False
    t._read_vehicle_key = lambda: None
    t._now = lambda: 10.0
    t.loop()
    return t.data.snapshot


def test_debug_snapshot_shows_ghosts_without_evaluating_them():
    ghost = make_vehicle(7, 0.0, -30.0, 0.0, 0.0, is_tmp=True)
    ai = make_vehicle(8, 3.5, -40.0, 0.0, 0.0)
    snap = _aeb_frame([ghost, ai], ncz_ids={7})
    by_id = {v["vid"]: v for v in snap.vehicles}
    assert by_id[7]["ghost"] is True
    assert not by_id[8].get("ghost", False)
    assert snap.ghost_ids == {7}
    assert 7 not in snap.vehicle_arcs and 7 not in snap.colliding_ids


def test_a_latched_ghost_is_drawn_as_a_threat_not_a_ghost():
    ghost = make_vehicle(7, 0.0, -30.0, 0.0, 0.0, is_tmp=True)
    snap = _aeb_frame([ghost], ncz_ids={7}, latched={7})
    assert snap.ghost_ids == set()
    assert not any(v.get("ghost") for v in snap.vehicles)
