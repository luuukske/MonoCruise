"""TMP speed from the TruckersMP SDK, and the packet-age lag / crash split. See core/radar/README.md §19.

A synthetic TMP truck sends its own velocity at ~4.5 Hz the way TruckersMP does. The
client extrapolates it for 300 ms past a late packet and then freezes it.
"""
from __future__ import annotations

import math

from core.aeb.clip_schema import RadarFrameRecord
from core.radar import tmp_sdk
from core.radar.tmp_sdk import (
    FLAG_LINEAR, FLAG_PLACEMENT, FLAG_SAMPLED, KIND_TRACTOR, KIND_TRAILER, SdkBody, SdkFrame,
    attach, crash_evidence, decode_frame, encode_frame,
)
from core.radar.traffic import _CRASH_HOLD_S, Position, Quaternion, Size, Vehicle

DT = 0.0667
PACKET_S = 0.22
EXTRAPOLATE_S = 0.30
FLAGS = FLAG_PLACEMENT | FLAG_LINEAR | FLAG_SAMPLED


def _body(x: float, z: float, vz: float, age_ms: float, updates: int, arrival_s: float,
          kind: int = KIND_TRACTOR, vx: float = 0.0) -> SdkBody:
    return SdkBody(x, 1.0, z, 1.0, 0.0, 0.0, 0.0, vx, 0.0, vz, 0.0, 0.0, 0.0,
                   age_ms, updates, 0.0, kind, FLAGS, 0, arrival_s)


def _vehicle(z: float, vid: int = 1, is_trailer: bool = False, is_tmp: bool = True) -> Vehicle:
    return Vehicle(Position(0.0, 1.0, z), Quaternion(1.0, 0.0, 0.0, 0.0), Size(2.5, 3.0, 6.0),
                   0.0, 0.0, 0, [], vid, is_tmp, is_trailer)


class Sender:
    """Remote truck at yaw 0 (moving toward -z), its packets, and the pose the client shows."""

    def __init__(self, speed: float):
        self.t = 0.0
        self.z = 0.0
        self.speed = speed
        self.shown_z = 0.0
        self.pkt_t = 0.0
        self.pkt_v = speed
        self.pkt_z = 0.0
        self.updates = 1
        self.vx = 0.0

    def advance(self, dt: float, accel: float = 0.0, send: bool = True, freeze_after_s: float = EXTRAPOLATE_S):
        self.t += dt
        self.speed = max(0.0, self.speed + accel * dt)
        self.z -= self.speed * dt
        if send and self.t - self.pkt_t >= PACKET_S - 1e-9:
            self.pkt_t, self.pkt_v, self.pkt_z = self.t, self.speed, self.z
            self.updates += 1
        since = self.t - self.pkt_t
        self.shown_z = self.pkt_z - self.pkt_v * min(since, freeze_after_s)

    def body(self) -> SdkBody:
        return _body(0.0, self.shown_z, -self.pkt_v, (self.t - self.pkt_t) * 1000.0,
                     self.updates, self.pkt_t, vx=self.vx)


def _frame(prev: Vehicle, sender: Sender, ego_z: float, ego_speed: float, with_sdk: bool = True) -> Vehicle:
    cur = _vehicle(sender.shown_z)
    attach([cur], SdkFrame(sender.t, True, (sender.body(),)) if with_sdk else None)
    cur.update_from_last(prev, sender.t, 0.0, 1.0, ego_z, ego_speed)
    return cur


def _cruise(sender: Sender, seconds: float, ego_z: float = 300.0, ego_speed: float = 0.0) -> Vehicle:
    v = _vehicle(sender.shown_z)
    v.time = 0.0
    for _ in range(int(seconds / DT)):
        sender.advance(DT)
        v = _frame(v, sender, ego_z, ego_speed)
    return v


def test_matched_truck_reads_the_senders_speed():
    sender = Sender(20.0)
    v = _cruise(sender, 3.0)
    assert abs(v.speed - 20.0) < 0.05
    assert abs(v.acc_speed - 20.0) < 0.3
    assert not v.netcode_stall and not v.crash_confirmed


def test_a_far_frozen_stall_holds_the_senders_speed():
    sender = Sender(20.0)
    v = _cruise(sender, 2.0)
    speeds, stalls = [], []
    for _ in range(int(0.6 / DT)):
        sender.advance(DT, send=False)
        v = _frame(v, sender, 300.0, 0.0)
        speeds.append(v.speed)
        stalls.append(v.netcode_stall)
    assert any(stalls), "late packets must flag the stall"
    assert min(speeds) > 19.5
    assert not v.crash_confirmed


def test_a_close_frozen_stall_reads_as_a_stop():
    """Inside 0.3 s TTC the frozen body is what ego can hit, exactly like today's lag freeze."""
    sender = Sender(20.0)
    v = _cruise(sender, 2.0, ego_z=8.0, ego_speed=20.0)
    for _ in range(int(0.6 / DT)):
        sender.advance(DT, send=False)
        v = _frame(v, sender, sender.shown_z + 5.0, 20.0)
    assert v.speed < 8.0


def test_a_close_body_still_extrapolated_keeps_its_speed():
    """Packet age alone is not a stop: the client still moves the body in ego's world."""
    sender = Sender(20.0)
    v = _cruise(sender, 2.0, ego_z=8.0, ego_speed=20.0)
    for _ in range(int(0.6 / DT)):
        sender.advance(DT, send=False, freeze_after_s=10.0)
        v = _frame(v, sender, sender.shown_z + 5.0, 20.0)
        assert v.speed > 19.5


def test_a_real_stop_tracks_down_without_a_crash():
    sender = Sender(20.0)
    v = _cruise(sender, 2.0)
    worst_lag = 0.0
    for _ in range(int(4.0 / DT)):
        sender.advance(DT, accel=-7.0)
        v = _frame(v, sender, 300.0, 0.0)
        worst_lag = max(worst_lag, v.speed - sender.speed)
        assert not v.crash_confirmed
    assert v.speed < 0.5
    assert worst_lag < 2.0, "one packet interval of 7 m/s2 is about 1.9 m/s, AEB adds no filter lag"


def test_a_crash_between_two_packets_confirms():
    sender = Sender(20.0)
    v = _cruise(sender, 2.0)
    while sender.t - sender.pkt_t < PACKET_S - DT:
        sender.advance(DT)
        v = _frame(v, sender, 300.0, 0.0)
    sender.speed = 3.0
    for _ in range(int(0.5 / DT)):
        sender.advance(DT)
        v = _frame(v, sender, 300.0, 0.0)
        if v.crash_confirmed:
            break
    assert v.crash_confirmed and v.sdk_crash
    t_hit = sender.t
    for _ in range(int((_CRASH_HOLD_S + 0.5) / DT)):
        sender.advance(DT)
        v = _frame(v, sender, 300.0, 0.0)
        if sender.t < t_hit + _CRASH_HOLD_S - DT:
            assert v.crash_confirmed
    assert not v.crash_confirmed


def test_a_stall_never_confirms_a_crash():
    sender = Sender(20.0)
    v = _cruise(sender, 2.0)
    for _ in range(int(0.65 / DT)):
        sender.advance(DT, send=False)
        v = _frame(v, sender, 300.0, 0.0)
        assert not v.crash_confirmed
    for _ in range(int(1.0 / DT)):
        sender.advance(DT)
        v = _frame(v, sender, 300.0, 0.0)
        assert not v.crash_confirmed


def test_crash_evidence_bars_sit_above_normal_driving():
    """Crash-free TMP driving (2026-10-12, 30 min): up to 24 m/s2, 156 deg/s slow, 123 deg/s above 8 m/s."""
    assert not crash_evidence(0.0, -6.9, 0.0, -1.8, 0.213)       # the 24 m/s2 low-speed stop
    assert not crash_evidence(0.0, -16.6, 0.0, -12.8, 0.217)     # 17.5 m/s2 at speed
    assert crash_evidence(0.0, -20.0, 0.0, -10.0, 0.22)          # 45 m/s2, 10 m/s gone
    assert not crash_evidence(0.0, -10.0, 0.0, -5.0, 0.10)       # 50 m/s2 but under 6 m/s lost
    slow = math.radians(156.0 * 0.22)
    assert not crash_evidence(0.0, -6.0, -6.0 * math.sin(slow), -6.0 * math.cos(slow), 0.22)
    turn = math.radians(123.0 * 0.22)
    assert not crash_evidence(0.0, -10.0, -10.0 * math.sin(turn), -10.0 * math.cos(turn), 0.22)
    spin = math.radians(210.0 * 0.22)
    assert crash_evidence(0.0, -10.0, -10.0 * math.sin(spin), -10.0 * math.cos(spin), 0.22)
    assert crash_evidence(0.0, -10.0, 0.0, 8.0, 0.22)            # knocked back
    assert not crash_evidence(0.0, -20.0, 0.0, 0.0, 0.01)        # samples too close to judge


def test_a_reused_id_does_not_fake_a_crash():
    sender = Sender(20.0)
    v = _cruise(sender, 2.0)
    sender.updates = 3
    sender.pkt_v = 0.0
    sender.advance(DT)
    v = _frame(v, sender, 300.0, 0.0)
    assert not v.crash_confirmed


def test_without_sdk_the_position_fit_runs_as_before():
    sender = Sender(20.0)
    a = _vehicle(0.0)
    a.time = 0.0
    b = _vehicle(0.0)
    b.time = 0.0
    for _ in range(int(2.0 / DT)):
        sender.advance(DT)
        a = _frame(a, sender, 300.0, 0.0, with_sdk=False)
        nb = _vehicle(sender.shown_z)
        nb.update_from_last(b, sender.t, 0.0, 1.0, 300.0, 0.0)
        b = nb
        assert a.speed == b.speed and a.acc_speed == b.acc_speed


def test_attach_pairs_by_place_and_kind():
    tractor, trailer, far, ai = _vehicle(0.0, 1), _vehicle(-8.0, 2, True), _vehicle(-40.0, 3), _vehicle(0.5, 4, is_tmp=False)
    bodies = (
        _body(0.4, 0.0, -10.0, 50.0, 5, 1.0),
        _body(0.0, -7.0, -10.0, 50.0, 5, 1.0, kind=KIND_TRAILER),
        _body(0.0, -50.0, -10.0, 50.0, 5, 1.0),
    )
    attach([tractor, trailer, far, ai], SdkFrame(1.05, True, bodies))
    assert tractor.sdk is bodies[0]
    assert trailer.sdk is bodies[1]
    assert far.sdk is None, "10 m off is another truck"
    assert ai.sdk is None


def test_attach_is_off_with_the_switch(monkeypatch):
    v = _vehicle(0.0)
    monkeypatch.setattr(tmp_sdk, "SDK_SPEED_ENABLED", False)
    attach([v], SdkFrame(1.0, True, (_body(0.0, 0.0, -10.0, 50.0, 5, 1.0),)))
    assert v.sdk is None


def test_frames_survive_the_clip_round_trip():
    raw = tmp_sdk.HEADER.pack(1, 7, 10_000_000, 0xFFFF, 1, 1) + tmp_sdk.SLOT.pack(
        1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -12.0, 0.0, 0.0, 0.0, 100.0, 9, 5.0, 0, FLAGS, 0)
    frame = decode_frame(raw, 10.05)
    assert abs(frame.bodies[0].age_ms - 150.0) < 1e-3
    assert abs(frame.bodies[0].arrival_s - 9.9) < 1e-6
    assert decode_frame(encode_frame(frame)) == frame
    rec = RadarFrameRecord(t_wall=1.0, t_mono=2.0, tmp_sdk_buf=encode_frame(frame))
    assert RadarFrameRecord.from_json(rec.to_json()).tmp_sdk_buf == rec.tmp_sdk_buf
    assert "tmp_sdk_buf" not in RadarFrameRecord(t_wall=1.0, t_mono=2.0).to_json()


def test_a_clip_keeps_only_matched_bodies_and_replays_the_same_pairs():
    recs = [_vehicle(0.0, 1), _vehicle(-20.0, 2), _vehicle(-40.0, 3)]
    bodies = (
        _body(0.0, 0.3, -10.0, 50.0, 5, 1.0),
        _body(0.0, -200.0, -10.0, 50.0, 5, 1.0),
        _body(0.0, -20.4, -10.0, 50.0, 5, 1.0),
        _body(0.0, -0.9, -10.0, 50.0, 5, 1.0),
    )
    frame = SdkFrame(1.05, True, bodies)
    attach(recs, frame)
    live = [v.sdk for v in recs]
    kept = tmp_sdk.matched_only(frame, recs)
    assert len(kept.bodies) == 2
    replayed = [_vehicle(0.0, 1), _vehicle(-20.0, 2), _vehicle(-40.0, 3)]
    attach(replayed, decode_frame(encode_frame(kept)))
    assert [v.sdk is None for v in replayed] == [b is None for b in live]
    assert [(v.sdk.z if v.sdk else None) for v in replayed] == [(b.z if b else None) for b in live]


def test_a_long_frozen_stall_ramps_down_like_the_lag_freeze():
    """After the hold the speed ramps (1 - frac^2) over one freeze length, never in one step."""
    headway = 1.5
    sender = Sender(25.0)
    v = _cruise(sender, 3.0, ego_z=25.0 * headway, ego_speed=25.0)
    trace = []
    t0 = sender.t
    for _ in range(int(2.5 / DT)):
        sender.advance(DT, send=False)
        v = _frame(v, sender, sender.shown_z + 25.0 * headway, 25.0)
        trace.append((sender.t - t0, v.speed))
    freeze = 0.5 / math.log(4.0 / 0.3) * math.log(headway / 0.3)
    frozen_at = EXTRAPOLATE_S + DT
    assert all(s > 24.5 for t, s in trace if t < frozen_at + 2.0 * freeze - DT), "held through the hold"
    t_90 = next(t for t, s in trace if s < 22.5)
    t_10 = next(t for t, s in trace if s < 2.5)
    assert t_10 - t_90 >= 3 * DT - 1e-9, "ramped over several frames, not stepped"
    assert t_10 <= frozen_at + 3.0 * freeze + 0.3
    assert trace[-1][1] < 0.5


def _drop_on_next_packet(v: Vehicle, sender: Sender, to_speed: float) -> Vehicle:
    while sender.t - sender.pkt_t < PACKET_S - DT:
        sender.advance(DT)
        v = _frame(v, sender, 300.0, 0.0)
    sender.speed = to_speed
    sender.advance(DT)
    return _frame(v, sender, 300.0, 0.0)


def test_aeb_reads_a_crash_drop_the_frame_its_packet_lands():
    """AEB's speed is the packet's own: no EMA between a crash and AEB. ACC keeps its filter."""
    sender = Sender(25.0)
    v = _drop_on_next_packet(_cruise(sender, 3.0), sender, 4.0)
    assert sender.pkt_v == 4.0
    assert v.speed == 4.0
    assert v.acc_speed > 10.0


def test_aeb_raw_speed_switch_restores_the_filtered_speed(monkeypatch):
    monkeypatch.setattr(tmp_sdk, "AEB_RAW_SPEED", False)
    sender = Sender(25.0)
    v = _drop_on_next_packet(_cruise(sender, 3.0), sender, 4.0)
    assert v.speed > 10.0


def _dropout(v: Vehicle, sender: Sender, seconds: float, shown_speed: float) -> list[float]:
    """The SDK match is lost while the client shows the body at ``shown_speed``."""
    z, t, speeds = sender.shown_z, sender.t, []
    for _ in range(int(round(seconds / DT))):
        t += DT
        z -= shown_speed * DT
        cur = _vehicle(z)
        attach([cur], None)
        cur.update_from_last(v, t, 0.0, 1.0, 300.0, 0.0)
        v = cur
        speeds.append(v.speed)
    return speeds, v


def test_a_short_match_dropout_keeps_the_senders_speed():
    """Rubber-banding can push the shown pose past the match gate for a frame or four."""
    sender = Sender(20.0)
    speeds, _ = _dropout(_cruise(sender, 3.0), sender, 4 * DT, shown_speed=12.0)
    assert speeds == [20.0] * 4


def test_a_long_match_dropout_fades_to_the_position_fit_without_a_step():
    sender = Sender(20.0)
    speeds, v = _dropout(_cruise(sender, 3.0), sender, 1.5, shown_speed=12.0)
    assert max(abs(b - a) for a, b in zip(speeds, speeds[1:])) < 2.0
    assert v._sdk_aeb_speed is None
    assert speeds[-1] < 16.0


def test_a_crash_on_the_position_track_ends_the_dropout_hold():
    sender = Sender(20.0)
    v = _cruise(sender, 3.0)
    v._crash_hold_until = sender.t + 2.0
    speeds, v = _dropout(v, sender, DT, shown_speed=12.0)
    assert v.crash_confirmed
    assert v._sdk_aeb_speed is None
    assert speeds[0] != 20.0
