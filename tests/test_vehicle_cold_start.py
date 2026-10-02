"""First-sighting speed: live anchor pairing and the replay-only seed. See core/radar/README.md section 7."""
from __future__ import annotations

import math
import struct

from core.aeb.clip_replay import cold_start_speeds, decode_radar_stream
from core.aeb.clip_schema import (
    AEBTickRecord, Clip, ClipMetadata, ConsumedContext, EgoTelemetry, LiveAEB,
    RadarFrameRecord,
)
from core.radar.elevation import BODY_DATUM_FRAC
from core.radar.reader import _BUF_SIZE, _TOTAL_FORMAT, TrafficReader
from core.radar.traffic import Position, Quaternion, Size, Vehicle

_HZ = 30.0
_BODY_H = 3.0
_BODY_Y = BODY_DATUM_FRAC * _BODY_H
_LEAD_MS = 20.0


def _straight_vehicle(t_idx: int, dt: float, speed: float, vid: int = 7) -> Vehicle:
    """AI car on the -Z axis; yaw 0 makes fwd = (0, -1)."""
    return Vehicle(
        Position(0.0, 0.0, -speed * t_idx * dt), Quaternion(1.0, 0.0, 0.0, 0.0),
        Size(2.0, 1.5, 4.5), speed, 0.0, 0, [], vid, False, False,
    )


def test_a_newly_seen_vehicle_does_not_read_half_its_speed():
    """Sub-frames freeze Vehicle.time, so the pose carried with it must match."""
    dt = 1.0 / _HZ
    prev = _straight_vehicle(0, dt, _LEAD_MS)
    prev.time = 0.0
    speeds = []
    for k in range(1, 8):
        cur = _straight_vehicle(k, dt, _LEAD_MS)
        cur.update_from_last(prev, k * dt, 0.0, 0.0, 50.0, 25.0)
        speeds.append(cur.speed)
        prev = cur
    assert max(abs(s - _LEAD_MS) for s in speeds) < 0.05, speeds


def _traffic_buf(pz: float, speed: float, vid: int = 7) -> bytes:
    """One-slot buffer; quaternion (0, 0, 1, 0) is yaw pi, so the lead runs up +Z."""
    flat: list = [0.0, _BODY_Y, pz, 0.0, 0.0, 1.0, 0.0, 2.5, _BODY_H, 6.0, speed, 0.0]
    flat += [0, vid, 0, 0] + [0.0] * 30
    for _ in range(39):
        flat += [0.0] * 12 + [0, 0, 0, 0] + [0.0] * 30
    buf = struct.pack(_TOTAL_FORMAT, *flat)
    assert len(buf) == _BUF_SIZE
    return buf


def _lead_clip(buffer_speed: float, n: int = 40) -> Clip:
    """A lead already at _LEAD_MS when the clip window opens."""
    dt = 1.0 / _HZ
    frames, ticks = [], []
    for i in range(n):
        t = i * dt
        frames.append(RadarFrameRecord(
            t_wall=1000.0 + t, t_mono=t,
            ego=EgoTelemetry(coordinateZ=0.0, rotationX=0.5, speed=25.0),
            traffic_buf=_traffic_buf(50.0 + _LEAD_MS * t, buffer_speed),
            parked_buf=None,
        ))
        ticks.append(AEBTickRecord(
            t_mono=t, radar_t_mono=t,
            consumed=ConsumedContext(max_brake_ms2=10.0),
            live_aeb=LiveAEB(),
        ))
    return Clip(
        metadata=ClipMetadata.create(trigger_source="auto_engagement",
                                     session_kind="SP"),
        radar_frames=frames, aeb_ticks=ticks,
    )


def test_replay_measures_the_start_speed_instead_of_assuming_zero():
    """A clip window is a cut of a continuous stream: nothing in frame 0 is at rest."""
    clip = _lead_clip(buffer_speed=0.0)
    assert abs(cold_start_speeds(clip)[7] - _LEAD_MS) < 0.05

    veh_by_t, _ego, frame_t, _off = decode_radar_stream(clip)
    first = veh_by_t[frame_t[0]][0]
    assert abs(first.speed - _LEAD_MS) < 0.05, first.speed


def test_the_seeded_replay_has_no_start_of_clip_transient():
    """The jump this fixes: 0 km/h, then a ramp, on every clip."""
    veh_by_t, _ego, frame_t, _off = decode_radar_stream(_lead_clip(buffer_speed=0.0))
    trace = [veh_by_t[ft][0].speed for ft in frame_t[:20]]
    assert max(abs(s - _LEAD_MS) for s in trace) < 0.1, trace


def test_curvature_stays_unknown_until_three_real_samples():
    """The seed is two samples on purpose: a straight prehistory must not claim kappa."""
    veh_by_t, _ego, frame_t, _off = decode_radar_stream(_lead_clip(buffer_speed=0.0))
    assert veh_by_t[frame_t[0]][0].curvature_from_history() is None


def test_live_reads_are_never_given_a_seed():
    """Reading ahead is only possible offline; a live reader must stay cold."""
    assert TrafficReader()._cold_start_speeds == {}
