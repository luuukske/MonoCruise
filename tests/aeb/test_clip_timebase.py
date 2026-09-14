"""Replay timebase: legacy ego re-pairing and the simulated clock. See core/aeb/README.md section 16."""
from __future__ import annotations

import math
import struct

import pytest

from core.aeb.clip_replay import decode_radar_stream
from core.aeb.clip_schema import (
    AEBTickRecord, Clip, ClipMetadata, ConsumedContext, EgoTelemetry, LiveAEB,
    RadarFrameRecord,
)
from core.aeb.clip_timebase import pairing_lag_steps, replay_frames
from core.radar.elevation import BODY_DATUM_FRAC
from core.radar.reader import _BUF_SIZE, _TOTAL_FORMAT

_STEP_HZ = 60.0
_FRAME_S = 1.0 / 30.0
_WALL_EPOCH = 1_780_000_000.0
_BODY_H = 3.0
_BODY_Y = BODY_DATUM_FRAC * _BODY_H
_EGO_V0 = 22.0
_EGO_DECEL = 0.8
# Read jitter and the telemetry-poll staleness pattern of a legacy capture.
_JITTER_S = (0.0, 0.004, -0.005, 0.006, -0.002, 0.003, -0.006, 0.001, 0.005, -0.004)
_LEGACY_STALE = (0, 1, 2, 1, 0, 1, 2, 2, 1, 0, 1, 0, 2)


def _ego_track(steps: int) -> tuple[list[float], list[float]]:
    """Game-style ego: each step advances by the speed read at the end of that step."""
    speeds, zs = [_EGO_V0], [0.0]
    for k in range(1, steps + 1):
        speeds.append(_EGO_V0 - _EGO_DECEL * k / _STEP_HZ)
        zs.append(zs[-1] + speeds[-1] / _STEP_HZ)
    return speeds, zs


def _traffic_buf(cars: list[tuple[float, float, float, int]], is_tmp: bool) -> bytes:
    """Cars running up +Z: (x, z, buffer speed, id). Quaternion (0, 0, 1, 0) is yaw pi."""
    flat: list = []
    for x, z, speed, vid in cars:
        flat += [x, _BODY_Y, z, 0.0, 0.0, 1.0, 0.0, 2.5, _BODY_H, 6.0, speed, 0.0]
        flat += [0, vid, 1 if is_tmp else 0, 0] + [0.0] * 30
    for _ in range(40 - len(cars)):
        flat += [0.0] * 12 + [0, 0, 0, 0] + [0.0] * 30
    buf = struct.pack(_TOTAL_FORMAT, *flat)
    assert len(buf) == _BUF_SIZE
    return buf


def _game_clip(schema: int, stale: tuple[int, ...], is_tmp: bool = False, frames: int = 150):
    """Frames sampled like live radar; returns the clip and each frame's traffic step."""
    car_speeds = (20.0, 25.0, 18.0, 23.0)
    car_lanes = (0.0, 3.5, -3.5, 7.0)
    speeds, zs = _ego_track(int(frames * _FRAME_S * _STEP_HZ) + 10)
    out, ticks, traffic_steps = [], [], []
    k_ego = 0
    for i in range(frames):
        t_read = 0.2 + i * _FRAME_S + _JITTER_S[i % len(_JITTER_S)]
        k_traffic = int(t_read * _STEP_HZ)
        # A telemetry poll never goes back in time, whatever the staleness pattern says.
        k_ego = max(k_ego, k_traffic - stale[i % len(stale)])
        cars = [
            (lane, 40.0 + 12.0 * n + v * k_traffic / _STEP_HZ, 0.0 if is_tmp else v, 7 + n)
            for n, (v, lane) in enumerate(zip(car_speeds, car_lanes))
        ]
        out.append(RadarFrameRecord(
            t_wall=_WALL_EPOCH + t_read, t_mono=t_read,
            ego=EgoTelemetry(coordinateZ=zs[k_ego], rotationX=0.5, speed=speeds[k_ego]),
            traffic_buf=_traffic_buf(cars, is_tmp), parked_buf=None,
        ))
        ticks.append(AEBTickRecord(
            t_mono=t_read, radar_t_mono=t_read,
            consumed=ConsumedContext(max_brake_ms2=10.0), live_aeb=LiveAEB(),
        ))
        traffic_steps.append(k_traffic)
    meta = ClipMetadata.create(session_kind="TMP" if is_tmp else "SP")
    meta.schema_version = schema
    return Clip(metadata=meta, radar_frames=out, aeb_ticks=ticks), traffic_steps, zs


def _ego_step_of(frame: RadarFrameRecord, zs: list[float]) -> int:
    """The physics step a recorded ego pose was sampled on."""
    return min(range(len(zs)), key=lambda k: abs(zs[k] - frame.ego.coordinateZ))


@pytest.mark.parametrize("is_tmp", [False, True])
def test_legacy_ego_pose_is_re_paired_onto_the_traffic_step(is_tmp):
    clip, k_traffic, zs = _game_clip(schema=4, stale=_LEGACY_STALE, is_tmp=is_tmp)
    recorded = [abs(f.ego.coordinateZ - zs[k]) for f, k in zip(clip.radar_frames, k_traffic)]
    assert max(recorded) > 0.5, "the fixture must carry legacy pairing error"

    frames = replay_frames(clip)
    errors = [abs(f.ego.coordinateZ - zs[k]) for f, k in zip(frames, k_traffic)]
    assert max(errors) < 0.01, max(errors)


def test_replay_clock_is_simulated_time_on_the_traffic_step():
    clip, k_traffic, _ = _game_clip(schema=4, stale=_LEGACY_STALE)
    frames = replay_frames(clip)
    for f, k in zip(frames, k_traffic):
        expected = (k - k_traffic[0]) / _STEP_HZ
        assert f.t_wall - frames[0].t_wall == pytest.approx(expected, abs=1e-6)


def test_current_schema_keeps_its_pose_and_still_gets_the_clock():
    clip, k_traffic, zs = _game_clip(schema=5, stale=(0,))
    frames = replay_frames(clip)
    assert [f.ego.coordinateZ for f in frames] == [f.ego.coordinateZ for f in clip.radar_frames]
    steps = [(f.t_wall - frames[0].t_wall) * _STEP_HZ for f in frames]
    assert steps == pytest.approx([float(k - k_traffic[0]) for k in k_traffic], abs=1e-4)


def test_replay_never_mutates_the_clip():
    clip, _, _ = _game_clip(schema=4, stale=_LEGACY_STALE)
    before = [(f.t_wall, f.ego.coordinateZ, f.ego.speed) for f in clip.radar_frames]
    replay_frames(clip)
    decode_radar_stream(clip)
    assert [(f.t_wall, f.ego.coordinateZ, f.ego.speed) for f in clip.radar_frames] == before


def test_decoded_stream_hands_the_re_paired_pose_to_consumers():
    clip, k_traffic, zs = _game_clip(schema=4, stale=_LEGACY_STALE)
    _veh, ego_by_t, _frame_t, _off = decode_radar_stream(clip)
    for f, k in zip(clip.radar_frames, k_traffic):
        assert abs(ego_by_t[f.t_mono].coordinateZ - zs[k]) < 0.01


def test_a_clip_without_physics_steps_keeps_wall_time():
    """Synthetic clips move continuously; the counts cannot explain them, so nothing changes."""
    frames, ticks = [], []
    for i in range(40):
        t = i * _FRAME_S
        frames.append(RadarFrameRecord(
            t_wall=1000.0 + t, t_mono=t,
            ego=EgoTelemetry(rotationX=0.5, speed=25.0),
            traffic_buf=_traffic_buf([(0.0, 50.0 + 20.0 * t, 0.0, 7)], is_tmp=False),
            parked_buf=None,
        ))
        ticks.append(AEBTickRecord(t_mono=t, radar_t_mono=t))
    meta = ClipMetadata.create()
    meta.schema_version = 4
    clip = Clip(metadata=meta, radar_frames=frames, aeb_ticks=ticks)
    out = replay_frames(clip)
    assert [(f.t_wall, f.ego.coordinateZ) for f in out] == [(f.t_wall, f.ego.coordinateZ) for f in frames]


def test_pairing_lag_is_measured_from_the_freshest_ego_sample():
    stale = [1, 2, 0, 1, 3, 2, 1]
    ego_steps = [None] + [2 - (b - a) for a, b in zip(stale, stale[1:])]
    traffic_steps = [None] + [2] * (len(stale) - 1)
    assert pairing_lag_steps(traffic_steps, ego_steps) == stale


def test_uncounted_pairs_hold_the_lag_instead_of_guessing():
    ego_steps = [None, 2, None, 3, 2]
    traffic_steps = [None, 3, 2, None, 2]
    assert pairing_lag_steps(traffic_steps, ego_steps) == [0, 1, 1, 1, 1]


def test_ego_heading_advance_follows_the_turn():
    """A turning ego is advanced along the arc midpoint, not the stale heading."""
    clip, k_traffic, zs = _game_clip(schema=4, stale=_LEGACY_STALE)
    yaw_per_step = 0.0005
    for f in clip.radar_frames:
        f.ego.rotationX = 0.5 + yaw_per_step * _ego_step_of(f, zs)
    frames = replay_frames(clip)
    for f, k in zip(frames, k_traffic):
        assert f.ego.rotationX == pytest.approx(0.5 + yaw_per_step * k, abs=1e-9)
        assert math.isfinite(f.ego.coordinateX)
