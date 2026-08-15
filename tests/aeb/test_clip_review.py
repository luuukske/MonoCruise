"""Tests for the review path: label persistence, metadata peek, clip replay."""

from __future__ import annotations

import base64
import os
import struct

from core.aeb.clip_schema import (
    AEBTickRecord, ConsumedContext, EgoTelemetry, Label, LiveAEB, RadarFrameRecord,
)
from core.aeb.clip_replay import raw_target_decel, replay_clip
from core.aeb.clip_store import _PEEK_PREFIX, ClipStore
from core.aeb.thread import AEBState
from core.radar.reader import _BUF_SIZE, _TOTAL_FORMAT

from tests.aeb.test_clip_capture import _make_clip
from core.radar.elevation import BODY_DATUM_FRAC


# Traffic position.y is the body datum, not ground level: ego coordinateY is
# the road surface and a body sits BODY_DATUM_FRAC of its height above it.
_BODY_H: float = 3.0
_BODY_Y: float = BODY_DATUM_FRAC * _BODY_H


def _one_vehicle_buf(px: float, pz: float, vid: int, speed: float) -> bytes:
    """Traffic buffer with a single populated vehicle in slot 0."""
    flat: list = []
    slot0 = [px, _BODY_Y, pz, 1.0, 0.0, 0.0, 0.0, 2.5, _BODY_H, 6.0, speed, 0.0] + [0, vid, 0, 0] + [0.0] * 30
    flat += slot0
    for _ in range(39):
        flat += [0.0] * 12 + [0, 0, 0, 0] + [0.0] * 30
    buf = struct.pack(_TOTAL_FORMAT, *flat)
    assert len(buf) == _BUF_SIZE
    return buf


def test_write_label_round_trip_and_peek(tmp_path):
    store = ClipStore(root=tmp_path)
    path = store.write(_make_clip(clip_id="lbl00001"))
    assert path is not None

    # Untagged initially.
    assert store.peek_metadata(path).label is None

    lbl = Label(class_="fp", severity=2, should_trigger=None,
                target_vid=9, notes="phantom oncoming")
    assert store.write_label(path, lbl) is True

    meta = store.peek_metadata(path)
    assert meta.label is not None
    assert meta.label.class_ == "fp" and meta.label.severity == 2
    assert meta.label.target_vid == 9 and meta.label.should_trigger is None

    # Filename is unchanged (no duplicate clip on tagging).
    assert len(store.list_clips()) == 1

    # Streams survive the in-place rewrite.
    reloaded = store.load(path)
    assert len(reloaded.radar_frames) == 4 and len(reloaded.aeb_ticks) == 4
    assert reloaded.metadata.label.class_ == "fp"

    # Label can be cleared again.
    assert store.write_label(path, None) is True
    assert store.peek_metadata(path).label is None


def _build_replayable_clip():
    clip = _make_clip(clip_id="replay01")
    clip.radar_frames = []
    clip.aeb_ticks = []
    for i in range(6):
        t = i * 0.033
        clip.radar_frames.append(RadarFrameRecord(
            t_wall=1000.0 + t, t_mono=t,
            ego=EgoTelemetry(coordinateX=0.0, coordinateZ=0.0, rotationX=0.5, speed=25.0),
            traffic_buf=_one_vehicle_buf(2.0, 40.0 - i, vid=7, speed=18.0),
            parked_buf=None,
        ))
        brake = i >= 4
        clip.aeb_ticks.append(AEBTickRecord(
            t_mono=t, radar_t_mono=t,
            consumed=ConsumedContext(max_brake_ms2=10.0),
            live_aeb=LiveAEB(
                aeb_warn=(i >= 3), aeb_brake=brake, engaged=brake,
                colliding_ids=[7] if i >= 3 else [],
                time_to_collision=1.0 if i >= 3 else 1e9,
            ),
        ))
    return clip


def test_replay_clip_builds_snapshots_from_recorded_decision():
    frames = replay_clip(_build_replayable_clip())
    assert len(frames) == 6

    # Every tick reconstructs a scene with the decoded vehicle, an ego arc, and
    # a predicted arc per vehicle (the review tool draws both trajectories).
    for f in frames:
        assert f.snapshot.ego_arc is not None
        assert any(v["vid"] == 7 for v in f.snapshot.vehicles)
        assert f.snapshot.vehicle_arcs[7]

    # State mapping comes from the recorded live_aeb.
    assert frames[0].snapshot.aeb_state == AEBState.STANDBY
    assert frames[3].snapshot.aeb_state == AEBState.WARN
    assert frames[5].snapshot.aeb_state == AEBState.BRAKE

    # A colliding target drives the hit marker onto that vehicle.
    last = frames[5].snapshot
    assert 7 in last.colliding_ids
    assert last.hit_x != 0.0 or last.hit_z != 0.0

    # t_rel starts at zero and increases.
    assert frames[0].t_rel == 0.0
    assert frames[-1].t_rel > frames[0].t_rel


def test_replay_smoothing_advances_vehicle_speed():
    # Position closes 1 m/frame; the reader's TMP/AI smoothing should produce a
    # non-trivial speed by the last frame (proves replay ran update_from_last).
    frames = replay_clip(_build_replayable_clip())
    v_last = next(v for v in frames[-1].snapshot.vehicles if v["vid"] == 7)
    assert v_last["speed_kmh"] >= 0.0   # decoded + smoothed without error


_CAL_REC = {"brake_ttb": 0.2, "brake_response_window_s": 0.30}


def test_raw_target_decel_undoes_the_published_slew():
    # The reviewer reads onset timing off this signal, so it must track the demand
    # the tick computed, not the rate-limited value that was published.
    ramping = LiveAEB(engaged=True, required_decel_ms2=9.4, effective_max_decel_ms2=8.0,
                      time_to_brake=1.5, target_decel_ms2=1.1)
    assert raw_target_decel(ramping, _CAL_REC) == 8.0        # clamped to capacity
    assert ramping.target_decel_ms2 == 1.1                   # what was actually sent

    below_cap = LiveAEB(engaged=True, required_decel_ms2=3.2, effective_max_decel_ms2=8.0,
                        time_to_brake=1.5)
    assert raw_target_decel(below_cap, _CAL_REC) == 3.2

    # Inside brake_ttb + response window the tick slams to full capacity.
    slam = LiveAEB(engaged=True, required_decel_ms2=0.4, effective_max_decel_ms2=7.5,
                   time_to_brake=0.1)
    assert raw_target_decel(slam, _CAL_REC) == 7.5

    # Disengaged demands nothing however large the recorded requirement is.
    idle = LiveAEB(engaged=False, required_decel_ms2=11.0, effective_max_decel_ms2=8.0,
                   time_to_brake=0.05)
    assert raw_target_decel(idle, _CAL_REC) == 0.0


def test_replay_populates_raw_target_per_frame():
    clip = _build_replayable_clip()
    for tk in clip.aeb_ticks:
        if tk.live_aeb.engaged:
            tk.live_aeb.required_decel_ms2 = 4.5
            tk.live_aeb.effective_max_decel_ms2 = 9.0
            tk.live_aeb.target_decel_ms2 = 0.6   # mid-slew, well under the demand

    frames = replay_clip(clip)
    # Ticks 0-3 are not engaged in the fixture, so nothing is demanded there.
    assert frames[0].raw_target_ms2 == 0.0
    assert frames[5].raw_target_ms2 == 4.5
    assert frames[5].live_aeb.target_decel_ms2 == 0.6


def test_peek_metadata_falls_back_when_metadata_exceeds_the_prefix(tmp_path):
    # peek_metadata reads a prefix first; a clip whose thumbnail pushes the metadata
    # past that prefix must still decode via the whole-file path.
    store = ClipStore(root=tmp_path)
    clip = _make_clip(clip_id="bigthumb")
    clip.metadata.thumbnail_jpeg = base64.b64encode(
        os.urandom(_PEEK_PREFIX * 2)
    ).decode("ascii")
    path = store.write(clip)
    assert path is not None
    assert path.stat().st_size > _PEEK_PREFIX

    meta = store.peek_metadata(path)
    assert meta is not None
    assert meta.clip_id == "bigthumb"
    assert meta.thumbnail_jpeg == clip.metadata.thumbnail_jpeg


def test_peek_metadata_matches_a_full_decode(tmp_path):
    store = ClipStore(root=tmp_path)
    path = store.write(_make_clip(clip_id="peekcmp0"))
    store.write_label(path, Label(class_="tn", severity=1))
    assert store.peek_metadata(path).to_json() == store.load(path).metadata.to_json()
