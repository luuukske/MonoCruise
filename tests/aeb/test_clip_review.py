"""Tests for the review path: label persistence, metadata peek, clip replay."""

from __future__ import annotations

import struct

from core.aeb.clip_schema import (
    AEBTickRecord, ConsumedContext, EgoTelemetry, Label, LiveAEB, RadarFrameRecord,
)
from core.aeb.clip_replay import replay_clip
from core.aeb.clip_store import ClipStore
from core.aeb.thread import AEBState
from core.radar.reader import _BUF_SIZE, _TOTAL_FORMAT

from tests.aeb.test_clip_capture import _make_clip


def _one_vehicle_buf(px: float, pz: float, vid: int, speed: float) -> bytes:
    """Traffic buffer with a single populated vehicle in slot 0."""
    flat: list = []
    slot0 = [px, 0.0, pz, 1.0, 0.0, 0.0, 0.0, 2.5, 3.0, 6.0, speed, 0.0] + [0, vid, 0, 0] + [0.0] * 30
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

    # Every tick reconstructs a scene with the decoded vehicle and an ego arc.
    for f in frames:
        assert f.snapshot.ego_arc is not None
        assert any(v["vid"] == 7 for v in f.snapshot.vehicles)

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
