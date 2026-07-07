"""Tests for the AEB clip capture-and-save path: schema, store, recorder.

Exercises serialization round-trip, on-disk rotation, the background writer, and
the recorder's retroactive freeze / trigger-priority / cooldown logic without a
game, shared memory, or the live threads.
"""

from __future__ import annotations

import os

import pytest

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.clip_schema import (
    AEBTickRecord,
    AEBWarmState,
    Clip,
    ClipMetadata,
    ConsumedContext,
    EgoTelemetry,
    Label,
    LiveAEB,
    RadarFrameRecord,
)
from core.aeb.clip_store import (
    AsyncClipWriter,
    ClipStore,
    deserialize_clip,
    serialize_clip,
)
from core.aeb.recorder import AEBClipRecorder, calibration_fingerprint


def _make_clip(clip_id: str = "test0001", trigger: str = "manual") -> Clip:
    cid, cfields = calibration_fingerprint(CAL)
    meta = ClipMetadata.create(
        clip_id=clip_id,
        trigger_source=trigger,
        session_kind="TMP",
        calibration_id=cid,
        calibration=cfields,
        aeb_warm_state=AEBWarmState(engaged=True, latched_threat_ids=[7, 9],
                                    latched_filter_ego_kmh=52.0, target_decel_ms2=3.5),
    )
    frames = [
        RadarFrameRecord(
            t_wall=1000.0 + i * 0.033,
            t_mono=100.0 + i * 0.033,
            ego=EgoTelemetry(coordinateX=float(i), speed=25.0, rotationX=0.5),
            traffic_buf=bytes([i % 256]) * 6960,
            parked_buf=None if i % 2 else bytes([i % 256]) * 1720,
        )
        for i in range(4)
    ]
    ticks = [
        AEBTickRecord(
            t_mono=100.0 + i * 0.033,
            radar_t_mono=100.0 + i * 0.033,
            consumed=ConsumedContext(max_brake_ms2=7.8, brakeval=0.1 * i),
            live_aeb=LiveAEB(
                aeb_warn=bool(i), engaged=bool(i), colliding_ids=[7],
                suppression_reasons={"9": ["OppositeLaneFilter"]},
            ),
        )
        for i in range(4)
    ]
    return Clip(metadata=meta, radar_frames=frames, aeb_ticks=ticks)


def test_serialization_round_trip_preserves_streams_and_bytes():
    clip = _make_clip()
    blob = serialize_clip(clip)
    back = deserialize_clip(blob)

    assert back.metadata.clip_id == clip.metadata.clip_id
    assert back.metadata.trigger_source == "manual"
    assert back.metadata.session_kind == "TMP"
    assert back.metadata.label is None
    assert back.metadata.aeb_warm_state.latched_threat_ids == [7, 9]
    assert back.metadata.aeb_warm_state.latched_filter_ego_kmh == 52.0

    assert len(back.radar_frames) == 4
    assert len(back.aeb_ticks) == 4
    # Raw buffers survive base64 exactly, including the nullable parked buffer.
    assert back.radar_frames[0].traffic_buf == clip.radar_frames[0].traffic_buf
    assert back.radar_frames[1].parked_buf is None
    assert back.radar_frames[0].parked_buf == clip.radar_frames[0].parked_buf
    assert back.aeb_ticks[0].live_aeb.suppression_reasons == {"9": ["OppositeLaneFilter"]}
    assert back.aeb_ticks[1].live_aeb.colliding_ids == [7]


def test_top_level_json_shape_matches_payload_contract():
    d = _make_clip().to_json_dict()
    # Metadata fields sit at the top level; streams are arrays beside them.
    assert d["schema_version"] == 1
    assert "clip_id" in d and "calibration" in d
    assert d["label"] is None
    assert isinstance(d["radar_frames"], list) and isinstance(d["aeb_ticks"], list)
    assert d["radar_frames"][0]["traffic_buf"] is not None


def test_label_round_trip_uses_class_key():
    lbl = Label(class_="fp", severity=4, should_trigger=None,
                target_vid=7, notes="phantom on oncoming")
    d = lbl.to_json()
    assert d["class"] == "fp"          # python keyword mapped to JSON "class"
    assert "class_" not in d
    assert Label.from_json(d).class_ == "fp"


def test_store_write_load_and_count_rotation(tmp_path):
    store = ClipStore(root=tmp_path, max_clips=3)
    for i in range(5):
        path = store.write(_make_clip(clip_id=f"clip{i:04d}"))
        assert path is not None
        # Force deterministic ordering so the newest three are the survivors.
        os.utime(path, (100 + i, 100 + i))

    infos = store.list_clips()
    assert len(infos) == 3
    survivors = {store.load(info.path).metadata.clip_id for info in infos}
    assert survivors == {"clip0002", "clip0003", "clip0004"}


def test_async_writer_persists_and_notifies(tmp_path):
    saved_names: list[str] = []
    store = ClipStore(root=tmp_path)
    writer = AsyncClipWriter(store, notify=saved_names.append)
    writer.start()
    try:
        assert writer.submit(_make_clip(clip_id="async001")) is True
    finally:
        writer.stop()

    infos = store.list_clips()
    assert len(infos) == 1
    assert store.load(infos[0].path).metadata.clip_id == "async001"
    assert saved_names == [infos[0].name]


class _StubWriter:
    """Captures submitted clips without touching disk or a thread."""

    def __init__(self) -> None:
        self.clips: list[Clip] = []

    def submit(self, clip: Clip) -> bool:
        self.clips.append(clip)
        return True


def _drive(recorder: AEBClipRecorder, t0: float, t1: float, hz: float = 30.0,
           warm_from=lambda t: AEBWarmState()) -> None:
    dt = 1.0 / hz
    t = t0
    while t <= t1 + 1e-9:
        recorder.push_radar_frame(RadarFrameRecord(t_wall=t, t_mono=t, ego=EgoTelemetry(speed=20.0)))
        recorder.push_aeb_tick(
            AEBTickRecord(t_mono=t, radar_t_mono=t, live_aeb=LiveAEB()),
            warm_state=warm_from(t),
        )
        t += dt


def test_recorder_freezes_window_after_post_roll():
    writer = _StubWriter()
    rec = AEBClipRecorder(writer, pre_s=8.0, post_s=3.0, burn_in_s=2.0)

    _drive(rec, 0.0, 10.0)                       # fill the pre-roll
    assert rec.trigger("auto_engagement", session_kind="TMP", calibration=CAL) == "started"
    assert writer.clips == []                     # nothing until the post-roll elapses
    _drive(rec, 10.0 + 1 / 30, 13.2)             # run past trigger_t + post_s

    assert len(writer.clips) == 1
    clip = writer.clips[0]
    m = clip.metadata
    assert m.trigger_source == "auto_engagement"
    assert m.session_kind == "TMP"
    assert m.calibration_id and "full_brake_decel" in m.calibration
    assert m.pre_s == 8.0 and m.post_s == 3.0 and m.burn_in_s == 2.0
    # Window is [trigger-8, trigger+3] = 11 s at 30 Hz.
    assert 300 <= m.frame_count <= 340
    assert m.frame_count == len(clip.radar_frames)
    assert m.tick_count == len(clip.aeb_ticks)


def test_recorder_selects_window_start_warm_state():
    writer = _StubWriter()
    rec = AEBClipRecorder(writer, pre_s=8.0, post_s=3.0)
    # Encode each tick's time into its warm state so we can identify the pick.
    _drive(rec, 0.0, 10.0, warm_from=lambda t: AEBWarmState(target_decel_ms2=t))
    rec.trigger("manual")
    _drive(rec, 10.0 + 1 / 30, 13.2, warm_from=lambda t: AEBWarmState(target_decel_ms2=t))

    ws = writer.clips[0].metadata.aeb_warm_state
    # Earliest in-window tick is at ~ trigger_t - pre_s = 2.0 s.
    assert ws.target_decel_ms2 == pytest.approx(2.0, abs=0.05)


def test_trigger_priority_folds_lower_into_also_triggered():
    writer = _StubWriter()
    rec = AEBClipRecorder(writer, pre_s=2.0, post_s=1.0)

    _drive(rec, 0.0, 3.0)
    assert rec.trigger("auto_engagement") == "started"
    assert rec.trigger("manual") == "folded"          # higher priority, same window
    assert rec.trigger("random") == "folded"
    _drive(rec, 3.0 + 1 / 30, 4.2)

    m = writer.clips[0].metadata
    assert m.trigger_source == "manual"
    assert set(m.also_triggered) == {"auto_engagement", "random"}


def test_auto_trigger_respects_cooldown_manual_bypasses():
    writer = _StubWriter()
    rec = AEBClipRecorder(writer, pre_s=2.0, post_s=1.0, cooldown_s=5.0)

    _drive(rec, 0.0, 3.0)
    rec.trigger("auto_engagement")
    _drive(rec, 3.0 + 1 / 30, 4.2)                     # completes the first capture
    assert len(writer.clips) == 1

    # Still inside the cooldown window: auto ignored, manual honoured.
    assert rec.trigger("auto_engagement") == "ignored"
    assert rec.trigger("manual") == "started"


def test_disabled_recorder_is_inert():
    writer = _StubWriter()
    rec = AEBClipRecorder(writer, enabled=False)
    _drive(rec, 0.0, 12.0)
    assert rec.trigger("manual") == "disabled"
    assert writer.clips == []
