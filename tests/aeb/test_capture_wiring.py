"""Wiring tests for the radar / AEB capture glue. Exercises the thread-side helpers that the game
would drive (`_capture_frame`, `_capture_aeb_tick`, `_build_warm_state`) with a stub recorder, so
the mapping from live state into the clip schema is covered without a game or the loops."""
from __future__ import annotations

import core.aeb.thread as aeb_thread_mod
from core.aeb.clip_schema import AEBTickRecord, RadarFrameRecord
from core.aeb.thread import AEBSnapshot, AEBState, AEBThread
from core.radar.thread import RadarThread


class _StubRecorder:
    def __init__(self) -> None:
        self.frames: list = []
        self.ticks: list = []
        self.triggers: list = []

    def push_radar_frame(self, frame) -> None:
        self.frames.append(frame)

    def push_aeb_tick(self, tick, warm_state=None) -> None:
        self.ticks.append((tick, warm_state))

    def trigger(self, source, **kw) -> str:
        self.triggers.append((source, kw))
        return "started"


def test_radar_capture_frame_maps_ego_and_buffers():
    rt = RadarThread()
    stub = _StubRecorder()
    rt._capture_frame(
        stub, now_mono=5.0, t_wall=1000.0,
        ego_x=1.0, ego_y=2.0, ego_z=3.0, ego_yaw_norm=0.5,
        ego_speed=25.0, ego_steer=0.1, ego_pitch_raw=0.0,
        ego_has_trailer=True, paused=False,
        traffic_buf=b"x" * 6960, parked_buf=None,
    )
    assert len(stub.frames) == 1
    f = stub.frames[0]
    assert isinstance(f, RadarFrameRecord)
    assert f.t_mono == 5.0 and f.t_wall == 1000.0
    assert f.ego.rotationX == 0.5 and f.ego.speed == 25.0 and f.ego.ego_has_trailer is True
    assert f.traffic_buf == b"x" * 6960 and f.parked_buf is None


def test_aeb_capture_tick_maps_decision_and_fires_trigger(monkeypatch):
    stub = _StubRecorder()
    monkeypatch.setattr(aeb_thread_mod, "get_recorder", lambda: stub)
    at = AEBThread()
    try:
        at._engaged = False
        at._capture_prev_state = AEBState.STANDBY
        snap = AEBSnapshot(
            colliding_ids={7}, suppressed_ids={9}, braking_worsens_ids=set(),
            suppression_reasons={9: []}, time_to_brake=0.2, time_to_collision=1.5,
        )
        at._capture_aeb_tick(
            now_mono=5.0, radar_t_mono=4.9, aeb_active=True, snap=snap,
            max_brake_ms2=11.0, required_decel=6.0, effective_max_decel=10.0,
            target_published=5.0, ff_decel=3.0, aeb_warn=True, aeb_brake=False,
            new_state=AEBState.WARN, tmp_traffic_session=True,
        )
        assert len(stub.ticks) == 1
        tick, warm = stub.ticks[0]
        assert isinstance(tick, AEBTickRecord)
        assert tick.radar_t_mono == 4.9
        assert tick.consumed.max_brake_ms2 == 11.0 and tick.consumed.aeb_enabled is True
        assert tick.live_aeb.aeb_warn is True and tick.live_aeb.colliding_ids == [7]
        assert warm.engaged is False

        # WARN entry from STANDBY fires an auto_engagement trigger.
        assert len(stub.triggers) == 1
        source, kw = stub.triggers[0]
        assert source == "auto_engagement" and kw["session_kind"] == "TMP"
        assert at._capture_prev_state == AEBState.WARN
    finally:
        at.teardown()


def test_aeb_capture_tick_no_trigger_without_state_entry(monkeypatch):
    stub = _StubRecorder()
    monkeypatch.setattr(aeb_thread_mod, "get_recorder", lambda: stub)
    at = AEBThread()
    try:
        at._capture_prev_state = AEBState.WARN
        snap = AEBSnapshot()
        at._capture_aeb_tick(
            now_mono=6.0, radar_t_mono=6.0, aeb_active=True, snap=snap,
            max_brake_ms2=11.0, required_decel=0.0, effective_max_decel=10.0,
            target_published=0.0, ff_decel=0.0, aeb_warn=True, aeb_brake=False,
            new_state=AEBState.WARN, tmp_traffic_session=False,
        )
        assert len(stub.ticks) == 1        # tick still recorded
        assert stub.triggers == []         # no WARN->WARN edge, no trigger
    finally:
        at.teardown()


def test_aeb_capture_tick_inert_when_recorder_disabled(monkeypatch):
    monkeypatch.setattr(aeb_thread_mod, "get_recorder", lambda: None)
    at = AEBThread()
    try:
        at._capture_aeb_tick(
            now_mono=1.0, radar_t_mono=1.0, aeb_active=False, snap=AEBSnapshot(),
            max_brake_ms2=11.0, required_decel=0.0, effective_max_decel=10.0,
            target_published=0.0, ff_decel=0.0, aeb_warn=False, aeb_brake=False,
            new_state=AEBState.STANDBY, tmp_traffic_session=False,
        )
        # No recorder: nothing to observe, and no exception is the point.
    finally:
        at.teardown()
