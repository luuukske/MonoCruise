"""Tests for boundary-negative (TN) and crash clip capture samplers.

Three layers:
- recorder policy: background sources (``shadow_near`` / ``random``) are
  auto-tagged true negatives, throttled on their own long timer, and skip the
  context thumbnail; real events (``auto_engagement``, ``auto_crash``, ``manual``)
  keep their unlabelled, thumbnailed behaviour.
- thread TN condition: ``_capture_aeb_tick`` requests a ``shadow_near`` capture
  when AEB stayed silent while a nearby filtered vehicle lies inside ego's
  predicted corridor, and stays quiet otherwise (including while user braking).
- thread crash condition: ``_capture_aeb_tick`` requests an ``auto_crash``
  capture on a sudden ego speed drop above the minimum speed floor.
"""

from __future__ import annotations

import core.aeb.thread as aeb_thread
from core.aeb.calibration import DEFAULT as CAL
from core.aeb.filters import FilterResult
from core.aeb.thread import AEBSnapshot, AEBState, AEBThread, _INF
from core.radar.traffic import build_arc

from tests.aeb.test_clip_capture import _StubWriter, _drive
from core.aeb.recorder import AEBClipRecorder


def _veh(vid: int, x: float = 0.0, z: float = -30.0, half_w: float = 1.15) -> dict:
    return {"vid": vid, "x": x, "z": z, "half_w": half_w}


def _tn_snap(**kw) -> AEBSnapshot:
    """Snapshot with a filtered vehicle ahead inside ego's corridor."""
    defaults = dict(
        ego_x=0.0,
        ego_z=0.0,
        ego_yaw=0.0,
        ego_speed=20.0,
        ego_half_w=1.15,
        time_to_collision=_INF,
        aeb_state=AEBState.STANDBY,
        suppressed_ids={5},
        vehicles=[_veh(5)],
        suppression_reasons={
            5: [FilterResult(suppressed=True, reason="SweepPassFilter")],
        },
    )
    merged = {**defaults, **kw}
    if merged.get("ego_arc") is None:
        merged["ego_arc"] = build_arc(
            merged["ego_x"], merged["ego_z"], merged["ego_yaw"],
            max(float(merged["ego_speed"]), 2.0), 0.0,
            float(merged["ego_half_w"]), 10.0,
        )
    return AEBSnapshot(**merged)


def _finalize(rec: AEBClipRecorder, t0: float, t1: float) -> None:
    _drive(rec, t0, t1)


def test_shadow_near_auto_tagged_true_negative():
    writer = _StubWriter()
    rec = AEBClipRecorder(writer, pre_s=2.0, post_s=1.0)
    _drive(rec, 0.0, 3.0)
    assert rec.trigger("shadow_near", calibration=CAL) == "started"
    _drive(rec, 3.0 + 1 / 30, 4.2)

    lbl = writer.clips[0].metadata.label
    assert lbl is not None
    assert lbl.class_ == "tn"
    assert lbl.should_trigger is None            # must-not-trigger negative
    assert "auto" in lbl.notes.lower()


def test_random_auto_tagged_true_negative():
    writer = _StubWriter()
    rec = AEBClipRecorder(writer, pre_s=2.0, post_s=1.0)
    _drive(rec, 0.0, 3.0)
    rec.trigger("random", calibration=CAL)
    _drive(rec, 3.0 + 1 / 30, 4.2)
    assert writer.clips[0].metadata.label.class_ == "tn"


def test_real_events_are_not_auto_tagged():
    for source in ("auto_engagement", "auto_crash", "manual"):
        writer = _StubWriter()
        rec = AEBClipRecorder(writer, pre_s=2.0, post_s=1.0)
        _drive(rec, 0.0, 3.0)
        rec.trigger(source, calibration=CAL)
        _drive(rec, 3.0 + 1 / 30, 4.2)
        assert writer.clips[0].metadata.label is None, source


def test_shadow_near_skips_thumbnail_real_event_keeps_it():
    # Same provider wired for both; only the real event grabs a thumbnail.
    shadow_w = _StubWriter()
    shadow = AEBClipRecorder(shadow_w, pre_s=2.0, post_s=1.0,
                             screenshot_provider=lambda: "THUMB64")
    _drive(shadow, 0.0, 3.0)
    shadow.trigger("shadow_near", calibration=CAL)
    _drive(shadow, 3.0 + 1 / 30, 4.2)
    assert shadow_w.clips[0].metadata.thumbnail_jpeg is None

    manual_w = _StubWriter()
    manual = AEBClipRecorder(manual_w, pre_s=2.0, post_s=1.0,
                             screenshot_provider=lambda: "THUMB64")
    _drive(manual, 0.0, 3.0)
    manual.trigger("manual")
    _drive(manual, 3.0 + 1 / 30, 4.2)
    assert manual_w.clips[0].metadata.thumbnail_jpeg == "THUMB64"


def test_tn_cooldown_throttles_shadow_near_only():
    writer = _StubWriter()
    rec = AEBClipRecorder(writer, pre_s=2.0, post_s=1.0,
                          cooldown_s=0.0, tn_cooldown_s=10.0)
    _drive(rec, 0.0, 3.0)
    assert rec.trigger("shadow_near", calibration=CAL) == "started"
    _drive(rec, 3.0 + 1 / 30, 4.2)                 # finalize; arms tn cooldown

    # Inside the TN cooldown: another background sample is throttled, but a real
    # event (which does not share the timer) is honoured.
    assert rec.trigger("shadow_near", calibration=CAL) == "ignored"
    assert rec.trigger("manual") == "started"
    _drive(rec, 4.2 + 1 / 30, 5.4)                 # let the manual capture finalize

    # Past the TN cooldown window: background sampling resumes.
    _drive(rec, 5.4 + 1 / 30, 16.0)
    assert rec.trigger("shadow_near", calibration=CAL) == "started"


def test_auto_crash_keeps_thumbnail():
    writer = _StubWriter()
    rec = AEBClipRecorder(writer, pre_s=2.0, post_s=1.0,
                         screenshot_provider=lambda: "THUMB64")
    _drive(rec, 0.0, 3.0)
    rec.trigger("auto_crash", calibration=CAL)
    _drive(rec, 3.0 + 1 / 30, 4.2)
    assert writer.clips[0].metadata.thumbnail_jpeg == "THUMB64"


def test_auto_crash_wins_priority_over_engagement_when_folded():
    writer = _StubWriter()
    rec = AEBClipRecorder(writer, pre_s=2.0, post_s=1.0)
    _drive(rec, 0.0, 3.0)
    assert rec.trigger("auto_engagement", calibration=CAL) == "started"
    assert rec.trigger("auto_crash", calibration=CAL) == "folded"
    _drive(rec, 3.0 + 1 / 30, 4.2)

    m = writer.clips[0].metadata
    assert m.trigger_source == "auto_crash"
    assert "auto_engagement" in m.also_triggered
    assert m.label is None


def test_shadow_near_folding_up_drops_the_tn_label():
    # A background capture that a real engagement folds into is a real event:
    # the winning source decides the label, so no auto TN tag survives.
    writer = _StubWriter()
    rec = AEBClipRecorder(writer, pre_s=2.0, post_s=1.0)
    _drive(rec, 0.0, 3.0)
    assert rec.trigger("shadow_near", calibration=CAL) == "started"
    assert rec.trigger("auto_engagement", calibration=CAL) == "folded"
    _drive(rec, 3.0 + 1 / 30, 4.2)

    m = writer.clips[0].metadata
    assert m.trigger_source == "auto_engagement"
    assert "shadow_near" in m.also_triggered
    assert m.label is None


class _RecorderSpy:
    """Stands in for the process recorder; records trigger sources."""

    def __init__(self) -> None:
        self.triggers: list[str] = []

    def push_aeb_tick(self, tick, warm_state=None) -> None:
        pass

    def trigger(self, source, **kw) -> str:
        self.triggers.append(source)
        return "started"


def _tick(thread: AEBThread, snap: AEBSnapshot, new_state: AEBState) -> _RecorderSpy:
    spy = _RecorderSpy()
    orig = aeb_thread.get_recorder
    aeb_thread.get_recorder = lambda: spy
    try:
        thread._capture_aeb_tick(
            now_mono=100.0, radar_t_mono=100.0, aeb_active=True, snap=snap,
            max_brake_ms2=7.8, required_decel=0.0, effective_max_decel=7.8,
            target_published=0.0, ff_decel=0.0,
            aeb_warn=(new_state != AEBState.STANDBY),
            aeb_brake=(new_state == AEBState.BRAKE),
            new_state=new_state, tmp_traffic_session=False,
        )
    finally:
        aeb_thread.get_recorder = orig
    return spy


def test_thread_fires_shadow_on_filtered_candidate_while_silent():
    t = AEBThread()
    t._engaged = False
    spy = _tick(t, _tn_snap(), AEBState.STANDBY)
    assert "shadow_near" in spy.triggers


def test_thread_no_shadow_while_user_braking():
    t = AEBThread()
    t._engaged = False
    t._read_user_braking = lambda: True
    spy = _tick(t, _tn_snap(), AEBState.STANDBY)
    assert "shadow_near" not in spy.triggers


def test_thread_fires_auto_crash_on_speed_drop():
    t = AEBThread()
    t._prev_ego_speed_capture_ms = 50.0 / 3.6
    snap = _tn_snap(ego_speed=(50.0 - 6.0) / 3.6)
    spy = _tick(t, snap, AEBState.STANDBY)
    assert "auto_crash" in spy.triggers


def test_thread_no_auto_crash_on_gentle_decel():
    t = AEBThread()
    t._prev_ego_speed_capture_ms = 50.0 / 3.6
    snap = _tn_snap(ego_speed=(50.0 - 2.0) / 3.6)
    spy = _tick(t, snap, AEBState.STANDBY)
    assert "auto_crash" not in spy.triggers


def test_thread_no_auto_crash_below_min_speed():
    t = AEBThread()
    t._prev_ego_speed_capture_ms = 35.0 / 3.6
    snap = _tn_snap(ego_speed=5.0 / 3.6)
    spy = _tick(t, snap, AEBState.STANDBY)
    assert "auto_crash" not in spy.triggers


def test_thread_no_auto_crash_without_prior_tick():
    t = AEBThread()
    spy = _tick(t, _tn_snap(ego_speed=20.0), AEBState.STANDBY)
    assert "auto_crash" not in spy.triggers


def test_thread_no_shadow_on_distant_filtered_vehicle():
    t = AEBThread()
    t._engaged = False
    snap = _tn_snap(vehicles=[_veh(5, x=0.0, z=-150.0)])
    spy = _tick(t, snap, AEBState.STANDBY)
    assert spy.triggers == []


def test_thread_no_shadow_when_filtered_vehicle_outside_ego_corridor():
    t = AEBThread()
    t._engaged = False
    snap = _tn_snap(vehicles=[_veh(5, x=20.0, z=-30.0)])
    spy = _tick(t, snap, AEBState.STANDBY)
    assert spy.triggers == []


def test_thread_no_shadow_on_non_spatial_filter_only():
    t = AEBThread()
    t._engaged = False
    snap = _tn_snap(
        suppression_reasons={
            5: [FilterResult(suppressed=True, reason="TmpRelSpeedFilter")],
        },
    )
    spy = _tick(t, snap, AEBState.STANDBY)
    assert spy.triggers == []


def test_thread_no_shadow_when_filtered_id_missing_from_vehicles():
    t = AEBThread()
    t._engaged = False
    snap = _tn_snap(vehicles=[])
    spy = _tick(t, snap, AEBState.STANDBY)
    assert spy.triggers == []


def test_thread_no_shadow_on_near_miss_without_filter():
    t = AEBThread()
    t._engaged = False
    snap = AEBSnapshot(ego_speed=20.0, time_to_collision=2.0,
                       aeb_state=AEBState.STANDBY)
    spy = _tick(t, snap, AEBState.STANDBY)
    assert spy.triggers == []


def test_thread_no_shadow_when_nothing_in_play():
    t = AEBThread()
    t._engaged = False
    snap = AEBSnapshot(ego_speed=20.0, time_to_collision=_INF,
                       aeb_state=AEBState.STANDBY)
    spy = _tick(t, snap, AEBState.STANDBY)
    assert spy.triggers == []


def test_thread_no_shadow_while_crawling():
    t = AEBThread()
    t._engaged = False
    snap = _tn_snap(ego_speed=0.5, time_to_collision=1.0)
    spy = _tick(t, snap, AEBState.STANDBY)
    assert spy.triggers == []


def test_thread_no_shadow_when_warning():
    # A real WARN entry captures as auto_engagement, never as a background TN.
    t = AEBThread()
    t._engaged = False
    t._capture_prev_state = AEBState.STANDBY
    snap = _tn_snap(time_to_collision=1.0, aeb_state=AEBState.WARN)
    spy = _tick(t, snap, AEBState.WARN)
    assert "shadow_near" not in spy.triggers
    assert "auto_engagement" in spy.triggers
