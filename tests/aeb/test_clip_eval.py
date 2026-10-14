"""Tests for headless AEB replay + parity + candidate calibration diff."""

from __future__ import annotations

import struct
from dataclasses import replace

from core.aeb.calibration import DEFAULT
from core.aeb.clip_eval import (
    TRIGGER_SLACK_S, EvalTick, diff_calibrations, outcome_under, parity_report,
    run_headless,
)
from core.aeb.clip_schema import (
    AEBTickRecord, Clip, ClipMetadata, ConsumedContext, EgoTelemetry, Label,
    LiveAEB, RadarFrameRecord,
)
from core.radar.reader import _BUF_SIZE, _TOTAL_FORMAT
from core.radar.elevation import BODY_DATUM_FRAC


# Traffic position.y is the body datum, not ground level: ego coordinateY is
# the road surface and a body sits BODY_DATUM_FRAC of its height above it.
_BODY_H: float = 3.0
_BODY_Y: float = BODY_DATUM_FRAC * _BODY_H


def _stopped_vehicle_buf(px: float, pz: float, vid: int) -> bytes:
    flat: list = []
    flat += [px, _BODY_Y, pz, 1.0, 0.0, 0.0, 0.0, 2.5, _BODY_H, 6.0, 0.0, 0.0] + [0, vid, 0, 0] + [0.0] * 30
    for _ in range(39):
        flat += [0.0] * 12 + [0, 0, 0, 0] + [0.0] * 30
    buf = struct.pack(_TOTAL_FORMAT, *flat)
    assert len(buf) == _BUF_SIZE
    return buf


def _collision_clip(n: int = 90, hz: float = 30.0, px: float = 0.0) -> Clip:
    """Ego at 20 m/s closing on a stopped vehicle 45 m ahead (+z), ``px`` m to the side."""
    dt = 1.0 / hz
    meta = ClipMetadata.create(trigger_source="auto_engagement", session_kind="SP")
    frames, ticks = [], []
    for i in range(n):
        t = i * dt
        ego_z = 20.0 * t
        frames.append(RadarFrameRecord(
            t_wall=1000.0 + t, t_mono=t,
            ego=EgoTelemetry(coordinateX=0.0, coordinateZ=ego_z, rotationX=0.5,
                             rotationY=0.0, speed=20.0),
            traffic_buf=_stopped_vehicle_buf(px, 45.0, vid=3),
            parked_buf=None,
        ))
        ticks.append(AEBTickRecord(
            t_mono=t, radar_t_mono=t,
            consumed=ConsumedContext(max_brake_ms2=10.0, aeb_enabled=True),
            live_aeb=LiveAEB(),
        ))
    return Clip(metadata=meta, radar_frames=frames, aeb_ticks=ticks)


def test_headless_replay_runs_and_engages():
    clip = _collision_clip()
    ev = run_headless(clip)
    assert len(ev) == len(clip.aeb_ticks)
    # A stopped obstacle dead ahead must produce an engagement somewhere.
    assert any(e.aeb_brake for e in ev), "expected AEB to engage on stopped lead"
    assert any(3 in e.colliding_ids for e in ev)


def test_run_headless_deterministic():
    clip = _collision_clip()
    a = run_headless(clip)
    b = run_headless(clip)
    assert [e.aeb_brake for e in a] == [e.aeb_brake for e in b]
    assert [e.target_decel_ms2 for e in a] == [e.target_decel_ms2 for e in b]


def test_parity_matches_when_recorded_equals_replay():
    # Seed each tick's "recorded" decision from a headless run, then parity must
    # be exact: this exercises warm-state apply + the comparison plumbing.
    clip = _collision_clip()
    ev = run_headless(clip)
    for tk, e in zip(sorted(clip.aeb_ticks, key=lambda x: x.t_mono), ev):
        tk.live_aeb = LiveAEB(
            aeb_warn=e.aeb_warn, aeb_brake=e.aeb_brake, engaged=e.engaged,
            colliding_ids=sorted(e.colliding_ids),
        )
    rep = parity_report(clip, burn_in_s=0.0)
    assert rep.mismatches == []
    assert rep.warn_agreement == 1.0 and rep.brake_agreement == 1.0
    assert rep.compared == len(clip.aeb_ticks)


def test_diff_calibrations_flags_removed_engagement():
    clip = _collision_clip()
    baseline = run_headless(clip, cal=DEFAULT)
    assert any(e.aeb_brake for e in baseline)   # baseline engages

    # A candidate whose min engage speed is above ego speed can never start a
    # new engagement, so the brake vanishes entirely.
    candidate = replace(DEFAULT, aeb_min_engage_speed_kmh=1000.0)
    cand = run_headless(clip, cal=candidate)
    assert not any(e.aeb_brake for e in cand)

    diffs = diff_calibrations(clip, candidate)
    assert diffs, "expected diff where baseline brakes and candidate does not"
    assert any(d.baseline_brake and not d.candidate_brake for d in diffs)


def test_outcome_true_positive_and_false_negative():
    clip = _collision_clip()
    bt = [e.t_rel for e in run_headless(clip) if e.aeb_brake]
    clip.metadata.label = Label(class_="tp", severity=4,
                                should_trigger={"from_t": bt[0], "to_t": bt[-1]})
    assert outcome_under(clip).verdict == "true_positive"
    off = replace(DEFAULT, aeb_min_engage_speed_kmh=1000.0)
    assert outcome_under(clip, off).verdict == "false_negative"


def test_outcome_false_positive_and_true_negative():
    clip = _collision_clip()
    clip.metadata.label = Label(class_="fp", severity=2, should_trigger=None)
    assert outcome_under(clip).verdict == "false_positive"
    off = replace(DEFAULT, aeb_min_engage_speed_kmh=1000.0)
    assert outcome_under(clip, off).verdict == "true_negative"


def test_outcome_false_warn_when_warn_without_brake(monkeypatch):
    clip = _collision_clip()
    clip.metadata.label = Label(class_="fp", severity=2, should_trigger=None)

    class _Tick:
        def __init__(self, t_rel, warn, brake):
            self.t_rel = t_rel
            self.aeb_warn = warn
            self.aeb_brake = brake
            self.target_decel_ms2 = 0.0

    monkeypatch.setattr(
        "core.aeb.clip_eval.run_headless",
        lambda *a, **k: [_Tick(0.0, True, False), _Tick(0.5, True, False)],
    )
    o = outcome_under(clip, burn_in_s=0.0)
    assert o.verdict == "false_warn"
    assert o.warn_window == (0.0, 0.5)
    assert o.brake_window is None
    assert not o.engaged


# The engage bar can never be met and the TTB slam never fires: no brake at all.
_NO_BAR = replace(DEFAULT, aeb_engage_frac=1e4, aeb_engage_frac_certain=1e4, brake_ttb=-1e3)


def _labelled_tp(clip: Clip) -> Clip:
    bt = [e.t_rel for e in run_headless(clip) if e.aeb_brake]
    clip.metadata.label = Label(class_="tp", severity=4,
                                should_trigger={"from_t": bt[0], "to_t": bt[-1]})
    return clip


def test_engage_bar_cannot_turn_a_geometric_hit_into_a_miss():
    clip = _labelled_tp(_collision_clip())
    assert not any(e.aeb_brake for e in run_headless(clip, cal=_NO_BAR))
    o = outcome_under(clip, _NO_BAR)
    assert o.verdict == "true_positive"
    assert not o.braked_in_window and not o.engaged


def test_target_off_the_path_is_a_geometric_miss():
    clip = _collision_clip(px=8.0)
    assert not any(e.geom_engage or e.aeb_brake for e in run_headless(clip))
    clip.metadata.label = Label(class_="fn", severity=3,
                                should_trigger={"from_t": 0.5, "to_t": 2.5})
    assert outcome_under(clip).verdict == "false_negative"


def _tick(t: float, brake: bool = False, geom: bool = False) -> EvalTick:
    return EvalTick(t_rel=t, t_mono=t, aeb_warn=brake, aeb_brake=brake, engaged=brake,
                    target_decel_ms2=0.0, required_decel_ms2=0.0, time_to_brake=1e9,
                    time_to_collision=1e9, geom_engage=geom)


def _outcome(monkeypatch, ticks, window=(4.0, 5.0)):
    clip = _collision_clip(n=2)
    clip.metadata.label = Label(class_="tp", severity=3,
                                should_trigger={"from_t": window[0], "to_t": window[1]})
    monkeypatch.setattr("core.aeb.clip_eval.run_headless", lambda *a, **k: ticks)
    return outcome_under(clip, burn_in_s=0.0)


def test_geometry_inside_the_window_is_a_hit_and_outside_is_late_or_miss(monkeypatch):
    inside = _outcome(monkeypatch, [_tick(3.0), _tick(4.5, geom=True), _tick(6.0)])
    assert inside.verdict == "true_positive" and not inside.braked_in_window
    late = _outcome(monkeypatch, [_tick(4.5), _tick(5.5, geom=True)])
    assert late.verdict == "late"
    early = _outcome(monkeypatch, [_tick(3.5, geom=True), _tick(4.5)])
    assert early.verdict == "false_negative"


def test_brake_held_into_the_window_is_a_hit(monkeypatch):
    o = _outcome(monkeypatch, [_tick(3.8, brake=True, geom=True), _tick(4.2, brake=True)])
    assert o.verdict == "true_positive" and o.braked_in_window
    assert o.stray_onsets == ()


def test_only_onsets_past_the_slack_are_stray(monkeypatch):
    lo, hi = 4.0 - TRIGGER_SLACK_S, 5.0 + TRIGGER_SLACK_S
    ticks = [
        _tick(lo - 0.5, brake=True), _tick(lo - 0.4),
        _tick(lo + 0.5, brake=True), _tick(lo + 0.6),
        _tick(4.5, brake=True, geom=True), _tick(4.6),
        _tick(hi - 0.5, brake=True), _tick(hi - 0.4),
        _tick(hi + 0.5, brake=True),
    ]
    o = _outcome(monkeypatch, ticks)
    assert o.verdict == "true_positive"
    assert o.stray_onsets == (lo - 0.5, hi + 0.5)
