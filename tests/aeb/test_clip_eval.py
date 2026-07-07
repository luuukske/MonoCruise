"""Tests for headless AEB replay + parity + candidate calibration diff."""

from __future__ import annotations

import struct
from dataclasses import replace

from core.aeb.calibration import DEFAULT
from core.aeb.clip_eval import (
    diff_calibrations, outcome_under, parity_report, run_headless,
)
from core.aeb.clip_schema import (
    AEBTickRecord, Clip, ClipMetadata, ConsumedContext, EgoTelemetry, Label,
    LiveAEB, RadarFrameRecord,
)
from core.radar.reader import _BUF_SIZE, _TOTAL_FORMAT


def _stopped_vehicle_buf(px: float, pz: float, vid: int) -> bytes:
    flat: list = []
    flat += [px, 0.0, pz, 1.0, 0.0, 0.0, 0.0, 2.5, 3.0, 6.0, 0.0, 0.0] + [0, vid, 0, 0] + [0.0] * 30
    for _ in range(39):
        flat += [0.0] * 12 + [0, 0, 0, 0] + [0.0] * 30
    buf = struct.pack(_TOTAL_FORMAT, *flat)
    assert len(buf) == _BUF_SIZE
    return buf


def _collision_clip(n: int = 90, hz: float = 30.0) -> Clip:
    """Ego at 20 m/s closing on a stopped vehicle 45 m directly ahead (+z)."""
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
            traffic_buf=_stopped_vehicle_buf(0.0, 45.0, vid=3),
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
    assert all(d.baseline_brake and not d.candidate_brake for d in diffs)


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
