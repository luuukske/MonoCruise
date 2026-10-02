"""A latched threat lets go once the danger has passed.

The latched set used to carry a headway hold: while any latched id sat inside 1.5 s of
headway, AEB kept braking at 70 % of max, whatever the target was doing. Clip 33d87007
braked 3.8 s after a cut-in had matched speed and was pulling away, and the driver had
to floor the gas to get out. The set now only keeps a target visible to the pipeline
(filter bypass, instant re-engage) while it can still steer into ego's lane, and never
for longer than `latched_max_s` past its last hit. Braking after a threat comes only
from demand, the TTB slam or the geometry latch. See core/aeb/README.md (Latched-threat
hold).
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

import core.aeb.clip_eval as clip_eval
from core.aeb.calibration import DEFAULT as CAL
from core.aeb.clip_eval import run_headless
from core.aeb.clip_replay import decode_radar_stream
from core.aeb.clip_schema import (
    AEBTickRecord, Clip, ClipMetadata, ConsumedContext, EgoTelemetry, LiveAEB,
    RadarFrameRecord,
)
from core.aeb.thread import AEBThread
from core.radar.elevation import BODY_DATUM_FRAC
from core.radar.reader import _BUF_SIZE, _TOTAL_FORMAT

# Traffic position.y is the body datum, not ground level (README elevation).
_BODY_H: float = 3.0
_BODY_Y: float = BODY_DATUM_FRAC * _BODY_H
_HZ = 30.0
_DT = 1.0 / _HZ
_CAPACITY = 10.0
_LEAD_ID = 3


def _clip_from_profile(ego_speeds: list[float], lead_speeds: list[float],
                       gap_m: float, lead_x: list[float] | None = None) -> Clip:
    """One same-facing lead; positions integrated from the speed profiles."""
    frames, ticks = [], []
    ego_z, lead_z = 0.0, gap_m
    for i, (ego_ms, lead_ms) in enumerate(zip(ego_speeds, lead_speeds)):
        t = i * _DT
        x = lead_x[i] if lead_x is not None else 0.0
        # yaw = pi faces +Z like ego: quaternion (cos(pi/2), 0, sin(pi/2), 0).
        flat: list = [x, _BODY_Y, lead_z, 0.0, 0.0, 1.0, 0.0,
                      2.5, _BODY_H, 6.0, lead_ms, 0.0]
        flat += [0, _LEAD_ID, 0, 0] + [0.0] * 30
        for _ in range(39):
            flat += [0.0] * 12 + [0, 0, 0, 0] + [0.0] * 30
        buf = struct.pack(_TOTAL_FORMAT, *flat)
        assert len(buf) == _BUF_SIZE
        frames.append(RadarFrameRecord(
            t_wall=1000.0 + t, t_mono=t,
            ego=EgoTelemetry(coordinateX=0.0, coordinateZ=ego_z,
                             rotationX=0.5, rotationY=0.0, speed=ego_ms),
            traffic_buf=buf, parked_buf=None,
        ))
        ticks.append(AEBTickRecord(
            t_mono=t, radar_t_mono=t,
            consumed=ConsumedContext(max_brake_ms2=_CAPACITY, aeb_enabled=True),
            live_aeb=LiveAEB(),
        ))
        ego_z += ego_ms * _DT
        lead_z += lead_ms * _DT
    return Clip(metadata=ClipMetadata.create(session_kind="SP"),
                radar_frames=frames, aeb_ticks=ticks)


_N_CLOSE, _N_AFTER = 45, 105


def _pull_away_clip() -> Clip:
    """Ego closes on a slower lead, then the lead accelerates away while ego sheds speed."""
    ego_speeds = [21.0] * _N_CLOSE
    lead_speeds = [15.0] * _N_CLOSE
    for i in range(_N_AFTER):
        ego_speeds.append(max(4.0, 21.0 - 3.0 * (i + 1) * _DT))
        lead_speeds.append(min(30.0, 15.0 + 4.0 * (i + 1) * _DT))
    return _clip_from_profile(ego_speeds, lead_speeds, gap_m=18.0)


def _speed_match_clip() -> Clip:
    """Same entry, but the lead only climbs to ego's speed: closing goes to 0, not below."""
    ego_speeds = [21.0] * (_N_CLOSE + _N_AFTER)
    lead_speeds = [15.0] * _N_CLOSE
    for i in range(_N_AFTER):
        lead_speeds.append(min(21.0, 15.0 + 3.0 * (i + 1) * _DT))
    return _clip_from_profile(ego_speeds, lead_speeds, gap_m=18.0)


def _braked_past_match_clip() -> Clip:
    """Ego brakes through a steady lead's speed at a tight gap; the lead never changes."""
    ego_speeds = [21.0] * _N_CLOSE
    for i in range(_N_AFTER):
        ego_speeds.append(max(13.5, 21.0 - 3.0 * (i + 1) * _DT))
    lead_speeds = [15.0] * (_N_CLOSE + _N_AFTER)
    return _clip_from_profile(ego_speeds, lead_speeds, gap_m=18.0)


def _swerve_clip(settle_x: float) -> Clip:
    """Ego closes on a slower lead that swerves out to `settle_x` and matches ego's speed."""
    ego_speeds = [21.0] * (_N_CLOSE + _N_AFTER)
    lead_speeds = [15.0] * _N_CLOSE
    lead_x = [0.0] * _N_CLOSE
    for i in range(_N_AFTER):
        frac = min(1.0, (i + 1) * _DT / 1.0)
        lead_x.append(settle_x * frac)
        lead_speeds.append(min(21.0, 15.0 + 6.0 * (i + 1) * _DT))
    return _clip_from_profile(ego_speeds, lead_speeds, gap_m=18.0, lead_x=lead_x)


def _threat_and_brake_ends(clip: Clip, as_recorded: bool = False) -> tuple[float, float]:
    stream = decode_radar_stream(clip, as_recorded=as_recorded)
    ticks = run_headless(clip, cal=CAL, stream=stream)
    hits = [e.t_rel for e in ticks if e.colliding_ids]
    braked = [e.t_rel for e in ticks if e.aeb_brake]
    assert hits and braked, "scenario must engage"
    return hits[-1], braked[-1]


@pytest.mark.parametrize("make", [_speed_match_clip, _braked_past_match_clip, _pull_away_clip])
def test_brake_ends_with_the_threat(make):
    """No hold past the last hit: a matched, opening or pulled-away lead is not braked for.

    The first two pinned the opposite before: the hold kept braking a matched lead to the
    clip end. A lead that keeps braking still holds AEB through the demand and the
    geometry latch, not through the latched set.
    """
    last_hit, last_brake = _threat_and_brake_ends(make())
    assert last_brake <= last_hit + _DT + 1e-6


def _latched_timeline(clip: Clip, monkeypatch) -> list[tuple[float, bool, bool]]:
    """(t, lead latched, lead colliding) per tick, from the real thread."""
    seen: list[tuple[float, bool, bool]] = []

    class _Spy(AEBThread):
        def loop(self):
            super().loop()
            snap = self.data.snapshot
            seen.append((self._now(), _LEAD_ID in self._latched_threat_ids,
                         _LEAD_ID in set(snap.colliding_ids)))

    monkeypatch.setattr(clip_eval, "AEBThread", _Spy)
    run_headless(clip, cal=CAL)
    return seen


def test_a_target_that_can_still_steer_in_stays_latched_for_a_bounded_time(monkeypatch):
    """Adjacent lane ahead: kept past the old 0.5 s ego-lane grace, dropped at latched_max_s."""
    seen = _latched_timeline(_swerve_clip(settle_x=3.6), monkeypatch)
    hit_times = [t for t, _, hit in seen if hit]
    assert hit_times, "the lead must be a hit before it swerves"
    last_hit = hit_times[-1]
    mid = [lat for t, lat, _ in seen
           if last_hit + CAL.latched_scope_release_s + 0.2 < t < last_hit + CAL.latched_max_s - 0.2]
    late = [lat for t, lat, _ in seen if t > last_hit + CAL.latched_max_s + _DT]
    assert mid and all(mid), "a car one lane over can still steer in: keep it latched"
    assert late and not any(late), "no latch outlives latched_max_s past the last hit"


def test_a_target_beyond_the_steer_in_band_is_released_after_the_grace(monkeypatch):
    seen = _latched_timeline(_swerve_clip(settle_x=9.0), monkeypatch)
    last_hit = [t for t, _, hit in seen if hit][-1]
    out_of_band = [t for t, lat, _ in seen if t > last_hit and not lat]
    assert out_of_band, "a target two lanes over must be released"
    assert out_of_band[0] < last_hit + CAL.latched_max_s - 0.2, "the band, not the cap, released it"


def _store_clip(name: str) -> Path | None:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        return None
    for store in ("aeb_clips_contributed", "aeb_clips"):
        p = Path(base) / "MonoCruise" / store / name
        if p.is_file():
            return p
    return None


CLIP_33D87007 = "20260921T210539Z_auto_engagement_33d87007.json.gz"
CLIP_D16D0575 = "20260911T200700Z_auto_engagement_d16d0575.json.gz"


@pytest.mark.needs_clips
@pytest.mark.skipif(_store_clip(CLIP_33D87007) is None, reason="clip 33d87007 not in local clip store")
def test_clip_33d87007_stops_braking_when_the_cut_in_is_matched():
    """Cut-in at 120 km/h: the threat ends at 5.31 s, the old hold braked on to 9.21 s."""
    from core.aeb.clip_store import ClipStore

    clip = ClipStore().load(_store_clip(CLIP_33D87007))
    assert clip is not None
    last_hit, last_brake = _threat_and_brake_ends(clip)
    assert last_hit < 5.5
    assert last_brake <= last_hit + 0.1


@pytest.mark.needs_clips
@pytest.mark.skipif(_store_clip(CLIP_D16D0575) is None, reason="clip d16d0575 not in local clip store")
def test_clip_d16d0575_stops_braking_when_the_threat_collapses():
    """Entry is real on both clocks (0.74 s behind a lead braking 1.5 m/s^2); the tail is not.

    The old hold braked at 0.4 m/s^2 of demand to 10.7 s. Now each brake ends with its hits,
    and the lead closing in again on the recorded clock is a second, separate event.
    """
    from core.aeb.clip_store import ClipStore

    clip = ClipStore().load(_store_clip(CLIP_D16D0575))
    assert clip is not None
    for as_recorded in (True, False):
        last_hit, last_brake = _threat_and_brake_ends(clip, as_recorded=as_recorded)
        assert last_brake <= last_hit + 0.1
