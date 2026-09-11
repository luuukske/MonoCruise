"""Latched-threat hold releases on an opening gap, not just on raw headway.

The hold exists for a lead ego has speed-matched at an unsafe gap. Raw headway is a
following-distance metric, so on its own it cannot tell that case apart from a lead
that is pulling away, and the brake stayed floored at 70 % of max until the truck was
slow enough for the same metres to read as 1.5 s. The release must key on whether the
gap is still unsafe, not on the sign of the closing rate: braking past speed-match is
the normal end of a correct intervention. See core/aeb/README.md (Latched-threat
hold, opening-gap lookahead).
"""
from __future__ import annotations

import struct
from dataclasses import replace

import pytest

from core.aeb.calibration import DEFAULT as CAL
from core.aeb.clip_eval import run_headless
from core.aeb.clip_schema import (
    AEBTickRecord, Clip, ClipMetadata, ConsumedContext, EgoTelemetry, LiveAEB,
    RadarFrameRecord,
)
from core.radar.elevation import BODY_DATUM_FRAC
from core.radar.reader import _BUF_SIZE, _TOTAL_FORMAT

# Traffic position.y is the body datum, not ground level (README elevation).
_BODY_H: float = 3.0
_BODY_Y: float = BODY_DATUM_FRAC * _BODY_H
_HZ = 30.0
_CAPACITY = 10.0
_NO_RELEASE = replace(CAL, latched_open_lookahead_s=0.0)


def _clip_from_profile(ego_speeds: list[float], lead_speeds: list[float],
                       gap_m: float) -> Clip:
    """One in-lane, same-facing lead; positions integrated from the speed profiles."""
    dt = 1.0 / _HZ
    frames, ticks = [], []
    ego_z, lead_z = 0.0, gap_m
    for i, (ego_ms, lead_ms) in enumerate(zip(ego_speeds, lead_speeds)):
        t = i * dt
        # yaw = pi faces +Z like ego: quaternion (cos(pi/2), 0, sin(pi/2), 0).
        flat: list = [0.0, _BODY_Y, lead_z, 0.0, 0.0, 1.0, 0.0,
                      2.5, _BODY_H, 6.0, lead_ms, 0.0]
        flat += [0, 3, 0, 0] + [0.0] * 30
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
        ego_z += ego_ms * dt
        lead_z += lead_ms * dt
    return Clip(metadata=ClipMetadata.create(session_kind="SP"),
                radar_frames=frames, aeb_ticks=ticks)


_N_CLOSE, _N_AFTER = 45, 105


def _pull_away_clip() -> Clip:
    """Ego closes on a slower lead, then the lead accelerates away while ego sheds speed.

    Replay ego speed is scripted, so the hold's own braking cannot feed back into
    it and open the gap further. That understates the effect against a recorded
    clip: on `d16d0575` the un-gated hold never releases inside the recording at
    all, where here raw headway eventually clears the bar on its own.
    """
    dt = 1.0 / _HZ
    ego_speeds = [21.0] * _N_CLOSE
    lead_speeds = [15.0] * _N_CLOSE
    for i in range(_N_AFTER):
        ego_speeds.append(max(4.0, 21.0 - 3.0 * (i + 1) * dt))
        lead_speeds.append(min(30.0, 15.0 + 4.0 * (i + 1) * dt))
    return _clip_from_profile(ego_speeds, lead_speeds, gap_m=18.0)


def _speed_match_clip() -> Clip:
    """Same entry, but the lead only climbs to ego's speed: closing goes to 0, not below.

    This is what the hold is for, and what the release has to survive.
    """
    dt = 1.0 / _HZ
    ego_speeds = [21.0] * (_N_CLOSE + _N_AFTER)
    lead_speeds = [15.0] * _N_CLOSE
    for i in range(_N_AFTER):
        lead_speeds.append(min(21.0, 15.0 + 3.0 * (i + 1) * dt))
    return _clip_from_profile(ego_speeds, lead_speeds, gap_m=18.0)


def _braked_past_match_clip() -> Clip:
    """Ego brakes through the lead's speed at a tight gap, the lead never changes.

    Closing goes negative while the gap is still metres, which is the normal end
    of a correct intervention. Clip `3c1bd9af` is this case in the wild.
    """
    dt = 1.0 / _HZ
    ego_speeds = [21.0] * _N_CLOSE
    for i in range(_N_AFTER):
        ego_speeds.append(max(13.5, 21.0 - 3.0 * (i + 1) * dt))
    lead_speeds = [15.0] * (_N_CLOSE + _N_AFTER)
    return _clip_from_profile(ego_speeds, lead_speeds, gap_m=18.0)


def _brake_window(clip: Clip, cal) -> tuple[float, float] | None:
    braked = [e.t_rel for e in run_headless(clip, cal=cal) if e.aeb_brake]
    return (braked[0], braked[-1]) if braked else None


def test_hold_releases_earlier_once_the_lead_is_pulling_away():
    """Same entry either way; the lookahead ends the brake sooner on an opening gap."""
    clip = _pull_away_clip()

    held = _brake_window(clip, _NO_RELEASE)
    assert held is not None, "scenario must engage without the opening-gap release"

    gated = _brake_window(clip, CAL)
    assert gated is not None, "the opening-gap release must not block engagement"
    assert gated[0] == held[0], "entry timing must be untouched"
    assert gated[1] < held[1] - 0.3, "an opening gap must release the hold sooner"


def test_hold_survives_a_speed_matched_lead():
    """Lead climbs to ego's speed and stops there: the case the hold exists for.

    Closing lands on 0 rather than going negative, so the hold must keep the brake
    on all the way to the clip end exactly as the un-gated build does.
    """
    clip = _speed_match_clip()
    end_t = clip.aeb_ticks[-1].t_mono
    gated = _brake_window(clip, CAL)
    assert gated is not None, "speed-matched scenario must still engage"
    assert gated[1] >= end_t - 0.1, "a matched gap must not release the hold"
    assert gated == _brake_window(clip, _NO_RELEASE)


def test_hold_survives_braking_past_the_lead_speed_at_a_tight_gap():
    """Negative closing at a tight gap is a correct intervention finishing, not a release.

    Regression for a rejected earlier design that released on `closing < -0.5 m/s`
    and cut this case from 2.7 s of braking to 0.6 s (README, opening-gap lookahead).
    """
    clip = _braked_past_match_clip()
    end_t = clip.aeb_ticks[-1].t_mono
    gated = _brake_window(clip, CAL)
    assert gated is not None, "braked-past-match scenario must still engage"
    assert gated[1] >= end_t - 0.1, "a tight gap must hold even once closing goes negative"
    assert gated == _brake_window(clip, _NO_RELEASE)


def test_lookahead_is_the_shipped_default():
    assert CAL.latched_open_lookahead_s == 1.0


CLIP_NAME = "20260911T200700Z_auto_engagement_d16d0575.json.gz"


def _clip_path():
    import os
    from pathlib import Path
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        return None
    for store in ("aeb_clips_contributed", "aeb_clips"):
        p = Path(base) / "MonoCruise" / store / CLIP_NAME
        if p.is_file():
            return p
    return None


@pytest.mark.needs_clips
@pytest.mark.skipif(_clip_path() is None, reason="clip d16d0575 not in local clip store")
def test_clip_d16d0575_stops_braking_inside_the_recording():
    """The reported case: threat gone at 9.14, un-gated brake still on when the clip ends.

    The recording stops 1.9 s after the threat collapses and the un-gated hold is
    still braking at 70 % of max, so the only bound the clip can prove is that the
    release now happens at all. See core/aeb/README.md (opening-gap lookahead).
    """
    from core.aeb.clip_store import ClipStore

    clip = ClipStore().load(_clip_path())
    assert clip is not None
    end_t = max(t.t_mono for t in clip.aeb_ticks) - min(
        [f.t_mono for f in clip.radar_frames] + [t.t_mono for t in clip.aeb_ticks]
    )

    held = _brake_window(clip, _NO_RELEASE)
    gated = _brake_window(clip, CAL)
    assert held is not None and gated is not None
    assert held[1] >= end_t - 0.1, "un-gated hold still braking when the recording ends"
    assert gated[0] == held[0], "entry timing must be untouched"
    assert gated[1] < end_t - 0.2, "the gated hold must release inside the recording"
