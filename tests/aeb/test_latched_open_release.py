"""Latched-threat hold releases on an opening gap, not just on headway.

The hold exists for a lead ego has speed-matched at an unsafe gap. Headway is a
following-distance metric, so on its own it cannot tell that case apart from a
lead that is pulling away, and the brake stayed floored at 70 % of max until the
truck was slow enough for the same metres to read as 1.5 s. See core/aeb/README.md
(Latched-threat hold, closing-rate release).
"""
from __future__ import annotations

import struct
from dataclasses import replace

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
_NO_RELEASE = replace(CAL, latched_open_release_ms=1e9)


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
    """Ego closes on a slower lead, then the lead accelerates away while ego coasts down.

    Ego sheds speed gently so its headway stays under `latched_min_headway_s`:
    the gap opens because the lead leaves, which is the case headway cannot see.
    """
    dt = 1.0 / _HZ
    ego_speeds = [21.0] * _N_CLOSE
    lead_speeds = [15.0] * _N_CLOSE
    for i in range(_N_AFTER):
        ego_speeds.append(max(6.0, 21.0 - 1.5 * (i + 1) * dt))
        lead_speeds.append(min(27.0, 15.0 + 3.0 * (i + 1) * dt))
    return _clip_from_profile(ego_speeds, lead_speeds, gap_m=18.0)


def _speed_match_clip() -> Clip:
    """Same entry, but the lead only climbs to ego's speed: closing goes to 0, not below.

    This is what the hold is for, and what the deadband has to survive.
    """
    dt = 1.0 / _HZ
    ego_speeds = [21.0] * (_N_CLOSE + _N_AFTER)
    lead_speeds = [15.0] * _N_CLOSE
    for i in range(_N_AFTER):
        lead_speeds.append(min(21.0, 15.0 + 3.0 * (i + 1) * dt))
    return _clip_from_profile(ego_speeds, lead_speeds, gap_m=18.0)


def _brake_window(clip: Clip, cal) -> tuple[float, float] | None:
    braked = [e.t_rel for e in run_headless(clip, cal=cal) if e.aeb_brake]
    return (braked[0], braked[-1]) if braked else None


def test_hold_releases_once_the_lead_is_pulling_away():
    """Both calibrations engage; only the gated one stops braking before the clip ends."""
    clip = _pull_away_clip()
    end_t = clip.aeb_ticks[-1].t_mono

    held = _brake_window(clip, _NO_RELEASE)
    assert held is not None, "scenario must engage without the closing-rate release"
    assert held[1] >= end_t - 0.1, "un-gated hold is expected to run to the clip end"

    gated = _brake_window(clip, CAL)
    assert gated is not None, "the closing-rate release must not block engagement"
    assert gated[0] == held[0], "entry timing must be untouched"
    assert gated[1] < held[1] - 0.5, "an opening gap must release the hold"


def test_hold_survives_a_speed_matched_lead():
    """Lead climbs to ego's speed and stops there: the case the hold exists for.

    Closing lands on 0 rather than going negative, so the deadband must hold the
    brake all the way to the clip end exactly as the un-gated build does.
    """
    clip = _speed_match_clip()
    end_t = clip.aeb_ticks[-1].t_mono
    gated = _brake_window(clip, CAL)
    assert gated is not None, "speed-matched scenario must still engage"
    assert gated[1] >= end_t - 0.1, "a matched gap must not release the hold"
    assert gated == _brake_window(clip, _NO_RELEASE)


def test_release_threshold_is_the_shipped_default():
    assert CAL.latched_open_release_ms == 0.5
