"""Closing-speed comfort floor: sub-1 m/s relative speed never takes the brake.

An automatic stop is itself a jolt, so AEB only earns it when the contact it
prevents is worth more than the stop costs. Under 1 m/s of relative speed the
contact is a 3.6 km/h nudge. See core/aeb/README.md (closing-speed floor).
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
_NO_FLOOR = replace(CAL, aeb_min_closing_ms=0.0)


def _following_clip(ego_ms: float, lead_ms: float, gap_m: float,
                    capacity: float = 10.0, n: int = 60) -> Clip:
    """Ego following an in-lane, same-facing lead at a constant speed difference."""
    dt = 1.0 / _HZ
    frames, ticks = [], []
    for i in range(n):
        t = i * dt
        lead_z = gap_m + lead_ms * t
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
            ego=EgoTelemetry(coordinateX=0.0, coordinateZ=ego_ms * t,
                             rotationX=0.5, rotationY=0.0, speed=ego_ms),
            traffic_buf=buf, parked_buf=None,
        ))
        ticks.append(AEBTickRecord(
            t_mono=t, radar_t_mono=t,
            consumed=ConsumedContext(max_brake_ms2=capacity, aeb_enabled=True),
            live_aeb=LiveAEB(),
        ))
    return Clip(metadata=ClipMetadata.create(session_kind="SP"),
                radar_frames=frames, aeb_ticks=ticks)


def _braked(clip: Clip, cal) -> bool:
    return any(e.aeb_brake for e in run_headless(clip, cal=cal))


def test_floor_silences_a_sub_bar_creep_into_the_lead():
    """0.6 m/s of closure onto a touching lead: the old bar braked, the floor does not."""
    clip = _following_clip(ego_ms=20.0, lead_ms=19.4, gap_m=8.0)
    assert _braked(clip, _NO_FLOOR), "geometry must demand a brake without the floor"
    assert not _braked(clip, CAL)


def test_floor_leaves_a_real_rear_end_alone():
    """6 m/s of closure is a rear-end, not a nudge: unchanged by the floor."""
    clip = _following_clip(ego_ms=20.0, lead_ms=14.0, gap_m=22.0)
    assert _braked(clip, CAL)
    assert _braked(clip, _NO_FLOOR)


def test_floor_costs_latency_not_silence_on_warn():
    """A vetoed target still warns: the floor is engagement entry only (README)."""
    clip = _following_clip(ego_ms=20.0, lead_ms=19.4, gap_m=8.0)
    assert any(e.aeb_warn for e in run_headless(clip, cal=CAL))


def test_floor_is_the_shipped_default():
    assert CAL.aeb_min_closing_ms == 1.0
