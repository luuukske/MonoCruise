"""Speed-scaled stop buffer, latched at engagement. See core/aeb/README.md (clearance demand).

The buffer follows the closing speed while AEB is idle, so a slow approach plans to stop
closer. Once engaged it keeps the buffer it engaged with: otherwise every stop would end at
the short buffer, since the closing speed falls to a crawl by the end of any stop.
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

from tests.aeb.test_stop_distance_envelope import stop_against_stationary

_BODY_H = 3.0
_BODY_Y = BODY_DATUM_FRAC * _BODY_H
_HZ = 30.0
_FLAT = replace(CAL, stop_buffer_low_m=CAL.stop_buffer)


def _approach(v0: float, lead_z: float, decel: float, t_brake: float, n: int = 90) -> Clip:
    """Ego closing on a stopped in-lane car, braking at ``decel`` from ``t_brake``."""
    frames, ticks = [], []
    z, v = 0.0, v0
    for i in range(n):
        t = i / _HZ
        flat: list = [0.0, _BODY_Y, lead_z, 0.0, 0.0, 1.0, 0.0,
                      2.5, _BODY_H, 6.0, 0.0, 0.0, 0, 3, 0, 0] + [0.0] * 30
        for _ in range(39):
            flat += [0.0] * 12 + [0, 0, 0, 0] + [0.0] * 30
        buf = struct.pack(_TOTAL_FORMAT, *flat)
        assert len(buf) == _BUF_SIZE
        frames.append(RadarFrameRecord(
            t_wall=1000.0 + t, t_mono=t,
            ego=EgoTelemetry(coordinateX=0.0, coordinateZ=z, rotationX=0.5,
                             rotationY=0.0, speed=v),
            traffic_buf=buf, parked_buf=None,
        ))
        ticks.append(AEBTickRecord(
            t_mono=t, radar_t_mono=t,
            consumed=ConsumedContext(max_brake_ms2=10.0, aeb_enabled=True),
            live_aeb=LiveAEB(),
        ))
        if t >= t_brake:
            v = max(0.0, v - decel / _HZ)
        z += v / _HZ
    return Clip(metadata=ClipMetadata.create(session_kind="SP"),
                radar_frames=frames, aeb_ticks=ticks)


def test_an_engagement_keeps_the_buffer_it_engaged_with():
    """Engaged at 60 km/h, the demand stays on the full buffer while ego slows below 40."""
    clip = _approach(60.0 / 3.6, 40.0, 7.0, 1.0)
    shipped, flat = run_headless(clip, cal=CAL), run_headless(clip, cal=_FLAT)
    late = [(a, b) for a, b, f in zip(shipped, flat, clip.radar_frames)
            if a.engaged and f.ego.speed < 40.0 / 3.6]
    assert len(late) > 3, "the scenario must stay engaged while ego slows"
    for a, b in late:
        assert a.clearance_required_ms2 == b.clearance_required_ms2


def test_a_slow_approach_plans_the_short_buffer():
    """Closing at 25 km/h the idle demand is lower than on a flat 0.5 m buffer."""
    clip = _approach(25.0 / 3.6, 12.0, 0.0, 99.0, n=30)
    shipped, flat = run_headless(clip, cal=CAL), run_headless(clip, cal=_FLAT)
    idle = [(a, b) for a, b in zip(shipped, flat)
            if not b.engaged and b.clearance_required_ms2 > 0.5]
    assert idle, "the scenario must build a demand before engaging"
    assert all(a.clearance_required_ms2 < b.clearance_required_ms2 for a, b in idle)


def test_fast_stops_keep_their_gap_and_slow_ones_end_closer():
    """Stop sim: 60 and 90 km/h unchanged, 25 to 40 km/h stop nearer, all still short."""
    for kmh in (60, 90):
        for cap, trailer in ((8.0, False), (13.89, True)):
            shipped = stop_against_stationary(kmh, cap, cap, trailer)["gap"]
            flat = stop_against_stationary(kmh, cap, cap, trailer, cal=_FLAT)["gap"]
            assert shipped == flat, (kmh, trailer)
    for kmh in (25, 30, 40):
        for cap, trailer in ((8.0, False), (13.89, True)):
            shipped = stop_against_stationary(kmh, cap, cap, trailer)["gap"]
            flat = stop_against_stationary(kmh, cap, cap, trailer, cal=_FLAT)["gap"]
            assert 0.3 < shipped < flat, (kmh, trailer)
