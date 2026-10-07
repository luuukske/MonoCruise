"""Warn classes: a clear threat warns a second ahead, a moving crosser only with the brake.

A clear threat is held dead ahead by both the ego arc and the measured line, driving
ego's way or stopped. Facing ego it is still an obstacle, not oncoming traffic: the
2 s oncoming window held a parked trailer's warn to the brake tick on every stop at
the 2026-10-06 test track (README, warn classes).
"""
from __future__ import annotations

import math
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

_HZ = 30.0
_BODY_H = 3.0
_OLD = replace(CAL, aeb_warn_clear_class=False, aeb_warn_lead_s=0.0,
               aeb_warn_crossers_with_brake=False)


def _yaw_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    """Buffer order (w, x, y, z); the reader swaps x and y, so yaw lives in the third slot.

    Yaw 0 faces -Z, toward ego; 180 faces +Z with ego; -90 faces +X.
    """
    half = math.radians(yaw_deg) / 2.0
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def _clip(ego_ms: float, x0: float, z0: float, yaw_deg: float, speed: float,
          n: int = 105) -> Clip:
    """Ego on +Z at ``ego_ms``; one 6 m body driving its own heading at ``speed``."""
    dt = 1.0 / _HZ
    yaw = math.radians(yaw_deg)
    fx, fz = -math.sin(yaw), -math.cos(yaw)
    frames, ticks = [], []
    for i in range(n):
        t = i * dt
        flat: list = [x0 + fx * speed * t, BODY_DATUM_FRAC * _BODY_H, z0 + fz * speed * t,
                      *_yaw_quat(yaw_deg), 2.5, _BODY_H, 6.0, speed, 0.0]
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
            consumed=ConsumedContext(max_brake_ms2=10.0, aeb_enabled=True),
            live_aeb=LiveAEB(),
        ))
    return Clip(metadata=ClipMetadata.create(session_kind="SP"),
                radar_frames=frames, aeb_ticks=ticks)


def _first(ticks, attr: str) -> float | None:
    return next((t.t_rel for t in ticks if getattr(t, attr)), None)


def _lead(ticks) -> float:
    return _first(ticks, "aeb_brake") - _first(ticks, "aeb_warn")


def test_a_stopped_car_dead_ahead_warns_a_second_before_the_brake():
    """The plain inattention case: closing on a stalled car in ego's lane, no braking."""
    clip = _clip(ego_ms=22.0, x0=0.0, z0=110.0, yaw_deg=180.0, speed=0.0, n=150)
    new, old = run_headless(clip), run_headless(clip, cal=_OLD)
    assert _first(new, "aeb_brake") == _first(old, "aeb_brake"), "only the warn may move"
    assert _lead(new) >= 1.0
    assert _lead(old) < 0.75


def test_a_parked_body_facing_ego_warns_a_second_before_the_brake():
    clip = _clip(ego_ms=18.0, x0=0.0, z0=95.0, yaw_deg=0.0, speed=0.0, n=170)
    new, old = run_headless(clip), run_headless(clip, cal=_OLD)
    assert _first(new, "aeb_brake") == _first(old, "aeb_brake"), "only the warn may move"
    assert _lead(new) >= 1.0
    assert _lead(old) < 0.05, "the oncoming class held the warn to the brake"


def test_a_stopped_body_off_the_path_keeps_the_oncoming_wait():
    """Queues facing ego clip the corridor on bends; exempting them cost 12 false warns."""
    clip = _clip(ego_ms=18.0, x0=2.2, z0=60.0, yaw_deg=0.0, speed=0.0)
    new = run_headless(clip)
    assert any(t.colliding_ids for t in new), "the body must reach the corridor"
    old = run_headless(clip, cal=_OLD)
    assert [(t.aeb_warn, t.aeb_brake) for t in new] == [(t.aeb_warn, t.aeb_brake) for t in old]


def test_moving_head_on_traffic_keeps_the_oncoming_wait():
    """Only stopped or same-way bodies are clear: a moving one decides as before."""
    clip = _clip(ego_ms=18.0, x0=0.0, z0=110.0, yaw_deg=0.0, speed=8.0)
    new = [(t.aeb_warn, t.aeb_brake) for t in run_headless(clip)]
    old = [(t.aeb_warn, t.aeb_brake) for t in run_headless(clip, cal=_OLD)]
    assert new == old


def test_a_moving_crosser_warns_only_with_the_brake():
    """Perpendicular traffic turns or stops at the last second: no cue before AEB brakes."""
    clip = _clip(ego_ms=15.0, x0=-30.0, z0=44.0, yaw_deg=-90.0, speed=10.0, n=120)
    new, old = run_headless(clip), run_headless(clip, cal=_OLD)
    brake = _first(new, "aeb_brake")
    assert brake is not None and _first(old, "aeb_brake") == brake
    assert _first(new, "aeb_warn") == brake
    assert _first(old, "aeb_warn") < brake, "the old classes warned before the brake"
