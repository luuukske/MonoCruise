"""Crossing traffic end to end: the whole FSM, not just the demand function.

`tests/aeb/harness.evaluate_frame` decides state from TTB alone, so the scenario
suite cannot see a change in required decel at all. These drive synthetic clips
through `run_headless`, which runs filters, demand, confirm windows and the
engagement latch exactly as the thread does.
"""
from __future__ import annotations

import math
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

_HZ = 30.0
_EGO_MS = 80.0 / 3.6
_BODY_H = 3.0
_BODY_Y = BODY_DATUM_FRAC * _BODY_H
# Ego drives up +Z with rotationX 0.5; traffic yaw 270 deg travels up +X.
_EGO_YAW_NORM = 0.5
_CROSS_YAW_DEG = 270.0

_NO_CLEARANCE = replace(CAL, clearance_required_enabled=False)


def _traffic_buf(px: float, pz: float, yaw_rad: float, speed: float,
                 vid: int = 3) -> bytes:
    qw = math.cos(yaw_rad / 2.0)
    qy = math.sin(yaw_rad / 2.0)
    flat: list = [px, _BODY_Y, pz, qw, 0.0, qy, 0.0, 2.5, _BODY_H, 6.0, speed, 0.0]
    flat += [0, vid, 0, 0] + [0.0] * 30
    for _ in range(39):
        flat += [0.0] * 12 + [0, 0, 0, 0] + [0.0] * 30
    buf = struct.pack(_TOTAL_FORMAT, *flat)
    assert len(buf) == _BUF_SIZE
    return buf


def crossing_clip(x0_m: float, conflict_z_m: float, cross_ms: float,
                  n: int = 80) -> Clip:
    """Ego straight at 80 km/h; a crosser sweeps left to right across `conflict_z_m`.

    `x0_m` is how far left of ego's path the crosser starts, so it sets the
    phasing: more negative means it reaches ego's lane later.
    """
    dt = 1.0 / _HZ
    frames, ticks = [], []
    for i in range(n):
        t = i * dt
        frames.append(RadarFrameRecord(
            t_wall=1000.0 + t, t_mono=t,
            ego=EgoTelemetry(
                coordinateX=0.0, coordinateZ=_EGO_MS * t,
                rotationX=_EGO_YAW_NORM, rotationY=0.0,
                speed=_EGO_MS, userSteer=0.0,
            ),
            traffic_buf=_traffic_buf(
                x0_m + cross_ms * t, conflict_z_m,
                math.radians(_CROSS_YAW_DEG), cross_ms,
            ),
            parked_buf=None,
        ))
        ticks.append(AEBTickRecord(
            t_mono=t, radar_t_mono=t,
            consumed=ConsumedContext(max_brake_ms2=10.0, aeb_enabled=True),
            live_aeb=LiveAEB(),
        ))
    return Clip(
        metadata=ClipMetadata.create(trigger_source="auto_engagement",
                                     session_kind="SP"),
        radar_frames=frames, aeb_ticks=ticks,
    )


def _first_brake_t(evs) -> float | None:
    return next((e.t_rel for e in evs if e.aeb_brake), None)


# (cross speed km/h, conflict range m, crosser start X m). Measured cases where
# the two models disagree on when to engage.
_LATE_CASES = [
    (30.0, 50.0, -24.0),
    (50.0, 50.0, -36.0),
    (50.0, 60.0, -42.0),
    (70.0, 50.0, -48.0),
]


@pytest.mark.parametrize("cross_kmh,conflict_z,x0", _LATE_CASES)
def test_a_genuine_crosser_still_brakes_but_later(cross_kmh, conflict_z, x0):
    """The comfort win: the same event, entered once yielding stops being possible.

    Stopping at the intersection point was never what the truck had to do, so
    pricing the demand as a yield moves entry later without dropping the event.
    """
    clip = crossing_clip(x0, conflict_z, cross_kmh / 3.6)
    new_t = _first_brake_t(run_headless(clip))
    old_t = _first_brake_t(run_headless(clip, cal=_NO_CLEARANCE))
    assert old_t is not None, "precondition: the old demand model braked here"
    assert new_t is not None, "a genuine crossing collision must still brake"
    assert new_t > old_t, f"expected a later entry, got {new_t:.2f} vs {old_t:.2f}"


def test_the_delay_is_real_but_bounded():
    """Later, not absent. A crosser must not slip past the whole event."""
    for cross_kmh, conflict_z, x0 in _LATE_CASES:
        clip = crossing_clip(x0, conflict_z, cross_kmh / 3.6)
        new_t = _first_brake_t(run_headless(clip))
        old_t = _first_brake_t(run_headless(clip, cal=_NO_CLEARANCE))
        assert 0.0 < new_t - old_t < 1.0, (
            f"{cross_kmh} km/h at {conflict_z} m: {new_t - old_t:.2f} s later"
        )


def test_the_model_knows_a_crosser_vacates_the_corridor():
    """`clears` is the fact the old demand had no way to express."""
    clip = crossing_clip(-36.0, 50.0, 50.0 / 3.6)
    evs = run_headless(clip)
    assert any(e.clearance_clears_ids for e in evs)


def test_a_crosser_gets_a_pass_speed_before_it_gets_a_brake():
    """At range the answer is a speed to go through at, not a stop."""
    clip = crossing_clip(-36.0, 50.0, 50.0 / 3.6)
    evs = run_headless(clip)
    early = [e for e in evs if e.colliding_ids and not e.aeb_brake]
    assert early, "expected ticks that see the crosser without engaging"
    assert any(e.clearance_v_pass_ms > 1.0 for e in early)


def test_a_crosser_that_stalls_in_the_lane_is_not_yielded_to():
    """Nothing to yield behind, so the demand is stop-short and the brake stands."""
    clip = crossing_clip(0.0, 50.0, 0.0)
    evs = run_headless(clip)
    assert _first_brake_t(evs) is not None
    assert not any(e.clearance_clears_ids for e in evs)
