"""Regression: road slope must never source an AEB warn or FF demand on its own.

Phantom warns fired after crashes because `downhill_offset` raised
`effective_required` and lowered `warn_threshold` at the same time, so a steep
enough pitch crossed the bar with no target at all (clip b530ea7b).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from core.aeb.calibration import DEFAULT as CAL_DEFAULT
from core.aeb.clip_eval import run_headless
from core.radar.elevation import MAX_EGO_GRADE
from core.aeb.clip_schema import (
    AEBTickRecord,
    Clip,
    ClipMetadata,
    ConsumedContext,
    EgoTelemetry,
    LiveAEB,
    RadarFrameRecord,
)

_CAPACITY_MS2 = 9.52          # PedalCapacityTracker value logged in clip b530ea7b
# rotationY is the grade, +ve = climbing. -0.0557 is ~20 deg down, past
# MAX_EGO_GRADE, so the clamp reads it level; _STEEP_DESCENT is the steepest road.
_PITCH_ROT_Y = -0.0557
_STEEP_DESCENT_ROT_Y = -math.atan(MAX_EGO_GRADE) / (2.0 * math.pi)
_DT = 0.032
_TICKS = 60


def _offset_for(rotation_y: float) -> float:
    """downhill_offset the thread computes for this telemetry pitch."""
    grade = math.tan(rotation_y * 2.0 * math.pi)
    if abs(grade) > MAX_EGO_GRADE:
        grade = 0.0
    return max(-9.81 * math.sin(math.atan(grade)), 0.0)


def _pitched_clip(rotation_y: float) -> Clip:
    """No-traffic clip on a steep grade: traffic_buf None decodes to zero vehicles."""
    frames = []
    ticks = []
    for i in range(_TICKS):
        t = 100.0 + i * _DT
        ego = EgoTelemetry(
            coordinateX=0.0, coordinateY=0.0, coordinateZ=-i * 3.05 * _DT,
            rotationX=0.5, rotationY=rotation_y, speed=3.05,
        )
        frames.append(RadarFrameRecord(t_wall=t, t_mono=t, ego=ego))
        ticks.append(AEBTickRecord(
            t_mono=t, radar_t_mono=t,
            consumed=ConsumedContext(max_brake_ms2=_CAPACITY_MS2),
            live_aeb=LiveAEB(),
        ))
    return Clip(metadata=ClipMetadata.create(), radar_frames=frames, aeb_ticks=ticks)


def test_slope_alone_never_warns_without_a_target():
    """Steepest real descent, empty road: the threat gate keeps it silent."""
    out = run_headless(_pitched_clip(_STEEP_DESCENT_ROT_Y), CAL_DEFAULT)
    assert out, "replay produced no ticks"

    # Precondition: the slope term is live and substantial, so silence here is
    # the threat gate doing its job and not the offset being zero.
    offset = _offset_for(_STEEP_DESCENT_ROT_Y)
    assert offset > 1.5, f"descent produced no slope demand ({offset:.2f})"

    assert not any(tk.colliding_ids for tk in out)
    assert all(tk.time_to_brake >= 1e8 for tk in out)
    assert not any(tk.aeb_warn for tk in out)
    assert not any(tk.aeb_brake for tk in out)


def test_offset_applies_on_a_descent_and_never_on_a_climb():
    """The sign. A climb hands brake force back; only a descent steals it.

    Inverted, this term penalised every climb and did nothing on any descent:
    38.8 % of corpus ticks carried a phantom penalty and 49.4 % were missing the
    allowance the term exists for.
    """
    descent = _offset_for(_STEEP_DESCENT_ROT_Y)
    climb = _offset_for(-_STEEP_DESCENT_ROT_Y)
    assert descent > 1.5, f"descent must steal brake force, got {descent:.2f}"
    assert climb == 0.0, f"a climb must never pay a decel penalty, got {climb:.2f}"


def test_absurd_pitch_reads_level():
    """A wreck or an airborne truck is not a road: past MAX_EGO_GRADE reads level.

    Unclamped this injected up to 9.81 m/s2 into effective_required and out of
    capability_decel. Same bound and same discard-not-clamp policy as the
    elevation gate, which is why MAX_EGO_GRADE is shared rather than copied.
    """
    assert _offset_for(_PITCH_ROT_Y) == 0.0, "20 deg down is not a road grade"
    out = run_headless(_pitched_clip(_PITCH_ROT_Y), CAL_DEFAULT)
    assert out
    assert max(tk.required_decel_ms2 for tk in out) < 0.5
    assert not any(tk.aeb_warn or tk.aeb_brake for tk in out)


def test_flat_road_empty_of_traffic_is_also_quiet():
    """Control: the gate is what silences the pitched case, not the empty vehicle list."""
    out = run_headless(_pitched_clip(0.0), CAL_DEFAULT)
    assert out
    assert not any(tk.aeb_warn or tk.aeb_brake for tk in out)


_CLIP_NAME = "20260726T160702Z_auto_engagement_b530ea7b.json.gz"


def _clip_path() -> Path | None:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        return None
    p = Path(base) / "MonoCruise" / "aeb_clips" / _CLIP_NAME
    return p if p.is_file() else None


@pytest.mark.needs_clips
@pytest.mark.skipif(
    _clip_path() is None, reason="clip b530ea7b not in local clip store",
)
def test_clip_b530ea7b_no_longer_warns():
    """The originating FP: 315 ticks, no target ever, 9 warn ticks before the fix."""
    from core.aeb.clip_store import ClipStore

    clip = ClipStore().load(_clip_path())
    assert clip is not None
    out = run_headless(clip, CAL_DEFAULT)

    assert not any(tk.colliding_ids for tk in out), "clip is target-free by construction"
    # Pitches to +20.8 deg nose-up (grade 0.379): a crash attitude, not a road,
    # so the clamp zeroes the slope term. It used to report 3.16 m/s2 here.
    assert max(tk.required_decel_ms2 for tk in out) < 0.5, "slope term not clamped"
    assert not any(tk.aeb_warn for tk in out)
