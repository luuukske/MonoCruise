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
_PITCH_ROT_Y = 0.0557         # telemetry rotationY: about 20 deg, offset ~3.4 m/s2
_DT = 0.032
_TICKS = 60


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
    """Steep pitch, empty road: no warn, even though the slope term clears the old bar."""
    out = run_headless(_pitched_clip(_PITCH_ROT_Y), CAL_DEFAULT)
    assert out, "replay produced no ticks"

    # Precondition: without the threat gate these ticks would have warned.
    offset = 9.81 * math.sin(_PITCH_ROT_Y * 2.0 * math.pi)
    effective_max = CAL_DEFAULT.ego_decel_frac * _CAPACITY_MS2 - offset
    assert offset >= CAL_DEFAULT.aeb_warn_frac * effective_max

    assert not any(tk.colliding_ids for tk in out)
    assert all(tk.time_to_brake >= 1e8 for tk in out)
    assert not any(tk.aeb_warn for tk in out)
    assert not any(tk.aeb_brake for tk in out)


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
    assert max(tk.required_decel_ms2 for tk in out) > 3.0, "slope term still reported"
    assert not any(tk.aeb_warn for tk in out)
