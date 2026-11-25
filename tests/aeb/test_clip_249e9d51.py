"""Regression: TMP car turning left across a 90 km/h ego (249e9d51). README TMP crossing scope."""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

CLIP_NAME = "20261124T154554Z_auto_crash_249e9d51.json.gz"


def _clip_path() -> Path | None:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        return None
    p = Path(base) / "MonoCruise" / "aeb_clips" / CLIP_NAME
    return p if p.is_file() else None


pytestmark = [
    pytest.mark.needs_clips,
    pytest.mark.skipif(_clip_path() is None, reason="clip 249e9d51 not in local clip store"),
]


@pytest.fixture(scope="module")
def clip():
    from core.aeb.clip_store import ClipStore
    c = ClipStore().load(_clip_path())
    assert c is not None
    return c


def _brake_onset(clip, cal) -> float | None:
    from core.aeb.clip_eval import run_headless
    braked = [e.t_rel for e in run_headless(clip, cal=cal) if e.aeb_brake]
    return braked[0] if braked else None


def test_brakes_inside_the_window_once_braking_can_no_longer_stop(clip):
    """Measured 3.54 s; the window is 3.126 to 4.495, eligible from 3.38 s, impact about 4.3 s."""
    from core.aeb.calibration import DEFAULT
    onset = _brake_onset(clip, DEFAULT)
    assert onset is not None
    assert 3.126 <= onset <= 3.58


def test_the_oblique_confirm_alone_brakes_at_375(clip):
    """Without the crosser confirm the 0.40 s oblique window holds the brake to 3.75 s."""
    from core.aeb.calibration import DEFAULT
    onset = _brake_onset(clip, replace(DEFAULT, tmp_cross_gate_confirm_s=None))
    assert onset is not None and 3.70 <= onset <= 3.80


def test_the_bare_stage_drops_it_to_impact(clip):
    """Without the gate TmpCrossTrafficFilter's 3 s end point drops the car from 31.7 m on."""
    from core.aeb.calibration import DEFAULT
    assert _brake_onset(clip, replace(DEFAULT, avoidability_gate_tmp_cross=False)) is None
