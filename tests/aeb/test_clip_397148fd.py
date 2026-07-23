"""Regression: crash clip 397148fd (TMP road-train). Replay must brake at should-trigger start."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

CLIP_NAME = "20260722T151614Z_auto_engagement_397148fd.json.gz"


def _clip_path() -> Path | None:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        return None
    p = Path(base) / "MonoCruise" / "aeb_clips" / CLIP_NAME
    return p if p.is_file() else None


pytestmark = pytest.mark.skipif(
    _clip_path() is None, reason="clip 397148fd not in local clip store",
)


@pytest.fixture(scope="module")
def clip():
    from core.aeb.clip_store import ClipStore
    c = ClipStore().load(_clip_path())
    assert c is not None
    return c


def test_crash_flag_latched_on_target_through_impact(clip):
    """Vehicle 440 must hold crash_confirmed from the collision to impact."""
    from core.aeb.clip_replay import decode_radar_stream

    veh_by_t, _, frame_t = decode_radar_stream(clip)
    t0 = frame_t[0]
    flag_by_t = {}
    lag_by_t = {}
    for ft in frame_t:
        v = next((x for x in veh_by_t.get(ft, []) if x.id == 440), None)
        if v is not None:
            flag_by_t[ft - t0] = v.crash_confirmed
            lag_by_t[ft - t0] = v._lag_since is not None
    # Crash flag must stay latched through window; wreck not frozen by lag filter.
    window = [t for t in flag_by_t if 7.3 <= t <= 9.2]
    assert window, "vehicle 440 missing from replay"
    assert all(flag_by_t[t] for t in window)
    assert not any(lag_by_t[t] for t in window)


def test_replayed_brake_starts_at_window_start(clip):
    """Headless replay must brake at the label window start, not 1 s late."""
    from core.aeb.clip_eval import outcome_under

    lbl = clip.metadata.label
    assert lbl is not None and lbl.should_trigger is not None
    out = outcome_under(clip)
    assert out.verdict == "true_positive"
    assert out.brake_window is not None
    from_t = float(lbl.should_trigger["from_t"])
    # Fixed pipeline must brake within 0.15 s of labeled should-trigger window start.
    assert out.brake_window[0] <= from_t + 0.15
