"""Pre-upload triage: the three refusals, the sampled straight class, and fail-open.

The rates behind these thresholds were measured over the contributed corpus and
are recorded in core/aeb/README.md section 15. These tests pin the shape of the
rules, not the rates.
"""
from __future__ import annotations

import pytest

from core.aeb import upload as upload_mod
from core.aeb.clip_schema import AEBTickRecord, Clip, ClipMetadata, LiveAEB
from core.aeb.clip_store import ClipStore, serialize_clip
from core.aeb.clip_triage import (
    NEAR_RANGE_M,
    QUIET_TTC_S,
    STRAIGHT_MAX_KMH,
    STRAIGHT_SAMPLE_EVERY,
    SceneSummary,
    is_straight_slow,
    sample_keeps,
    summarize,
    triage_reason,
)
from core.aeb.upload import ClipUploader, SubmissionLog


@pytest.fixture(autouse=True)
def _fresh_sample_counter():
    """The sample position lives in settings, so it leaks between tests otherwise."""
    from core.settings import Settings

    Settings.save({"aeb_triage_straight_seen": 0})
    yield
    Settings.save({"aeb_triage_straight_seen": 0})


def _summary(**over) -> SceneSummary:
    base = dict(decoded=True, n_vehicles=8, nearest_range_m=15.0, min_ttc_s=1.2,
                brake_ticks=9, ego_speed_at_action_kmh=70.0, straight_in_lane=False)
    base.update(over)
    return SceneSummary(**base)


def test_a_clip_with_no_traffic_is_refused():
    assert triage_reason(_summary(n_vehicles=0)) == "no traffic in the clip"


def test_nothing_near_ego_all_clip_is_refused():
    assert triage_reason(_summary(nearest_range_m=NEAR_RANGE_M + 1.0)) is not None
    assert triage_reason(_summary(nearest_range_m=NEAR_RANGE_M - 1.0)) is None


def test_a_quiet_clip_is_refused_only_when_aeb_never_braked():
    quiet = _summary(min_ttc_s=QUIET_TTC_S + 1.0, brake_ticks=0)
    assert triage_reason(quiet) is not None
    assert triage_reason(_summary(min_ttc_s=QUIET_TTC_S + 1.0, brake_ticks=3)) is None


def test_a_crash_clip_that_missed_a_close_target_still_goes():
    """G4 was deliberately not implemented: a missed collision is the point."""
    missed = _summary(min_ttc_s=0.4, brake_ticks=0, nearest_range_m=6.0)
    assert triage_reason(missed) is None


def test_an_undecodable_clip_fails_open():
    assert triage_reason(SceneSummary(decoded=False, n_vehicles=0)) is None
    assert is_straight_slow(SceneSummary(decoded=False, straight_in_lane=True)) is False


def test_straight_slow_is_not_a_refusal_reason():
    """It is sampled, so it must not short-circuit as an ineligibility."""
    s = _summary(straight_in_lane=True, ego_speed_at_action_kmh=20.0)
    assert triage_reason(s) is None
    assert is_straight_slow(s) is True


def test_straight_above_the_speed_bar_is_not_sampled():
    assert not is_straight_slow(
        _summary(straight_in_lane=True, ego_speed_at_action_kmh=STRAIGHT_MAX_KMH))


def test_other_geometry_below_the_speed_bar_is_not_sampled():
    assert not is_straight_slow(
        _summary(straight_in_lane=False, ego_speed_at_action_kmh=10.0))


def test_the_sample_keeps_the_first_and_then_every_tenth():
    kept = [n for n in range(30) if sample_keeps(n, STRAIGHT_SAMPLE_EVERY)]
    assert kept == [0, 10, 20]


def test_a_clip_with_no_ticks_is_never_judged():
    """No AEB ticks means nothing to decode against, so the clip is offered."""
    out = summarize(Clip(metadata=ClipMetadata(), radar_frames=[], aeb_ticks=[]))
    assert out.decoded is False
    assert triage_reason(out) is None


def _uploader(tmp_path):
    store = ClipStore(root=tmp_path / "clips")
    return ClipUploader(store, transport=lambda *a: (200, {}),
                        log=SubmissionLog(tmp_path / "log.jsonl"), min_send_gap_s=0.0)


def _blob() -> bytes:
    clip = Clip(metadata=ClipMetadata(clip_id="c" * 8),
                aeb_ticks=[AEBTickRecord(t_mono=0.0, radar_t_mono=0.0, live_aeb=LiveAEB())])
    return serialize_clip(clip)


def test_the_uploader_refuses_a_clip_triage_rejects(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_mod, "summarize", lambda clip: _summary(n_vehicles=0))
    assert _uploader(tmp_path)._triage_reason(_blob()) == "no traffic in the clip"


def test_the_uploader_samples_the_straight_class(tmp_path, monkeypatch):
    monkeypatch.setattr(
        upload_mod, "summarize",
        lambda clip: _summary(straight_in_lane=True, ego_speed_at_action_kmh=20.0))
    up = _uploader(tmp_path)
    verdicts = [up._triage_reason(_blob()) for _ in range(STRAIGHT_SAMPLE_EVERY + 1)]
    assert verdicts[0] is None
    assert all(v is not None for v in verdicts[1:STRAIGHT_SAMPLE_EVERY])
    assert verdicts[STRAIGHT_SAMPLE_EVERY] is None


def test_the_sample_counter_survives_a_new_uploader(tmp_path, monkeypatch):
    """A restart must not hand the driver a fresh keep on their next straight clip."""
    monkeypatch.setattr(
        upload_mod, "summarize",
        lambda clip: _summary(straight_in_lane=True, ego_speed_at_action_kmh=20.0))
    assert _uploader(tmp_path)._triage_reason(_blob()) is None
    assert _uploader(tmp_path)._triage_reason(_blob()) is not None


def test_a_blob_that_will_not_deserialize_fails_open(tmp_path):
    assert _uploader(tmp_path)._triage_reason(b"not a clip") is None


@pytest.mark.needs_clips
def test_triage_agrees_with_the_offline_features_on_a_real_clip():
    """Decode path runs end to end on a recorded clip rather than a stub."""
    from core.aeb.clip_store import contributed_clip_root, default_clip_root, deserialize_clip

    paths = []
    for root in (default_clip_root(), contributed_clip_root()):
        if root.is_dir():
            paths.extend(sorted(root.glob("*_auto_engagement_*.json.gz"))[:3])
    if not paths:
        pytest.skip("no recorded engagement clip available")
    for path in paths:
        out = summarize(deserialize_clip(path.read_bytes()))
        assert out.decoded is True
        assert out.n_vehicles >= 0
        assert out.nearest_range_m > 0.0
