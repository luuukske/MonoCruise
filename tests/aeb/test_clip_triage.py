"""Pre-upload triage: refusals, scene classes, sampling, the daily budget, fail-open.

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


_TRIAGE_STATE = {"aeb_triage_straight_seen": 0, "aeb_triage_standard_seen": 0,
                 "aeb_triage_reckless_seen": 0, "aeb_triage_day": "",
                 "aeb_triage_day_sent": 0}


@pytest.fixture(autouse=True)
def _fresh_sample_counter():
    """Sample positions and the day count live in settings, so they leak otherwise."""
    from core.settings import Settings

    Settings.save(dict(_TRIAGE_STATE))
    yield
    Settings.save(dict(_TRIAGE_STATE))


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


# Scene classes: who caused the encounter (README section 15).

from core.aeb import clip_triage as ct  # noqa: E402

_ACTION_T = 5.0
_DT = 1.0 / 30.0


def _ego(kmh: float = 70.0, yaw_at=lambda t: 0.0) -> list:
    return [(i * _DT, yaw_at(i * _DT), kmh / 3.6) for i in range(int(6.0 / _DT))]


def _track(vid: int, path, *, kmh: float = 70.0, dot: float = 1.0) -> ct._Track:
    """``path(t)`` gives (fwd, right, world yaw) in the ego frame at time t."""
    tr = ct._Track(vid=vid, length_m=5.0, heading_dot_at_min=dot, speed_max_kmh=kmh)
    for i in range(int(6.0 / _DT)):
        t = i * _DT
        fwd, right, yaw = path(t)
        tr.history.append((t, fwd, right, yaw, kmh / 3.6))
        tr.ticks += 1
        tr.ahead_ticks += fwd > 0.0
        tr.in_lane_ticks += abs(right) <= ct._CAL.lane_half_width
    return tr


def _ramp(t: float, t0: float, t1: float) -> float:
    return min(1.0, max(0.0, (t - t0) / (t1 - t0)))


def _classify(tracks, primary, ego, kmh=70.0):
    return ct._classify_scene({t.vid: t for t in tracks}, primary, ego, _ACTION_T, kmh)


def test_a_target_steering_into_ego_lane_is_another_drivers_cut_in():
    import math
    lead = _track(1, lambda t: (25.0, 3.7 * (1.0 - _ramp(t, 2.0, 4.0)),
                                math.radians(-6.0) * math.sin(math.pi * _ramp(t, 2.0, 4.0))))
    assert _classify([lead], lead, _ego()) == (ct.SCENE_OTHER_DRIVER, "cut-in")


def test_ego_changing_lane_into_a_short_gap_is_ego_reckless():
    import math
    ego = _ego(yaw_at=lambda t: math.radians(6.0) * math.sin(math.pi * _ramp(t, 2.0, 4.0)))
    lead = _track(1, lambda t: (10.0, 3.7 * (1.0 - _ramp(t, 2.0, 4.0)), 0.0))
    scene, why = _classify([lead], lead, ego)
    assert scene == ct.SCENE_EGO_RECKLESS
    assert "lane" in why


def test_ego_changing_lane_with_room_to_spare_is_not_reckless():
    import math
    ego = _ego(yaw_at=lambda t: math.radians(6.0) * math.sin(math.pi * _ramp(t, 2.0, 4.0)))
    lead = _track(1, lambda t: (38.0, 3.7 * (1.0 - _ramp(t, 2.0, 4.0)), 0.0))
    assert _classify([lead], lead, ego)[0] != ct.SCENE_EGO_RECKLESS


def test_both_turning_together_is_the_road_bending_not_a_cut_in():
    import math
    bend = lambda t: math.radians(8.0) * _ramp(t, 2.0, 4.0)  # noqa: E731
    lead = _track(1, lambda t: (25.0, 3.7 * (1.0 - _ramp(t, 2.0, 4.0)), bend(t)))
    assert _classify([lead], lead, _ego(yaw_at=bend))[0] != ct.SCENE_OTHER_DRIVER


def test_a_cut_in_in_front_of_a_crawling_truck_is_not_priority():
    import math
    lead = _track(1, lambda t: (25.0, 3.7 * (1.0 - _ramp(t, 2.0, 4.0)),
                                math.radians(-6.0) * math.sin(math.pi * _ramp(t, 2.0, 4.0))))
    assert _classify([lead], lead, _ego(kmh=10.0), kmh=10.0)[0] != ct.SCENE_OTHER_DRIVER


def test_ego_past_every_truck_limit_is_reckless_whatever_the_scene():
    lead = _track(1, lambda t: (25.0, 0.0, 0.0))
    scene, _ = _classify([lead], lead, _ego(kmh=150.0), kmh=ct.RECKLESS_SPEED_KMH)
    assert scene == ct.SCENE_EGO_RECKLESS


def test_a_crossing_target_is_another_driver_only_while_ego_is_moving():
    cross = _track(1, lambda t: (20.0, 0.0, 1.5), dot=0.0)
    assert _classify([cross], cross, _ego())[0] == ct.SCENE_OTHER_DRIVER
    assert _classify([cross], cross, _ego(kmh=5.0), kmh=5.0)[0] != ct.SCENE_OTHER_DRIVER


def test_a_steady_lead_in_lane_is_standard_braking():
    lead = _track(1, lambda t: (25.0, 0.2, 0.0))
    assert _classify([lead], lead, _ego()) == (ct.SCENE_STANDARD, "lead in lane")


def test_other_driver_scenes_are_never_sampled_or_budgeted():
    s = _summary(scene=ct.SCENE_OTHER_DRIVER, straight_in_lane=True, ego_speed_at_action_kmh=20.0)
    assert ct.sample_class(s) is None
    assert ct.counts_toward_budget(s) is False


def test_each_sampled_class_keeps_its_own_rate():
    assert ct.sample_class(_summary(scene=ct.SCENE_EGO_RECKLESS)) == (
        "reckless", ct.RECKLESS_SAMPLE_EVERY)
    assert ct.sample_class(_summary(scene=ct.SCENE_STANDARD)) == (
        "standard", ct.STANDARD_SAMPLE_EVERY)
    assert ct.sample_class(_summary(straight_in_lane=True, ego_speed_at_action_kmh=20.0)) == (
        "straight", STRAIGHT_SAMPLE_EVERY)
    assert ct.sample_class(_summary()) is None


def test_crash_clips_never_spend_the_daily_budget():
    """Crash clips hold most of the contributed misses."""
    assert ct.counts_toward_budget(_summary(crash_trigger=True)) is False
    assert ct.counts_toward_budget(_summary()) is True
    assert ct.counts_toward_budget(SceneSummary(decoded=False)) is False


def test_the_daily_budget_stops_ordinary_clips_but_not_priority_ones(tmp_path, monkeypatch):
    up = _uploader(tmp_path)
    monkeypatch.setattr(upload_mod, "summarize", lambda clip: _summary())
    verdicts = [up._triage_reason(_blob()) for _ in range(ct.DAILY_BUDGET + 1)]
    assert all(v is None for v in verdicts[:ct.DAILY_BUDGET])
    assert verdicts[-1] is not None
    monkeypatch.setattr(upload_mod, "summarize",
                        lambda clip: _summary(scene=ct.SCENE_OTHER_DRIVER))
    assert up._triage_reason(_blob()) is None


def test_the_daily_budget_resets_on_a_new_day(tmp_path, monkeypatch):
    from core.settings import Settings

    Settings.save({"aeb_triage_day": "2000-01-01", "aeb_triage_day_sent": ct.DAILY_BUDGET})
    monkeypatch.setattr(upload_mod, "summarize", lambda clip: _summary())
    assert _uploader(tmp_path)._triage_reason(_blob()) is None


def test_reckless_scenes_are_sampled_one_in_n(tmp_path, monkeypatch):
    monkeypatch.setattr(upload_mod, "summarize",
                        lambda clip: _summary(scene=ct.SCENE_EGO_RECKLESS, scene_why="x"))
    up = _uploader(tmp_path)
    verdicts = [up._triage_reason(_blob()) for _ in range(ct.RECKLESS_SAMPLE_EVERY + 1)]
    assert [v is None for v in verdicts] == (
        [True] + [False] * (ct.RECKLESS_SAMPLE_EVERY - 1) + [True])


def test_a_retried_clip_does_not_draw_a_sample_slot_again(tmp_path, monkeypatch):
    """It won its slot when first offered; a second draw would refuse it 9 times in 10."""
    monkeypatch.setattr(
        upload_mod, "summarize",
        lambda clip: _summary(straight_in_lane=True, ego_speed_at_action_kmh=20.0))
    up = _uploader(tmp_path)
    assert up._triage_reason(_blob()) is None
    assert up._triage_reason(_blob(), retry=True) is None
    assert up._triage_reason(_blob()) is not None
