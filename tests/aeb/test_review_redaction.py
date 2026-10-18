"""What the review endpoint's redaction relies on (ld-tech `api/v1/aeb_review.php`).

Reviewers get every t_wall shifted by the same whole number of seconds inside
[2^30, 2^31) s, so every difference stays exact and AEB must decide the same.
"""
from __future__ import annotations

import copy

from core.aeb.clip_eval import run_headless
from core.aeb.clip_replay import clip_t0, replay_clip
from core.aeb.clip_schema import Label
from tests.aeb.test_clip_eval import _collision_clip

_WALL0 = 1_792_000_151.7459497
_SHIFTS_S = (-404_179_200, -2_678_399, 2_678_401, 138_758_437)


def _real_clock_clip():
    clip = _collision_clip()
    for f in clip.radar_frames:
        f.t_wall = _WALL0 + (f.t_wall - 1000.0)
    return clip


def _shifted(clip, seconds: int):
    out = copy.deepcopy(clip)
    for f in out.radar_frames:
        f.t_wall = f.t_wall + float(seconds)
    return out


def _decisions(clip):
    return [(t.aeb_brake, t.aeb_warn, sorted(t.colliding_ids)) for t in run_headless(clip)]


def test_a_whole_second_shift_inside_the_binade_keeps_every_difference_exact():
    clip = _real_clock_clip()
    steps = [b.t_wall - a.t_wall for a, b in zip(clip.radar_frames, clip.radar_frames[1:])]
    for seconds in _SHIFTS_S:
        moved = _shifted(clip, seconds)
        assert all(1 << 30 <= f.t_wall < 1 << 31 for f in moved.radar_frames)
        assert [b.t_wall - a.t_wall for a, b in zip(moved.radar_frames, moved.radar_frames[1:])] == steps


def test_aeb_decides_the_same_on_a_shifted_clock():
    # Not bit-identical in general: two least-squares fits average absolute
    # timestamps, which moves target decel by float noise (2e-13 m/s2 measured).
    clip = _real_clock_clip()
    base = _decisions(clip)
    assert any(d[0] for d in base), "fixture must engage"
    base_review = [(f.t_rel, f.live_aeb.aeb_brake) for f in replay_clip(clip)]
    for seconds in _SHIFTS_S:
        moved = _shifted(clip, seconds)
        assert _decisions(moved) == base
        assert [(f.t_rel, f.live_aeb.aeb_brake) for f in replay_clip(moved)] == base_review


def test_label_windows_live_on_the_clock_the_shift_leaves_alone():
    # A should_trigger window is t_rel from t_mono, in the review tool and in scoring;
    # the server shifts only t_wall, so a window set on a redacted copy fits the original.
    clip = _real_clock_clip()
    base = [t.t_rel for t in run_headless(clip)]
    for seconds in _SHIFTS_S:
        moved = _shifted(clip, seconds)
        assert clip_t0(moved) == clip_t0(clip)
        assert [t.t_rel for t in run_headless(moved)] == base


def test_lukas_labels_serialise_exactly_as_before_and_reviewer_labels_carry_their_handle():
    mine = Label(class_="tp", severity=3)
    assert "by" not in mine.to_json()
    theirs = Label(class_="fp", severity=2, by="r_aa")
    assert theirs.to_json()["by"] == "r_aa"
    assert Label.from_json(theirs.to_json()).by == "r_aa"
    assert Label.from_json(mine.to_json()).by == ""
