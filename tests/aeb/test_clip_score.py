"""Tests for severity-weighted corpus scoring."""

from __future__ import annotations

from dataclasses import replace

from core.aeb.calibration import DEFAULT
from core.aeb.clip_eval import run_headless
from core.aeb.clip_schema import Label
from core.aeb.clip_score import (
    SAFETY_SCALE, audit_corpus, class_window_warning, score_clip, score_corpus,
)

from tests.aeb.test_clip_eval import _collision_clip

_OFF = replace(DEFAULT, aeb_min_engage_speed_kmh=1000.0)


def _tp_labelled():
    clip = _collision_clip()
    bt = [e.t_rel for e in run_headless(clip) if e.aeb_brake]
    clip.metadata.label = Label(class_="tp", severity=4,
                                should_trigger={"from_t": bt[0], "to_t": bt[-1]})
    return clip


def test_false_positive_costs_comfort_true_negative_free():
    clip = _collision_clip()
    clip.metadata.label = Label(class_="fp", severity=2, should_trigger=None)
    fp = score_clip(clip, DEFAULT, burn_in_s=0.0)
    tn = score_clip(clip, _OFF, burn_in_s=0.0)
    assert fp.verdict == "false_positive" and fp.cost > 0
    assert tn.verdict == "true_negative" and tn.cost == 0.0


def test_false_negative_pays_safety_scale():
    clip = _tp_labelled()
    miss = score_clip(clip, _OFF, burn_in_s=0.0)
    assert miss.verdict == "false_negative"
    assert miss.cost == 4 * SAFETY_SCALE


def test_true_positive_is_a_reward():
    clip = _tp_labelled()
    tp = score_clip(clip, DEFAULT, burn_in_s=0.0)
    assert tp.verdict == "true_positive"
    assert tp.cost < 0.0


def test_safety_dominates_comfort():
    # Same severity: a missed brake must cost more than a phantom brake.
    fn_clip = _tp_labelled()
    fn_clip.metadata.label = replace(fn_clip.metadata.label, severity=3)
    fp_clip = _collision_clip()
    fp_clip.metadata.label = Label(class_="fp", severity=3, should_trigger=None)
    fn_cost = score_clip(fn_clip, _OFF, burn_in_s=0.0).cost
    fp_cost = score_clip(fp_clip, DEFAULT, burn_in_s=0.0).cost
    assert fn_cost > fp_cost


def test_corpus_aggregates_and_excludes_ignore():
    good = _tp_labelled()
    bad = _collision_clip()
    bad.metadata.label = Label(class_="fp", severity=2, should_trigger=None)
    ignored = _collision_clip()
    ignored.metadata.label = Label(class_="ignore", severity=1)
    unlabelled = _collision_clip()

    cs = score_corpus([good, bad, ignored, unlabelled], DEFAULT, burn_in_s=0.0)
    assert cs.scored == 2                      # ignore + unlabelled dropped
    assert set(cs.verdict_counts) == {"true_positive", "false_positive"}
    assert abs(cs.total_cost - sum(p.cost for p in cs.per_clip)) < 1e-9


def test_label_consistency_rule():
    assert class_window_warning("fp", True) is not None       # fp must not have a window
    assert class_window_warning("fp", False) is None
    assert class_window_warning("tp", False) is not None       # tp needs a window
    assert class_window_warning("tp", True) is None
    assert class_window_warning("fn", False) is not None
    assert class_window_warning("ignore", True) is None


def test_audit_corpus_flags_class_window_mismatch():
    good = _tp_labelled()                                       # tp + window: consistent
    bad = _collision_clip()
    bad.metadata.label = Label(class_="fp", severity=1,
                               should_trigger={"from_t": 1.0, "to_t": 2.0})
    audit = audit_corpus([good, bad])
    assert [c for c, _ in audit] == [bad.metadata.clip_id]


def test_candidate_lowers_cost_on_false_positive_corpus():
    clip = _collision_clip()
    clip.metadata.label = Label(class_="fp", severity=3, should_trigger=None)
    base = score_corpus([clip], DEFAULT, burn_in_s=0.0)
    cand = score_corpus([clip], _OFF, burn_in_s=0.0)
    assert cand.total_cost < base.total_cost   # removing the FP improves the score
