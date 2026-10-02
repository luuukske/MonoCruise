"""Trail evidence gating: absence of a fit must never read as high confidence.

The pre-fix tracker set ``angle_amp = 1.0`` on both fit-failure paths, so a
stationary target with no trail scored better than a slow one with a noisy fit.
These tests pin the corrected ordering."""
from __future__ import annotations

import math

import pytest

from core.acc.scoring import (
    OFFSET_BASELINE_HIT, OFFSET_BASELINE_NO_ARC_HIT, OFFSET_BASELINE_NO_HISTORY,
    SCORE_MAX, SCORE_MIN, accumulate, offset_component, ScoreComponents,
)
from core.acc.trail_arc import (
    EVIDENCE_CHORD_FULL_M, EVIDENCE_CHORD_MIN_M, trail_evidence,
)
from core.cruise_control_thread.acc_controller import ANT_SCORE_FULL, ANT_SCORE_MIN


def test_evidence_ramps_with_chord_length():
    assert trail_evidence(0.0) == 0.0
    assert trail_evidence(EVIDENCE_CHORD_MIN_M) == 0.0
    assert trail_evidence(EVIDENCE_CHORD_FULL_M) == 1.0
    assert trail_evidence(EVIDENCE_CHORD_FULL_M * 3) == 1.0
    mid = trail_evidence((EVIDENCE_CHORD_MIN_M + EVIDENCE_CHORD_FULL_M) / 2)
    assert 0.4 < mid < 0.6


def test_evidence_is_monotonic():
    chords = [0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 40.0]
    values = [trail_evidence(c) for c in chords]
    assert values == sorted(values)


def test_zero_evidence_contributes_nothing():
    """No usable trail must leave the corridor geometry to decide alone."""
    for offset in (0.0, 1.0, 3.5, 10.0):
        assert offset_component(offset, 30.0, angle_amp=1.0, evidence=0.0) == 0.0


def test_missing_trail_never_outscores_a_real_fit():
    """The core inversion: absent evidence must not beat a slow in-lane target.

    A slow target spans little ground, so its measured arrival angle is
    noise-dominated. It must still read better than a target with no trail."""
    span = EVIDENCE_CHORD_FULL_M - EVIDENCE_CHORD_MIN_M
    slow_evidence = trail_evidence(EVIDENCE_CHORD_MIN_M + 0.3 * span)
    assert 0.0 < slow_evidence < 0.5
    slow_in_lane = offset_component(
        0.0, 30.0, angle_amp=0.09, baseline=OFFSET_BASELINE_HIT,
        evidence=slow_evidence,
    )
    no_trail = offset_component(
        0.0, 30.0, angle_amp=1.0, baseline=OFFSET_BASELINE_NO_HISTORY, evidence=0.0,
    )
    assert slow_in_lane > no_trail


def test_confident_steep_arrival_stays_a_release_signal():
    """A well-measured crossing angle must go negative: that is the cut-out cue."""
    confident_crossing = offset_component(
        0.0, 30.0, angle_amp=0.09, baseline=OFFSET_BASELINE_HIT, evidence=1.0,
    )
    assert confident_crossing < -0.5


def test_good_fit_still_dominates():
    """A clean centred fit must remain the strongest positive offset signal."""
    good = offset_component(
        0.0, 30.0, angle_amp=1.0, baseline=OFFSET_BASELINE_HIT, evidence=1.0,
    )
    assert good > offset_component(
        0.0, 30.0, angle_amp=1.0, baseline=OFFSET_BASELINE_HIT, evidence=0.5,
    )
    assert good > offset_component(
        3.5, 30.0, angle_amp=1.0, baseline=OFFSET_BASELINE_HIT, evidence=1.0,
    )


def test_no_arc_hit_stays_a_negative_signal_scaled_by_evidence():
    """A confident fit that misses the ego row is evidence of leaving the lane."""
    strong = offset_component(
        3.5, 30.0, angle_amp=1.0, baseline=OFFSET_BASELINE_NO_ARC_HIT, evidence=1.0,
    )
    weak = offset_component(
        3.5, 30.0, angle_amp=1.0, baseline=OFFSET_BASELINE_NO_ARC_HIT, evidence=0.2,
    )
    assert strong < weak < 0.0


@pytest.mark.parametrize("evidence", [0.0, 0.25, 0.5, 1.0])
def test_offset_component_is_finite_and_bounded(evidence):
    value = offset_component(2.0, 45.0, angle_amp=0.3, evidence=evidence)
    assert math.isfinite(value)
    assert -2.0 <= value <= 2.0


def test_score_ceiling_matches_consumer_confidence_saturation():
    """The clamp must reach full confidence without banking unusable score.

    Score above the consumer's saturation point does nothing except delay
    release, which is what produced multi-second lead hooking."""
    assert SCORE_MAX >= ANT_SCORE_FULL
    assert SCORE_MAX <= 2.0 * ANT_SCORE_FULL
    assert ANT_SCORE_MIN < ANT_SCORE_FULL


def test_accumulate_respects_the_clamp():
    comps = ScoreComponents(offset=1.5, yaw=0.0, path=5.0, angle=0.0)
    score = 0.0
    for _ in range(200):
        score = accumulate(score, 1.0 / 30.0, comps, 25.0)
    assert score == pytest.approx(SCORE_MAX)

    falling = ScoreComponents(offset=-1.5, yaw=-1.5, path=-4.0, angle=0.0)
    for _ in range(200):
        score = accumulate(score, 1.0 / 30.0, falling, 25.0)
    assert score == pytest.approx(SCORE_MIN)
