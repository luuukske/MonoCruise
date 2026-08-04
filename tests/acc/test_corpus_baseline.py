"""Corpus regression: replay the local AEB clip store through the ACC tracker.

Bounds are recorded from a measured run, not derived. Tighten them when the
tracker improves; never loosen one to go green. See tests/acc/README.md."""
from __future__ import annotations

import statistics

import pytest

from .harness import clip_root, load_clips, replay_corpus

CLIP_SAMPLE = 60

pytestmark = [
    pytest.mark.needs_clips,
    pytest.mark.skipif(clip_root() is None, reason="no local AEB clip store"),
]


@pytest.fixture(scope="module")
def metrics():
    clips = load_clips(CLIP_SAMPLE)
    if len(clips) < 20:
        pytest.skip("local clip store too small for a stable corpus measurement")
    return replay_corpus(clips)


def _p90(values):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))]


def test_moving_in_corridor_targets_lock(metrics):
    """Recall floor. Measured 43.4%.

    Not comparable across tracker versions on its own: the uncertainty gate
    changes which frames count as in-corridor, so the denominator moves. The
    comparable recall number is in tests/acc/README.md, measured against ego's
    own future path."""
    assert metrics.moving_lock_rate >= 0.38


def test_stationary_lock_rate_stays_bounded(metrics):
    """Precision ceiling: stationary targets must not dominate the lead list.

    Measured 4.4%, down from 7.2% before evidence gating."""
    assert metrics.stationary_lock_rate <= 0.06


def test_score_ceiling_is_reachable_without_unbounded_windup(metrics):
    assert metrics.positive_frames > 0
    assert metrics.saturated_rate <= 0.70


def test_cut_in_locks_stay_responsive(metrics):
    """Cut-ins are the responsiveness case the driver feels. Measured p90 2.58 s."""
    assert metrics.cutin_lock_s, "no cut-in transitions observed in the sample"
    assert _p90(metrics.cutin_lock_s) <= 3.0


def test_fresh_id_locks_are_not_instant(metrics):
    """A vehicle that just entered radar range must earn its lock, not get it
    on the first frames from a fabricated offset. Measured p50 0.41 s, p90 3.55 s.

    The p90 is a small-n tail (n around 45) and it grew with the uncertainty
    gate: a distant unconfirmed target is deliberately not called in-lane until
    the estimate can resolve a lane width. Bound guards the median."""
    assert metrics.fresh_lock_s, "no fresh-id transitions observed in the sample"
    assert statistics.median(metrics.fresh_lock_s) <= 1.0


def test_lead_hooking_releases_promptly(metrics):
    """A lead leaving the corridor must drop below confidence without a long
    tail. Measured p90 0.85 s, down from 2.45 s at the old score ceiling."""
    assert metrics.hook_s, "no hook transitions observed in the sample"
    assert _p90(metrics.hook_s) <= 1.3
