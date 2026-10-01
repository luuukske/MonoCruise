"""Invariant: the winner label says who is bidding, not whose bid won the min().

SendingThread merges on the label: "cc" is a max-merge (user gas may exceed the
mapper), "limiter" is a min-merge (hard cap on the user pedal). Labelling a tick
"limiter" just because the limiter's bid owns the arbitration min zeroes CC's
gas with the foot off the pedal, and the output jitters every time the bids
cross (set speed == global limit).
"""
from __future__ import annotations

import pytest

from core.cruise_control_thread.thread import CruiseControlThread
from core.longitudinal.base import LongOutput

arbitrate = CruiseControlThread._arbitrate_named


def test_no_active_bidder_reports_none():
    bid, commanding, winner = arbitrate(
        ("cc", LongOutput(None, False)),
        ("limiter", LongOutput(None, False)),
        ("acc", LongOutput(None, False)),
    )
    assert (bid, commanding, winner) == (0.0, False, "none")


def test_sole_limiter_bidder_is_labelled_limiter():
    bid, commanding, winner = arbitrate(
        ("cc", LongOutput(None, False)),
        ("limiter", LongOutput(-0.4, True)),
        ("acc", LongOutput(None, False)),
    )
    assert commanding is True
    assert winner == "limiter"
    assert bid == -0.4


def test_limiter_winning_the_min_against_cc_stays_labelled_cc():
    """The regression: limiter bid is lower, but CC is still bidding."""
    bid, _, winner = arbitrate(
        ("cc", LongOutput(0.8, True)),
        ("limiter", LongOutput(-0.2, True)),
        ("acc", LongOutput(None, False)),
    )
    assert winner == "cc"
    assert bid == -0.2, "arbitration still takes the min; only the label differs"


def test_acc_counts_as_the_cc_side():
    _, _, winner = arbitrate(
        ("cc", LongOutput(None, False)),
        ("limiter", LongOutput(-0.2, True)),
        ("acc", LongOutput(0.1, True)),
    )
    assert winner == "cc"


def test_label_is_stable_across_the_bid_crossover():
    """Set speed == global limit: bids cross tick to tick, the label must not."""
    labels = []
    for limiter_bid in (0.9, 0.5, 0.1, -0.1, -0.5):
        _, _, winner = arbitrate(
            ("cc", LongOutput(0.3, True)),
            ("limiter", LongOutput(limiter_bid, True)),
            ("acc", LongOutput(None, False)),
        )
        labels.append(winner)
    assert labels == ["cc"] * 5, f"winner flipped across the crossover: {labels}"


@pytest.mark.parametrize("bids, expected", [
    ((-1.0, -0.5), -1.0),
    ((0.2, 1.4), 0.2),
])
def test_arbitration_takes_the_minimum_bid(bids, expected):
    cc_bid, limiter_bid = bids
    bid, _, _ = arbitrate(
        ("cc", LongOutput(cc_bid, True)),
        ("limiter", LongOutput(limiter_bid, True)),
    )
    assert bid == expected


def test_inactive_bids_are_ignored_even_when_lower():
    bid, _, winner = arbitrate(
        ("cc", LongOutput(0.5, True)),
        ("limiter", LongOutput(-9.0, False)),
    )
    assert bid == 0.5
    assert winner == "cc"
