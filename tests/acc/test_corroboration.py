"""Inter-vehicular corroboration for slow traffic: grouping and gating.

Members are given directly in road coordinates (arc length, offset from the
centreline, heading error against the road tangent), which is what the tracker
computes before calling in."""
from __future__ import annotations

import math

import pytest

from core.acc.corroboration import (
    ALIGN_TO_ROAD_DEG,
    FULL_MEMBERS,
    LANE_SPREAD_M,
    MAX_EVIDENCE,
    MIN_MEMBERS,
    MIN_SPAN_M,
    PARALLEL_SPREAD_DEG,
    UNANCHORED_MAX_EVIDENCE,
    SlowVehicle,
    corroborated_evidence,
)
from core.acc.tracker import VALIDATED_STATIONARY_EVIDENCE


def _queue(count, offset=0.0, spacing=8.0, heading_deg=0.0, first_id=1):
    return [
        SlowVehicle(
            vid=first_id + i,
            s_m=20.0 + i * spacing,
            offset_m=offset,
            heading_err_rad=math.radians(heading_deg),
        )
        for i in range(count)
    ]


def test_a_vehicle_on_its_own_earns_nothing():
    """The alone case must be left exactly as it was, not penalised."""
    assert corroborated_evidence(_queue(1), True) == {}


def test_below_the_minimum_earns_nothing():
    assert corroborated_evidence(_queue(MIN_MEMBERS - 1), True) == {}


def test_a_long_queue_corroborates_every_member():
    earned = corroborated_evidence(_queue(20), True)
    assert len(earned) == 20
    assert all(v == pytest.approx(MAX_EVIDENCE) for v in earned.values())


def test_corroboration_rises_with_the_count_and_saturates():
    values = [
        next(iter(corroborated_evidence(_queue(n), True).values()))
        for n in range(MIN_MEMBERS, FULL_MEMBERS + 3)
    ]
    assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))
    assert values[-1] == pytest.approx(MAX_EVIDENCE)
    assert values[0] < values[-1]


def test_corroboration_never_outranks_watching_it_drive():
    """Inference from other vehicles must not beat direct observation of this
    one, or the evidence hierarchy inverts."""
    earned = corroborated_evidence(_queue(50), True)
    assert max(earned.values()) <= VALIDATED_STATIONARY_EVIDENCE + 1e-9


def test_an_unanchored_line_is_trusted_less():
    """With no confident road model the only anchor is ego's own arc."""
    anchored = corroborated_evidence(_queue(20), True)
    loose = corroborated_evidence(_queue(20), False)
    assert max(loose.values()) == pytest.approx(UNANCHORED_MAX_EVIDENCE)
    assert max(loose.values()) < max(anchored.values())


def test_traffic_across_the_road_does_not_corroborate_a_queue():
    """A queue holds a lane, so members two lanes apart are separate lines."""
    mine = _queue(MIN_MEMBERS, offset=0.0, first_id=1)
    theirs = _queue(MIN_MEMBERS, offset=3.0 * LANE_SPREAD_M, first_id=100)
    earned = corroborated_evidence(mine + theirs, True)
    assert {v.vid for v in mine} <= set(earned)
    assert {v.vid for v in theirs} <= set(earned)
    # Each earned only what its own lane supports, not the combined count.
    alone = corroborated_evidence(mine, True)
    assert earned[mine[0].vid] == pytest.approx(alone[mine[0].vid])


def test_a_line_pointing_off_the_road_is_rejected():
    """Ego turning away from a queue: the queue no longer matches the tangent."""
    skew = ALIGN_TO_ROAD_DEG + 10.0
    assert corroborated_evidence(_queue(20, heading_deg=skew), True) == {}


def test_a_scattered_group_is_rejected():
    """Mutually parallel is the point: a car park is not a queue."""
    members = [
        SlowVehicle(vid=i, s_m=20.0 + i * 8.0, offset_m=0.0,
                    heading_err_rad=math.radians(-20.0 + 13.0 * i))
        for i in range(4)
    ]
    spread = math.degrees(
        max(m.heading_err_rad for m in members)
        - min(m.heading_err_rad for m in members)
    )
    assert spread > PARALLEL_SPREAD_DEG, "fixture must actually be scattered"
    assert corroborated_evidence(members, True) == {}


def test_a_bunched_group_is_not_a_line():
    """Three returns within a car length describe no direction at all."""
    tight = _queue(MIN_MEMBERS, spacing=MIN_SPAN_M / (MIN_MEMBERS * 4.0))
    assert corroborated_evidence(tight, True) == {}


def test_oncoming_stopped_traffic_is_not_a_source_of_support():
    """A stopped oncoming line sits at 180 deg to the tangent."""
    assert corroborated_evidence(_queue(20, heading_deg=180.0), True) == {}
