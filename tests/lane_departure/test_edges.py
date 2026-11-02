"""What lies past each line of ego's lane: the next lane, an oncoming lane or the road's edge, from the map."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.lane_departure.edges import EDGE, LANE, ONCOMING, UNKNOWN, beyond, point_at
from core.road_layout.ego_lane import LineIndex


def _lane(x: float, travel: int, reverse: bool = False):
    """A straight lane along +z at ``x``; ``travel`` +1 runs in point order."""
    pts = tuple((x, z) for z in range(0, 401, 10))
    pts = pts[::-1] if reverse else pts
    return SimpleNamespace(pts=pts, travel=travel, min_x=x, max_x=x, min_z=0.0, max_z=400.0)


def _window(*lanes):
    return SimpleNamespace(lanes=lanes, index=LineIndex(lanes))


ROUTE = tuple((0.0, z) for z in range(100, 301, 10))


def test_a_point_along_the_route_carries_its_tangent():
    assert point_at(ROUTE, 25.0) == pytest.approx((0.0, 125.0, 0.0, 1.0))
    assert point_at(ROUTE, 1e6) == pytest.approx((0.0, 300.0, 0.0, 1.0)), "clamped to the end"
    assert point_at(((1.0, 1.0),), 5.0) is None


def test_a_lane_beside_with_traffic_our_way_is_a_lane_the_other_way_oncoming():
    # Travelling +z, right is -x (right = (-uz, ux)).
    window = _window(_lane(0.0, 1), _lane(-4.5, 1), _lane(4.5, 1, reverse=True))
    assert beyond(window, ROUTE, 20.0, 1.0) == LANE
    assert beyond(window, ROUTE, 20.0, -1.0) == ONCOMING


def test_a_lane_that_runs_against_point_order_counts_its_travel():
    window = _window(_lane(0.0, 1), _lane(4.5, -1))
    assert beyond(window, ROUTE, 20.0, -1.0) == ONCOMING
    window = _window(_lane(0.0, 1), _lane(4.5, -1, reverse=True))
    assert beyond(window, ROUTE, 20.0, -1.0) == LANE


def test_no_lane_past_a_line_is_the_road_edge_and_a_far_carriageway_is_too():
    window = _window(_lane(0.0, 1), _lane(14.0, 1, reverse=True))
    assert beyond(window, ROUTE, 20.0, 1.0) == EDGE
    assert beyond(window, ROUTE, 20.0, -1.0) == EDGE, "the other carriageway 14 m off is past a median"


def test_unknown_travel_or_no_map_says_so():
    assert beyond(_window(_lane(0.0, 1), _lane(-4.5, 0)), ROUTE, 20.0, 1.0) == UNKNOWN
    assert beyond(None, ROUTE, 20.0, 1.0) == UNKNOWN
