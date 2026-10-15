"""Agent clip-analysis tooling: hindsight geometry, lane ghost, dossier header.

Synthetic inputs only, so these run in CI without clips or a game install.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from tools.aeb_agent.context import classify, cumulative, project
from tools.aeb_agent.digest import _header
from tools.aeb_agent.features import ClipFeatures, world_to_ego
from tools.aeb_agent.lane_ghost import follow, grade

def _straight(n: int = 20, step: float = 10.0):
    # Ego frame convention: yaw 0 faces -z, and +x is to its right.
    return [(0.0, -i * step) for i in range(n)]


def test_project_gives_arc_length_and_right_offset():
    pts = _straight()
    s, d, _i = project(pts, cumulative(pts), 1.5, -42.0)
    assert s == pytest.approx(42.0)
    assert d == pytest.approx(1.5)


def test_project_right_offset_agrees_with_world_to_ego():
    pts = _straight()
    _s, d, _i = project(pts, cumulative(pts), -2.0, -30.0)
    _fwd, right = world_to_ego(-2.0, -30.0, 0.0, 0.0, 0.0)
    assert math.copysign(1, d) == math.copysign(1, right)


def test_project_extends_past_the_end():
    pts = _straight(3)
    s, _d, _i = project(pts, cumulative(pts), 0.0, -35.0)
    assert s == pytest.approx(35.0)


@pytest.mark.parametrize("d, dot, moving, rel", [
    (0.4, 1.0, True, "ego_lane"),
    (0.4, 1.0, False, "ego_lane"),
    (0.4, -1.0, True, "ego_lane_wrong_way"),
    (0.4, 0.0, True, "across_ego_lane"),
    (3.8, 1.0, True, "adjacent_right"),
    (-3.8, 1.0, True, "adjacent_left"),
    (-4.5, -1.0, True, "oncoming_left"),
    (12.0, 0.1, True, "crossing_right"),
    (25.0, 0.1, False, "beside_road_right"),
])
def test_classify(d, dot, moving, rel):
    assert classify(d, dot, moving) == rel


def test_agent_header_drops_capture_identity():
    row = SimpleNamespace(clip_id="c" * 36, origin="remote", path="x/20260101T000000Z_a.json.gz",
                          captured_at="2026-01-01T00:00:00Z", client_version="1.1.0",
                          session_kind="TMP", trigger_source="auto_engagement",
                          brake_reached=True)
    text = "\n".join(_header(row, ClipFeatures()))
    assert "captured" not in text and "client" not in text and "20260101" not in text
    assert "captured" in "\n".join(_header(row, ClipFeatures(), full=True))


def _route(offset=0.4, heading=0.0, branch=None):
    pts = tuple(_straight())
    return SimpleNamespace(pts=pts, offset_m=offset, heading_err_rad=heading,
                           branch_s=branch, reach_m=cumulative(pts)[-1])


def test_lane_ghost_follows_the_route_at_the_held_offset():
    times = [i * 0.1 for i in range(31)]
    g = follow(_route(), 0.4, 0.0, times, [10.0] * len(times))
    assert g.ok
    t, x, z, yaw = g.poses[-1]
    assert z == pytest.approx(-30.0, abs=1e-6)
    assert x == pytest.approx(0.4, abs=1e-6)
    assert yaw == pytest.approx(0.0, abs=1e-9)


def test_lane_ghost_refuses_when_ego_is_off_its_lane():
    times = [i * 0.1 for i in range(5)]
    assert not follow(_route(offset=2.0), 2.0, 0.0, times, [10.0] * 5).ok


def test_lane_ghost_stops_at_a_branch():
    times = [i * 0.1 for i in range(31)]
    g = follow(_route(branch=15.0), 0.0, 0.0, times, [10.0] * len(times))
    assert g.stops_at_branch and g.poses[-1][2] >= -15.0


def test_lane_grade_bands():
    base = {"ran": True, "mode": "lane_recorded_speed", "dt_at_min_s": 1.0}
    assert grade({**base, "min_separation_m": -3.0})["verdict"] == "collides"
    assert grade({**base, "min_separation_m": -0.2})["verdict"] == "likely"
    assert grade({**base, "min_separation_m": 0.5})["verdict"] == "close"
    assert grade({**base, "min_separation_m": 5.0})["verdict"] == "clear"
    assert grade({**base, "min_separation_m": -1.0, "dt_at_min_s": 0.1})["verdict"] == "degenerate"

