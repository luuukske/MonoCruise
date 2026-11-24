"""Route divergence: the memoized branches and culled distances give exactly what the plain measure gave."""
from __future__ import annotations

import math
import random

import core.road_layout.ego_route as er
from core.road_layout.ego_lane import LineIndex, extend_lane, join_candidates
from core.road_layout.game_map import LaneWindow, pack_lane
from core.road_layout.lanes import LaneLine


def _line(pts, kind="road", uid=1, lane_index=0, prefab_id=None):
    return LaneLine(kind, "test", tuple((x, 0.0, z) for x, z in pts), prefab_id, uid, lane_index, 2, 0.0)


def _junctions() -> LaneWindow:
    """A road east with a junction every 60 m, each with a straight-on curve and two turns."""
    lines = [_line([(float(x), 0.0) for x in range(-100, 1, 10)], uid=1)]
    for j in range(5):
        x0 = 60.0 * j
        pid = 100 + j
        lines.append(_line([(x0 + 2.0 * k, 0.0) for k in range(31)], kind="prefab", uid=pid, prefab_id=pid))
        for side, lane_index in ((1.0, 1), (-1.0, 2)):
            turn = [(x0 + 40.0 * math.sin(a / 12 * math.pi / 2), side * 40.0 * (1 - math.cos(a / 12 * math.pi / 2)))
                    for a in range(13)]
            lines.append(_line(turn, kind="prefab", uid=pid, lane_index=lane_index, prefab_id=pid))
            lines.append(_line([(turn[-1][0] + 0.0, turn[-1][1] + side * 10.0 * k) for k in range(25)], uid=200 + j * 2 + lane_index))
    lines.append(_line([(300.0 + 10.0 * k, 0.0) for k in range(30)], uid=2))
    lanes = tuple(pack_lane(line) for line in lines)
    return LaneWindow(tuple(lines), (), (0, 0, 0, 0), lanes, LineIndex(lanes))


def _plain_branches(lanes, near, spans, route, boundaries):
    """The branches as they were found: every one rebuilt, no memo, plain walks."""
    seen = {span.index for span in spans}
    branches = []
    for s_at, k in boundaries:
        prev_index, next_index = spans[k - 1].index, spans[k].index
        tip = er._route_point(route, s_at)
        tip_dir = er._route_dir(route, s_at)
        if tip is None or tip_dir is None:
            continue
        for _cost, cand, _end in join_candidates(near, tip, tip_dir, {prev_index, next_index}):
            if cand in seen:
                continue
            pts = er._join(extend_lane(lanes, cand, tip[0], tip[1], tip_dir[0], tip_dir[1], near), tip)
            if len(pts) >= 2:
                branches.append((s_at, pts))
    return branches, [(s_at, (pts, ())) for s_at, pts in branches], None


def _plain_divergence(route, _hops, shapes):
    """The measure as it was: every branch measured over all its segments."""
    out, first, s = [], None, 0.0
    total = er._length(route)
    while s <= total + 1e-6:
        widest = 0.0
        p = er._route_point(route, s)
        for s_at, (pts, _boxes) in shapes:
            if s < s_at or p is None or s > s_at + er._ALT_REACH_M:
                continue
            widest = max(widest, er._distance_to(pts, p[0], p[1]))
        if widest > 0.5 and first is None:
            first = s
        out.append((s, widest))
        s += er._DIV_STEP_M
    return tuple(out), first


def _poses(n: int):
    rng = random.Random(3)
    for _ in range(n):
        yield rng.uniform(-90.0, 290.0), rng.uniform(-2.0, 2.0), 1.0, rng.uniform(-0.05, 0.05)


def test_divergence_matches_the_plain_measure_bit_for_bit(monkeypatch):
    window = _junctions()
    fast = [er.build_route(window.lanes, window.index, 0, x, z, dx, dz) for x, z, dx, dz in _poses(60)]
    assert any(r is not None and r.branch_s is not None for r in fast), "the fixture has branches"
    assert window.index.branch_cache, "tips were memoized"
    monkeypatch.setattr(er, "_branches", _plain_branches)
    monkeypatch.setattr(er, "_divergence", _plain_divergence)
    fresh = _junctions()
    plain = [er.build_route(fresh.lanes, fresh.index, 0, x, z, dx, dz) for x, z, dx, dz in _poses(60)]
    assert fast == plain


def test_a_memoized_tip_returns_the_same_branches_as_a_cold_one():
    window = _junctions()
    route = er.build_route(window.lanes, window.index, 0, -50.0, 0.3, 1.0, 0.0)
    again = er.build_route(window.lanes, window.index, 0, -50.0, 0.3, 1.0, 0.0)
    cold = _junctions()
    assert route == again == er.build_route(cold.lanes, cold.index, 0, -50.0, 0.3, 1.0, 0.0)


def test_lanes_from_another_window_never_read_its_cache():
    window = _junctions()
    other = list(window.lanes)
    er.build_route(other, window.index, 0, -50.0, 0.3, 1.0, 0.0)
    assert not window.index.branch_cache


def test_walks_on_measured_lengths_equal_the_plain_walks():
    rng = random.Random(5)
    for n in (0, 1, 2, 3, 40):
        pts = tuple((10.0 * k + rng.uniform(-3, 3), rng.uniform(-20, 20)) for k in range(n))
        if n == 3:
            pts = (pts[0], pts[0], pts[2])
        hops = er._hops(pts)
        for s in [rng.uniform(-5, 500) for _ in range(200)] + [0.0, 1e-12]:
            assert er._point_at(pts, hops, s) == er._route_point(pts, s)
            assert er._dir_at(pts, hops, s) == er._route_dir(pts, s)


def test_route_points_are_the_route_without_its_divergence():
    window = _junctions()
    for x, z, dx, dz in _poses(60):
        full = er.build_route(window.lanes, window.index, 0, x, z, dx, dz)
        pts = er.route_points(window.lanes, window.index, 0, x, z, dx, dz)
        assert pts == (full.pts if full is not None else None)


def _consumed(route):
    """Every field AEB reads from a target route: all but ``divergence`` and ``branch_s``."""
    if route is None:
        return None
    return (route.pts, route.offset_m, route.heading_err_rad, route.lane_index, route.kind, route.lane_count,
            route.reach_m, route.alternatives, route.merge_s, route.merge_offset_m, route.merge_index)


def test_a_route_without_divergence_is_the_same_route_for_what_reads_it():
    window = _junctions()
    full = er.EgoRouter()
    bare = er.EgoRouter(divergence=False)
    seen_branch = False
    for x, z, dx, dz in _poses(60):
        a = full.step(window, x, 0.0, z, dx, dz)
        b = bare.step(window, x, 0.0, z, dx, dz)
        assert _consumed(a) == _consumed(b)
        if b is not None:
            assert b.divergence == () and b.branch_s is None
            seen_branch = seen_branch or a.branch_s is not None
    assert seen_branch, "the fixture has routes with branches"


def test_targets_are_routed_without_divergence():
    from core.road_layout.target_path import TargetRouter

    window = _junctions()
    route = TargetRouter().route(1, window, -50.0, 0.0, 0.3, 1.0, 0.0)
    assert route is not None and route.alternatives and route.divergence == ()
    kept = TargetRouter(divergence=True).route(1, window, -50.0, 0.0, 0.3, 1.0, 0.0)
    assert kept.divergence and _consumed(kept) == _consumed(route)


def test_culled_distance_equals_the_plain_distance():
    rng = random.Random(11)
    pts = tuple((10.0 * k + rng.uniform(-3, 3), rng.uniform(-20, 20)) for k in range(57))
    shape = er._shape(pts)
    for _ in range(500):
        x, z = rng.uniform(-50, 600), rng.uniform(-80, 80)
        assert er._distance_past(shape, x, z, 0.0) == er._distance_to(pts, x, z)
        floor = rng.uniform(0, 60)
        got = er._distance_past(shape, x, z, floor)
        assert got == er._distance_to(pts, x, z) or got <= floor
