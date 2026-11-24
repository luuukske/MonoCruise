"""RoadPaths answers a repeated lane query from this frame's memo, with the same answer."""
from __future__ import annotations

import math
import random

from core.aeb import road_paths as rp
from core.road_layout.road_bend import road_curve


def _bound(rng: random.Random) -> rp.RoadPaths:
    pts = [(2.5 * i, 8.0 * math.sin(i / 15.0)) for i in range(80)]
    curve = road_curve(pts)
    paths = rp.RoadPaths()
    paths._lane = rp._frame(curve, curve.length, 10.0, 0.3, 20.0, 0.0)
    paths._keep = rp._frame(curve, 120.0, 10.0, 0.3, 20.0, 0.0) if rng.random() < 0.5 else paths._lane
    return paths


def test_a_repeated_lane_offset_is_the_uncached_answer():
    rng = random.Random(8)
    for _ in range(30):
        paths = _bound(rng)
        for _ in range(40):
            x, z = rng.uniform(-20.0, 220.0), rng.uniform(-12.0, 12.0)
            for keep in (False, True, False, True):
                frame = paths._keep if keep else paths._lane
                assert paths.lane_offset(x, z, keep) == rp._lane_offset(frame, x, z)


def test_a_new_lane_frame_is_never_answered_from_the_old_one():
    rng = random.Random(1)
    paths = _bound(rng)
    first = paths.lane_offset(60.0, 2.0)
    pts = [(2.5 * i, 3.0) for i in range(80)]
    curve = road_curve(pts)
    paths._lane = rp._frame(curve, curve.length, 10.0, 3.0, 20.0, 0.0)
    assert paths.lane_offset(60.0, 2.0) == rp._lane_offset(paths._lane, 60.0, 2.0)
    assert paths.lane_offset(60.0, 2.0) != first
