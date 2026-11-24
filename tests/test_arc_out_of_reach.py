"""The arc collision's out-of-reach skip never changes an answer (core/radar/README.md §8)."""
from __future__ import annotations

import math
import random

import pytest

import core.radar.traffic as traffic
from core.radar.bent_path import bend_arc
from core.radar.traffic import arc_arc_collision, build_arc


def _arc(rng: random.Random, x: float, z: float):
    arc = build_arc(
        x, z, rng.uniform(-math.pi, math.pi), rng.choice([0.0, rng.uniform(-3.0, 35.0)]),
        rng.choice([0.0, rng.uniform(-0.002, 0.002), rng.uniform(-0.08, 0.08)]),
        rng.uniform(0.9, 1.5), rng.uniform(2.0, 6.0),
        decel=rng.choice([0.0, 0.0, rng.uniform(0.5, 8.0)]),
        accel=rng.choice([0.0, 0.0, rng.uniform(-2.0, 2.0)]),
        fwd_len=rng.choice([0.0, rng.uniform(1.0, 9.0)]),
        back_len=rng.choice([0.0, rng.uniform(1.0, 16.0)]),
        parallel_margin_scale=rng.choice([1.0, 0.3]),
    )
    if rng.random() < 0.3 and arc.speed > 1.0:
        k0 = arc.curvature
        arc = bend_arc(arc, (0.0, 20.0, 60.0), (k0, rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)))
    return arc


def _pairs(seed: int, n: int):
    rng = random.Random(seed)
    for _ in range(n):
        a = _arc(rng, 0.0, 0.0)
        if rng.random() < 0.5:
            # Beside a's own path, where the hits are.
            px, pz = a.position_at_dist(rng.uniform(0.0, a.arc_length))
            b = _arc(rng, px + rng.uniform(-12.0, 12.0), pz + rng.uniform(-12.0, 12.0))
        else:
            # Toward the reach boundary, where a wrong bound would show.
            reach = a.arc_length + rng.uniform(0.0, 260.0)
            ang = rng.uniform(-math.pi, math.pi)
            b = _arc(rng, reach * math.cos(ang), reach * math.sin(ang))
        yield a, b, rng.choice([0.0, 0.3, 0.5, 1.0]), rng.choice([0.0, 0.0, 1.5])


@pytest.mark.parametrize("seed", range(4))
def test_skipping_out_of_reach_pairs_gives_the_same_answer(seed, monkeypatch):
    pairs = list(_pairs(seed, 1500))
    fast = [arc_arc_collision(a, b, m, 24, g) for a, b, m, g in pairs]
    skipped = sum(
        traffic._out_of_reach(a, b, a.half_width + b.half_width + m + 1.0) for a, b, m, _ in pairs)
    monkeypatch.setattr(traffic, "_out_of_reach", lambda *_a: False)
    slow = [arc_arc_collision(a, b, m, 24, g) for a, b, m, g in pairs]
    assert fast == slow
    assert skipped > 100, "the sample must exercise the skip"
    assert sum(h is not None for h in slow) > 50, "and still contain hits"


def test_a_body_at_the_edge_of_reach_still_hits():
    ego = build_arc(0.0, 0.0, 0.0, 20.0, 0.0, 1.2, 3.0, fwd_len=8.0, back_len=8.0)
    # Ego runs 60 m toward -z; a stopped car whose front sits 1 m past where ego's front ends.
    car = build_arc(0.0, -(60.0 + 6.8 + 2.0), 0.0, 0.0, 0.0, 1.0, 3.0, fwd_len=2.0, back_len=2.0)
    assert not traffic._out_of_reach(ego, car, ego.half_width + car.half_width + 0.5)
    assert arc_arc_collision(ego, car, 0.5, 24) is not None
