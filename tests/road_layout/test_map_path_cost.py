"""MapPath's faster nearest search and branch-trust walk give the old answers exactly."""
from __future__ import annotations

import math
import random

from core.road_layout import map_path as mp


def _plain_nearest(xs, ys, px, py) -> int:
    best_i, best_d = 0, math.inf
    for i in range(len(xs)):
        d = (xs[i] - px) ** 2 + (ys[i] - py) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def _path(xs, ys) -> mp.MapPath:
    n = len(xs)
    return mp.MapPath(tuple(2.5 * i for i in range(n)), tuple(xs), tuple(ys), (0.0,) * n, (1.0,) * n, 0.0, 15.0)


def test_the_nearest_sample_is_the_first_strict_minimum_as_before():
    rng = random.Random(2)
    for _ in range(400):
        n = rng.randint(2, 80)
        xs = [2.5 * i + rng.gauss(0.0, 0.3) for i in range(n)]
        ys = [rng.gauss(0.0, 2.0) for _ in range(n)]
        if rng.random() < 0.2:
            # Equal distances: two samples mirrored about the query.
            k = rng.randrange(n - 1)
            xs[k + 1], ys[k + 1] = xs[k], -ys[k]
        px, py = rng.uniform(-10.0, 2.5 * n + 10.0), 0.0 if rng.random() < 0.3 else rng.gauss(0.0, 4.0)
        path = _path(xs, ys)
        i = _plain_nearest(xs, ys, px, py)
        ds = [(x - px) ** 2 + (y - py) ** 2 for x, y in zip(xs, ys)]
        assert ds.index(min(ds)) == i
        assert path.coords(px, py) == _plain_coords(path, px, py)


def _plain_coords(path: mp.MapPath, px: float, py: float):
    best_i = _plain_nearest(path.x, path.y, px, py)
    n = len(path.s)
    found = None
    for i in (best_i - 1, best_i):
        if i < 0 or i >= n - 1:
            continue
        ax, ay = path.x[i], path.y[i]
        sx, sy = path.x[i + 1] - ax, path.y[i + 1] - ay
        length2 = sx * sx + sy * sy
        if length2 < 1e-12:
            continue
        raw = ((px - ax) * sx + (py - ay) * sy) / length2
        t = max(0.0, min(1.0, raw))
        d2 = (px - ax - t * sx) ** 2 + (py - ay - t * sy) ** 2
        if found is None or d2 < found[0]:
            side = ((px - ax) * (-sy) + (py - ay) * sx) / math.sqrt(length2)
            found = (d2, i, raw, path.s[i] + t * (path.s[i + 1] - path.s[i]), side)
    if found is None:
        return None
    _d2, i, raw, s_at, side = found
    if (i == 0 and raw < 0.0) or (i == n - 2 and raw > 1.0):
        return None
    return s_at, side


def test_a_nan_first_sample_still_finds_the_nearest_real_one():
    path = _path([math.nan, 2.5, 5.0, 7.5], [0.0, 0.0, 0.0, 0.0])
    assert path.coords(5.2, 0.4) == _plain_coords(path, 5.2, 0.4)
    assert path.coords(5.2, 0.4) is not None


def test_branch_trust_walked_once_matches_the_per_sample_scan():
    rng = random.Random(4)
    params = mp.DEFAULT_PARAMS
    for _ in range(200):
        div = sorted((rng.uniform(0.0, 150.0), rng.uniform(0.0, 3.0)) for _ in range(rng.randint(0, 12)))
        if rng.random() < 0.3:
            rng.shuffle(div)
        ss = [2.5 * i for i in range(rng.randint(0, 70))]
        plain = []
        for s in ss:
            widest = 0.0
            for ds, gap in div:
                if ds > s:
                    break
                widest = max(widest, gap)
            plain.append(1.0 - mp._smoothstep(widest / params.branch_zero_m))
        assert mp._branch_trusts(div, ss, params) == plain


def test_the_coarse_search_finds_the_same_sample_on_uneven_paths():
    rng = random.Random(6)
    for _ in range(600):
        n = rng.randint(13, 90)
        xs, ys, x, y = [], [], 0.0, 0.0
        for _ in range(n):
            # Uneven steps, sharp turns and repeated points: the gap bound must hold for all.
            step = rng.choice([0.0, 0.5, 2.5, 2.5, 6.0])
            ang = rng.uniform(-math.pi, math.pi) if rng.random() < 0.2 else 0.0
            x, y = x + step * math.cos(ang), y + step * math.sin(ang)
            xs.append(x)
            ys.append(y)
        path = _path(xs, ys)
        for _ in range(10):
            px, py = rng.uniform(-40.0, x + 40.0), rng.uniform(-40.0, 40.0)
            assert path._nearest(px, py) == _plain_nearest(xs, ys, px, py)
