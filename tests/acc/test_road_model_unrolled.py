"""The road model's unrolled loops add in the old order, so the fit is bit-identical (README §9, Cost)."""
from __future__ import annotations

import math
import random

from core.acc import road_lsq
from core.acc import road_model as rm


def _rows(seed: int, n: int = 300, zero_weights: bool = False):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        basis = rm._basis(rng.uniform(-30.0, 170.0))
        centred = tuple(b - rng.uniform(-0.5, 0.5) * b for b in basis)
        w = 0.0 if zero_weights and rng.random() < 0.2 else rng.choice([1.0, rng.uniform(0.01, 2.0)])
        rows.append((rng.randint(-1, 8), centred, rng.gauss(0.0, 1.0), w))
    return rows


def _plain_normal_equations(rows):
    ata = [[0.0] * 4 for _ in range(4)]
    atb = [0.0] * 4
    for _, basis, y, w in rows:
        for i in range(4):
            atb[i] += w * basis[i] * y
            for j in range(4):
                ata[i][j] += w * basis[i] * basis[j]
    return ata, atb


def _plain_basis(s_m: float):
    v = s_m / rm.S_REF_M
    out = []
    power = 1.0
    for _ in range(4):
        power *= v
        out.append(power)
    return tuple(out)


def test_the_basis_is_the_old_power_loop():
    rng = random.Random(3)
    for s in [0.0, -0.0, 1e-9, 100.0] + [rng.uniform(-200.0, 400.0) for _ in range(500)]:
        assert repr(rm._basis(s)) == repr(_plain_basis(s))


def test_the_normal_equations_are_the_old_double_loop_bit_for_bit():
    for seed in range(20):
        rows = _rows(seed)
        assert repr(rm._normal_equations(rows)) == repr(_plain_normal_equations(rows))


def test_residuals_are_y_minus_predict_bit_for_bit():
    rows = _rows(5)
    beta = (0.3, -1.2, 2.5, -0.7)
    plain = [y - sum(b * c for b, c in zip(basis, beta)) for _, basis, y, _ in rows]
    assert repr(rm._residuals(rows, beta)) == repr(plain)
    assert repr([road_lsq.predict(beta, basis) for _, basis, _, _ in rows]) == repr(
        [sum(b * c for b, c in zip(basis, beta)) for _, basis, _, _ in rows])


def test_the_covariance_from_the_last_solve_matches_a_fresh_one():
    for zero in (False, True):
        rows = _rows(9, zero_weights=zero)
        fit = rm._fit(rows)
        assert fit is not None
        beta, _rms, ata, res = fit
        assert repr(rm._covariance(rows, beta, 4, ata, res)) == repr(rm._covariance(rows, beta, 4))


def test_the_trail_circle_matches_the_three_by_three_loop():
    rng = random.Random(1)
    for _ in range(200):
        k = rng.choice([1 / 40.0, -1 / 90.0, 1 / 400.0])
        s0 = rng.uniform(0.0, 80.0)
        pts = []
        for i in range(rng.randint(3, 60)):
            x, y = rm.arc_point(k, s0 + 0.5 * i)
            pts.append((x + rng.gauss(0.0, 0.05), y + rng.gauss(0.0, 0.05)))
        ordered = sorted(pts, key=lambda p: p[0] * p[0] + p[1] * p[1])
        mx = sum(p[0] for p in ordered) / len(ordered)
        my = sum(p[1] for p in ordered) / len(ordered)
        ata = [[0.0] * 3 for _ in range(3)]
        atb = [0.0] * 3
        for px, py in ordered:
            x, y = px - mx, py - my
            row = (x, y, 1.0)
            rhs = -(x * x + y * y)
            for i in range(3):
                atb[i] += row[i] * rhs
                for j in range(3):
                    ata[i][j] += row[i] * row[j]
        (x1, y1), (xn, yn) = ordered[0], ordered[-1]
        if math.hypot(xn - x1, yn - y1) < rm._REBASE_MIN_SPAN_M:
            continue
        sol = rm._solve(ata, atb)
        got = rm._trail_kappa(pts)
        if sol is None:
            assert got is None
            continue
        cx, cy = -0.5 * sol[0], -0.5 * sol[1]
        radius_sq = cx * cx + cy * cy - sol[2]
        if radius_sq <= 1e-9:
            assert got is None
            continue
        cross = (xn - x1) * (cy - (y1 - my)) - (yn - y1) * (cx - (x1 - mx))
        assert repr(got) == repr((-1.0 if cross > 0.0 else 1.0) / math.sqrt(radius_sq))


def _trails(seed: int, lengths):
    rng = random.Random(seed)
    out = []
    for sid, n in enumerate(lengths):
        lane = rng.choice([-3.5, 0.0, 3.5])
        s0 = rng.uniform(30.0, 120.0)
        for k in range(n):
            x, y = rm.arc_point(1 / 400.0, s0 - 0.5 * (n - 1 - k))
            out.append((sid, x, y + lane + rng.gauss(0.0, 0.05), 0.7))
    return out


def test_long_trails_keep_every_other_sample_and_their_whole_weight():
    samples = _trails(1, [5, 15, 16, 17, 64])
    thinned, dropped = rm._thin_trails(samples)
    for sid, n in enumerate([5, 15, 16, 17, 64]):
        before = [row for row in samples if row[0] == sid]
        after = [row for row in thinned if row[0] == sid]
        if n < rm._THIN_MIN_SAMPLES:
            assert after == before
            continue
        assert after[-1][:3] == before[-1][:3], "the newest sample is kept"
        assert [row[:3] for row in after] == [row[:3] for row in before[(n - 1) % 2::2]]
        assert math.isclose(sum(r[3] for r in after), sum(r[3] for r in before))
        assert dropped[sid] == n - len(after)


def test_thinning_moves_the_fit_and_its_trust_only_a_little(monkeypatch):
    samples = _trails(2, [64, 64, 40, 30, 8])
    full = (monkeypatch.setattr(rm, "_THIN_MIN_SAMPLES", 0), rm.fit_road_model([], samples, 1 / 400.0))[1]
    monkeypatch.setattr(rm, "_THIN_MIN_SAMPLES", 16)
    thin = rm.fit_road_model([], samples, 1 / 400.0)
    assert thin.n_sources == full.n_sources
    assert thin.base_kappa == full.base_kappa, "the base arc's vote reads every sample"
    assert (thin.far_weight, thin.span_med_m) == (full.far_weight, full.span_med_m)
    assert math.isclose(thin.confidence, full.confidence, abs_tol=0.02)
    assert math.isclose(thin.target_weight, full.target_weight, rel_tol=0.02)
    for s in (20.0, 60.0, 120.0):
        # A tenth of the measured lateral error at that range: the fit's own noise, not a new shape.
        assert abs(thin.road_coords(s, 0.0)[1] - full.road_coords(s, 0.0)[1]) < 0.1 * rm.lateral_sigma_m(s)
        assert math.isclose(thin.trust_at(s), full.trust_at(s), abs_tol=0.02)
