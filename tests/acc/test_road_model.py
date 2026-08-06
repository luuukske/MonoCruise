"""Road-model fit on synthetic geometry: no clips, no radar thread.

Ego sits at the origin facing +x; +y is ego's right. Sources are given as
(id, x, y, weight) samples in that frame."""
from __future__ import annotations

import math

import pytest

from core.acc.road_model import (
    MIN_SOURCE_SAMPLES, RoadModel, fit_road_model, from_curvature,
)


def _true_arc(kappa, x):
    """Exact circular offset through ego, not the parabolic approximation.

    The parabola undershoots by `R - sqrt(R^2 - x^2) - x^2/2R`, which is 2 m at
    60 m on an R100 bend. Synthetic roads must use the circle or the tests
    silently bless the bias they exist to catch."""
    if abs(kappa) < 1e-9:
        return 0.0
    radius = 1.0 / abs(kappa)
    sign = 1.0 if kappa > 0.0 else -1.0
    reach = min(abs(x), radius * 0.999)
    return sign * (math.sqrt(max(radius * radius - reach * reach, 0.0)) - radius)


def _lane(source_id, lane_offset, xs, kappa=0.0, weight=1.0, jitter=0.0):
    """Samples of a vehicle holding ``lane_offset`` on a road of curvature kappa.

    The offset is **perpendicular** to the road, because that is what a lane is.
    Offsetting in +y instead makes the neighbouring lane a different shape, not
    a parallel one, and the fit is then right to report that they disagree."""
    out = []
    for i, x in enumerate(xs):
        offset = lane_offset + (jitter * math.sin(i * 1.7) if jitter else 0.0)
        px, py = _parallel_point(kappa, x, offset)
        out.append((source_id, px, py, weight))
    return out


def _parallel_point(kappa, x, offset):
    """Point ``offset`` m right of the centreline, at the arc reaching ``x``."""
    if abs(kappa) < 1e-9:
        return x, offset
    radius = 1.0 / abs(kappa)
    sign = 1.0 if kappa > 0.0 else -1.0
    theta = math.asin(min(abs(x), radius * 0.999) / radius)
    bx, by = radius * math.sin(theta), -sign * radius * (1.0 - math.cos(theta))
    return bx + offset * sign * math.sin(theta), by + offset * math.cos(theta)


def _ego_path(kappa=0.0, back_m=20.0, n=10):
    xs = [-back_m * (i / (n - 1)) for i in range(n)]
    return [(x, _true_arc(kappa, x)) for x in xs]


def test_straight_road_from_ego_path_alone_is_flat():
    model = fit_road_model(_ego_path(), [], fallback_kappa=0.0)
    for x in (0.0, 40.0, 100.0, 150.0):
        assert model.lateral_at(x) == pytest.approx(0.0, abs=0.05)


def test_centreline_passes_through_ego():
    samples = _lane(1, 0.0, range(10, 140, 10))
    model = fit_road_model(_ego_path(), samples)
    assert model.lateral_at(0.0) == pytest.approx(0.0, abs=1e-9)


def test_curved_road_recovered_from_one_leading_vehicle():
    kappa = 1.0 / 400.0
    samples = _lane(1, 0.0, range(10, 140, 10), kappa=kappa)
    model = fit_road_model(_ego_path(kappa=kappa), samples)
    assert model.confidence > 0.0
    for x in (40.0, 80.0, 120.0):
        assert model.lateral_at(x) == pytest.approx(_true_arc(kappa, x), abs=0.3)


def test_adjacent_lane_vehicle_contributes_shape_not_offset():
    """A vehicle one lane over must bend the model, not shift it.

    This is the per-source offset elimination: without it the centreline would
    be dragged toward whichever lane happens to have traffic in it."""
    kappa = 1.0 / 300.0
    samples = _lane(1, 3.5, range(10, 140, 10), kappa=kappa)
    model = fit_road_model(_ego_path(kappa=kappa), samples)
    assert model.lateral_at(60.0) == pytest.approx(_true_arc(kappa, 60.0), abs=0.5)
    assert model.offset_of(60.0, 3.5 + _true_arc(kappa, 60.0)) == pytest.approx(3.5, abs=0.5)


def test_two_lanes_agree_on_one_centreline():
    kappa = 1.0 / 500.0
    xs = list(range(10, 140, 10))
    samples = _lane(1, 0.0, xs, kappa=kappa) + _lane(2, 3.5, xs, kappa=kappa)
    model = fit_road_model(_ego_path(kappa=kappa), samples)
    assert model.offset_of(100.0, _true_arc(kappa, 100.0)) == pytest.approx(0.0, abs=0.4)
    assert model.offset_of(100.0, 3.5 + _true_arc(kappa, 100.0)) == pytest.approx(3.5, abs=0.4)


def test_lane_changing_source_is_down_weighted():
    """A vehicle crossing lanes must not bend the shared road estimate."""
    xs = list(range(10, 140, 10))
    straight = _lane(1, 0.0, xs) + _lane(2, 3.5, xs)
    crossing = [(3, x, 3.5 * (i / (len(xs) - 1)), 1.0) for i, x in enumerate(xs)]
    clean = fit_road_model(_ego_path(), straight)
    polluted = fit_road_model(_ego_path(), straight + crossing)
    for x in (50.0, 100.0, 140.0):
        assert abs(polluted.lateral_at(x) - clean.lateral_at(x)) < 0.6


def test_single_sample_sources_are_ignored():
    thin = [(i, 50.0 + i, 4.0, 1.0) for i in range(6)]
    model = fit_road_model(_ego_path(), thin)
    assert model.n_sources == 0
    for x in (40.0, 100.0):
        assert model.lateral_at(x) == pytest.approx(0.0, abs=0.05)


def test_source_needs_minimum_samples():
    xs = list(range(10, 10 + 10 * (MIN_SOURCE_SAMPLES - 1), 10))
    model = fit_road_model(_ego_path(), _lane(1, 0.0, xs, kappa=1 / 300))
    assert model.n_sources == 0


def test_no_evidence_falls_back_to_constant_curvature():
    kappa = 1.0 / 250.0
    model = fit_road_model([], [], fallback_kappa=kappa)
    assert model.confidence == 0.0
    for x in (30.0, 90.0):
        assert model.lateral_at(x) == pytest.approx(_true_arc(kappa, x), abs=1e-6)


def test_curvature_sign_matches_arcpath_convention():
    """Positive kappa is a left turn, so the centreline goes to negative y."""
    model = from_curvature(1.0 / 200.0)
    assert model.lateral_at(50.0) < 0.0
    assert model.curvature_at(0.0) == pytest.approx(1.0 / 200.0, rel=1e-6)


def test_confidence_rises_with_evidence():
    xs = list(range(10, 140, 10))
    sparse = fit_road_model(_ego_path(), _lane(1, 0.0, xs[:4], weight=0.2))
    rich = fit_road_model(
        _ego_path(), _lane(1, 0.0, xs) + _lane(2, 3.5, xs) + _lane(3, -3.5, xs),
    )
    assert rich.confidence > sparse.confidence


def test_noisy_samples_lower_confidence_but_keep_shape():
    xs = list(range(10, 140, 10))
    noisy = fit_road_model(_ego_path(), _lane(1, 0.0, xs, jitter=0.4))
    assert noisy.confidence > 0.0
    assert abs(noisy.lateral_at(100.0)) < 1.5


def test_model_is_stateless_and_repeatable():
    xs = list(range(10, 140, 10))
    samples = _lane(1, 0.0, xs, kappa=1 / 400)
    first = fit_road_model(_ego_path(kappa=1 / 400), samples)
    second = fit_road_model(_ego_path(kappa=1 / 400), samples)
    assert isinstance(first, RoadModel)
    assert (first.c1, first.c2, first.c3, first.c4) == (
        second.c1, second.c2, second.c3, second.c4,
    )


@pytest.mark.parametrize("radius,x_eval", [(500.0, 100.0), (200.0, 100.0), (100.0, 60.0)])
@pytest.mark.parametrize("side", [1.0, -1.0])
def test_tight_corner_centreline_does_not_drift_outward(radius, x_eval, side):
    """The reported failure: on tight bends the centreline sat outside the road.

    A parabola undershoots a circle by 0.1 m at R500/100 m but 2.0 m at
    R100/60 m, always toward the outside of the bend."""
    kappa = side / radius
    xs = [x for x in range(10, int(x_eval) + 20, 10)]
    model = fit_road_model(_ego_path(kappa=kappa), _lane(1, 0.0, xs, kappa=kappa),
                           fallback_kappa=kappa)
    truth = _true_arc(kappa, x_eval)
    assert model.lateral_at(x_eval) == pytest.approx(truth, abs=0.5)
    # And specifically not biased toward the outside of the bend.
    assert abs(model.lateral_at(x_eval)) > abs(truth) - 0.5


def test_base_arc_matches_the_circle_not_the_parabola():
    from core.acc.road_model import base_arc_lateral

    kappa = 1.0 / 100.0
    parabola = -0.5 * kappa * 60.0 * 60.0
    exact = base_arc_lateral(kappa, 60.0)
    assert exact == pytest.approx(_true_arc(kappa, 60.0), abs=1e-9)
    assert abs(exact) - abs(parabola) == pytest.approx(2.0, abs=0.1)


def test_base_arc_is_bounded_past_its_radius():
    from core.acc.road_model import base_arc_lateral

    kappa = 1.0 / 50.0
    assert math.isfinite(base_arc_lateral(kappa, 500.0))
    assert abs(base_arc_lateral(kappa, 500.0)) <= 50.0


def _arc_lane(source_id, offset, kappa, s_values, weight=1.0):
    """Samples holding ``offset`` m right of the centreline, placed by arc length."""
    from core.acc.road_model import arc_normal, arc_point

    out = []
    for s_m in s_values:
        bx, by = arc_point(kappa, s_m)
        nx, ny = arc_normal(kappa, s_m)
        out.append((source_id, bx + offset * nx, by + offset * ny, weight))
    return out


@pytest.mark.parametrize("radius", [500.0, 100.0, 25.0])
@pytest.mark.parametrize("side", [1.0, -1.0])
@pytest.mark.parametrize("sweep_deg", [30.0, 90.0, 179.0])
def test_arc_coords_inverts_arc_point_at_any_angle(radius, side, sweep_deg):
    """The transform the whole model rests on, checked past the 90 deg fold."""
    from core.acc.road_model import arc_coords, arc_normal, arc_point

    kappa = side / radius
    s_m = radius * math.radians(sweep_deg)
    for offset in (0.0, 3.5, -3.5):
        bx, by = arc_point(kappa, s_m)
        nx, ny = arc_normal(kappa, s_m)
        got_s, got_n = arc_coords(kappa, bx + offset * nx, by + offset * ny)
        assert got_s == pytest.approx(s_m, abs=1e-6)
        assert got_n == pytest.approx(offset, abs=1e-9)


def test_arc_length_is_monotone_where_forward_distance_folds():
    """Forward distance stops being unique at 90 deg and decreases after it.

    That fold is what made two points on the road share an x, read as one source
    contradicting itself, and collapse confidence. Arc length never folds."""
    from core.acc.road_model import arc_coords, arc_point

    kappa = 1.0 / 40.0
    s_values = [i * 5.0 for i in range(1, 25)]
    forward = [arc_point(kappa, s)[0] for s in s_values]
    assert any(b < a for a, b in zip(forward, forward[1:])), "no fold to test"
    arc = [arc_coords(kappa, *arc_point(kappa, s))[0] for s in s_values]
    assert all(b > a for a, b in zip(arc, arc[1:]))


@pytest.mark.parametrize("sweep_deg", [60.0, 90.0, 135.0, 170.0])
def test_fit_survives_past_the_forward_distance_wall(sweep_deg):
    """A forward-distance fit gave up at 71.8 deg of heading change, where the
    base arc saturates and its own frozen value reads as sources disagreeing.

    Perfect samples on a perfect circle used to reach agreement rms 0.963 and
    confidence 0.00 by 80 deg. Indexed by arc length the only limit left is
    the circle closing on itself, just short of a half turn."""
    radius = 60.0
    kappa = 1.0 / radius
    span = radius * math.radians(sweep_deg)
    s_values = [span * i / 24.0 for i in range(1, 25)]
    samples = (
        _arc_lane(1, 0.0, kappa, s_values)
        + _arc_lane(2, 3.5, kappa, s_values)
        + _arc_lane(3, -3.5, kappa, s_values)
    )
    model = fit_road_model(_ego_path(kappa=kappa), samples, fallback_kappa=kappa)
    assert model.confidence > 0.0
    assert model.support_s_m == pytest.approx(span, rel=0.05)
    # The centreline still reports a lane offset correctly at the far end.
    far = _arc_lane(0, 3.5, kappa, [span * 0.9])[0]
    assert model.offset_of(far[1], far[2]) == pytest.approx(3.5, abs=0.35)


def test_a_hard_steer_standstill_cannot_alias_samples_round_its_own_circle():
    """Ego steering hard at a standstill reports R = 7 m, whose circle closes
    inside the sample span. Without the span limit those samples wrapped onto
    the near side and the fit answered with a 4 km deviation."""
    from core.acc.road_model import arc_span_limit

    kappa = 0.134
    assert arc_span_limit(kappa) < 25.0
    xs = list(range(10, 140, 10))
    model = fit_road_model(
        _ego_path(), _lane(1, 0.0, xs) + _lane(2, 3.5, xs), fallback_kappa=kappa,
    )
    assert model.support_s_m <= arc_span_limit(kappa)
    for s_m in (30.0, 80.0, 150.0):
        assert model.confidence_at(s_m) == 0.0
    assert all(abs(v) < 50.0 for v in (model.c1, model.c2, model.c3, model.c4))


def test_straight_roads_have_no_span_limit():
    from core.acc.road_model import arc_span_limit

    assert math.isinf(arc_span_limit(0.0))
    assert arc_span_limit(1.0 / 400.0) > 170.0


def test_support_is_arc_length_not_forward_distance():
    """Round a bend the two differ by sin(theta)/theta, so a forward-distance
    support cut the estimate off well short of the traffic that made it."""
    radius = 40.0
    kappa = 1.0 / radius
    span = radius * math.radians(120.0)
    s_values = [span * i / 20.0 for i in range(1, 21)]
    model = fit_road_model(
        _ego_path(kappa=kappa),
        _arc_lane(1, 0.0, kappa, s_values) + _arc_lane(2, 3.5, kappa, s_values),
        fallback_kappa=kappa,
    )
    forward_reach = max(
        abs(x) for _, x, _, _ in _arc_lane(1, 0.0, kappa, s_values)
    )
    assert model.support_s_m > forward_reach * 1.4
    assert model.confidence_at(span * 0.95) > 0.0


def _corner_lane(source_id, offset, radius, sweep_deg, n=20):
    """A lane on a road that runs straight up to ego and then bends right at R.

    Ego sits exactly at the bend start, so the base arc it hands the fit is
    straight while the road ahead is not. That is corner entry, and it is the
    case the cubic could not describe: its curvature is affine in s and a
    corner's is a step."""
    out = []
    for i in range(1, n + 1):
        theta = math.radians(sweep_deg) * i / n
        bx, by = radius * math.sin(theta), radius * (1.0 - math.cos(theta))
        out.append((source_id,
                    bx - offset * math.sin(theta),
                    by + offset * math.cos(theta), 1.0))
    return out


@pytest.mark.parametrize("radius", [200.0, 80.0, 40.0])
def test_corner_entry_stays_confident_past_45_degrees(radius):
    """Reported from the driver's seat: traffic round a bend stopped being
    tracked at roughly 45 deg, even with oncoming traffic confirming the road.

    On noiseless samples of a perfect corner the cubic reached agreement 0.49 m
    by 45 deg of visible bend, against a `_CONF_RESIDUAL_BAD_M` of 0.60, so
    confidence collapsed with nothing whatever wrong with the evidence."""
    samples = (
        _corner_lane(1, 0.0, radius, 45.0)
        + _corner_lane(2, 3.5, radius, 45.0)
    )
    model = fit_road_model(_ego_path(), samples, fallback_kappa=0.0)
    assert model.confidence > 0.35
    # A vehicle in ego's lane at the far end of the bend still reads in-lane,
    # and one a lane over still reads out of it.
    centre = _corner_lane(0, 0.0, radius, 45.0, n=1)[0]
    adjacent = _corner_lane(0, 3.5, radius, 45.0, n=1)[0]
    assert abs(model.offset_of(centre[1], centre[2])) < 0.5
    assert model.offset_of(adjacent[1], adjacent[2]) > 2.0


def _profile_lane(source_id, offset, kappa_at, length_m, n=24):
    """A lane on a road whose curvature follows ``kappa_at(s)``.

    Re-basing carries a road that is one arc, so the term that still has to
    exist for a road that is not gets tested on a road that is not."""
    out = []
    x = y = heading = 0.0
    step = length_m / (n * 4)
    for i in range(n * 4):
        heading += kappa_at(i * step) * step
        x += math.cos(heading) * step
        y -= math.sin(heading) * step
        if (i + 1) % 4 == 0:
            out.append((source_id,
                        x - offset * math.sin(heading),
                        y + offset * math.cos(heading), 1.0))
    return out


def test_a_bend_that_straightens_needs_the_quartic_term():
    """Pins the mechanism, not just the symptom.

    A single clothoid has curvature affine in s, so it can describe a bend
    tightening at a constant rate and nothing else. A bend that arrives and
    then leaves is the ordinary case a base arc cannot carry either. The
    corpus is the real evidence for `c4`; this pins that it is load-bearing."""
    def kappa_at(s_m):
        return -(1.0 / 250.0) * math.sin(math.pi * s_m / 120.0)

    samples = (
        _profile_lane(1, 0.0, kappa_at, 120.0)
        + _profile_lane(2, 3.5, kappa_at, 120.0)
    )
    model = fit_road_model(_ego_path(), samples, fallback_kappa=0.0)
    cubic_only = RoadModel(
        c1=model.c1, c2=model.c2, c3=model.c3, base_kappa=model.base_kappa,
    )
    worst = max(
        abs(cubic_only.raw_deviation_at(s) - model.raw_deviation_at(s))
        for s in (40.0, 70.0, 100.0)
    )
    assert worst > 0.5, "c4 is carrying nothing; re-check whether it earns its place"


def _smoother_step(sm, model, x=0.0, z=0.0, fwd=(1.0, 0.0), dt=1 / 30):
    return sm.step(model, x, z, fwd[0], fwd[1], dt)


def test_smoother_passes_normal_change_untouched():
    """Frame-to-frame change is far below the slew limit in normal driving."""
    from core.acc.road_model import from_curvature
    from core.acc.road_smoother import RoadSmoother

    sm = RoadSmoother()
    _smoother_step(sm, from_curvature(0.0))
    out = _smoother_step(sm, from_curvature(1.0 / 2000.0))
    assert out.lateral_at(50.0) == pytest.approx(
        from_curvature(1.0 / 2000.0).lateral_at(50.0), abs=0.05,
    )


def test_smoother_clips_a_step_event():
    """A sudden centreline snap is rate limited, not passed straight through."""
    from core.acc.road_model import from_curvature
    from core.acc.road_smoother import RoadSmoother, node_slew_budget_ms

    sm = RoadSmoother()
    _smoother_step(sm, from_curvature(0.0))
    out = _smoother_step(sm, from_curvature(1.0 / 60.0))
    step = node_slew_budget_ms(50.0) / 30.0
    assert abs(out.lateral_at(50.0)) <= step + 1e-6


def test_slew_budget_is_uniform_in_curvature():
    """Above the floor a node twice as far may move four times as fast, so one
    curvature change costs the same fraction of the budget at every distance."""
    from core.acc.road_smoother import SMOOTH_MIN_RATE_MS, node_slew_budget_ms

    near, far = node_slew_budget_ms(80.0), node_slew_budget_ms(160.0)
    assert near > SMOOTH_MIN_RATE_MS, "pick distances above the floor knee"
    assert far == pytest.approx(4.0 * near, rel=1e-6)
    # The floor keeps the nodes closest to ego from freezing.
    assert node_slew_budget_ms(5.0) == pytest.approx(SMOOTH_MIN_RATE_MS)


def test_smoother_still_reaches_the_new_shape():
    """Rate limiting delays a step, it does not reject it."""
    from core.acc.road_model import from_curvature
    from core.acc.road_smoother import RoadSmoother

    sm = RoadSmoother()
    target = from_curvature(1.0 / 200.0)
    out = None
    for _ in range(120):
        out = _smoother_step(sm, target)
    assert out.lateral_at(50.0) == pytest.approx(target.lateral_at(50.0), abs=0.2)


def test_smoother_keeps_the_centreline_through_ego():
    from core.acc.road_model import from_curvature
    from core.acc.road_smoother import RoadSmoother

    sm = RoadSmoother()
    for kappa in (0.0, 1.0 / 400.0, -1.0 / 250.0):
        out = _smoother_step(sm, from_curvature(kappa))
        assert out.lateral_at(0.0) == pytest.approx(0.0, abs=1e-9)


def test_smoother_resets_on_an_ego_jump():
    """A teleport must not be smoothed across."""
    from core.acc.road_model import from_curvature
    from core.acc.road_smoother import RoadSmoother

    sm = RoadSmoother()
    _smoother_step(sm, from_curvature(0.0))
    out = _smoother_step(sm, from_curvature(1.0 / 60.0), x=500.0, z=500.0)
    assert out.lateral_at(50.0) == pytest.approx(
        from_curvature(1.0 / 60.0).lateral_at(50.0), abs=1e-6,
    )


def test_smoother_does_not_carry_untrusted_extrapolation():
    """Beyond support the base arc goes on the grid, not the runaway cubic."""
    from core.acc.road_smoother import RoadSmoother

    wild = RoadModel(c1=0.0, c2=0.0, c3=400.0, confidence=1.0, support_s_m=30.0)
    out = _smoother_step(RoadSmoother(), wild)
    assert abs(out.lateral_at(140.0)) < 5.0
    assert abs(wild.raw_deviation_at(140.0)) > 500.0


def test_more_agreeing_traffic_never_lowers_confidence():
    """Corroboration must not be punished.

    The first agreement term was the range of the per-source residuals, a
    statistic with no breakdown point whose expected value grows with the number
    of sources. Confidence therefore fell to zero once five vehicles were in
    view, which is the opposite of what corroboration means, and showed up from
    the driver's seat as the prediction fading in and out in traffic."""
    xs = [20.0 * i for i in range(1, 7)]
    confidences = []
    for count in range(2, 7):
        samples = []
        for k in range(count):
            samples += _lane(k, 3.5 * k, xs, kappa=1.0 / 400.0)
        model = fit_road_model(_ego_path(1.0 / 400.0), samples, 1.0 / 400.0)
        confidences.append(model.confidence)
    assert all(b >= a - 1e-9 for a, b in zip(confidences, confidences[1:])), (
        f"confidence fell as sources were added: {confidences}"
    )
    assert confidences[-1] > 0.5


@pytest.mark.parametrize("drift_m", [0.5, 1.0, 3.5, 10.0])
def test_one_dissenter_does_not_cost_the_agreeing_sources_their_weight(drift_m):
    """Rejecting a vehicle leaving the road must not disarm the fit.

    The IRLS passes used to compound weights instead of rescaling the originals,
    so a first fit dragged by one dissenter cratered every source: four sources
    fitting perfectly still lost 94% of their weight, confidence hit zero, and
    the prediction dropped out exactly when traffic was there to support it."""
    kappa = 1.0 / 400.0
    xs = [20.0 * i for i in range(1, 7)]
    good = []
    for k in range(4):
        good += _lane(k, 3.5 * k, xs, kappa=kappa)
    clean = fit_road_model(_ego_path(kappa), good, kappa)

    a = drift_m / (120.0 ** 2)
    turning = [(99, x, a * x * x, 1.0) for x in xs]
    mixed = fit_road_model(_ego_path(kappa), good + turning, kappa)

    assert clean.confidence > 0.5
    assert mixed.target_weight >= clean.target_weight * 0.9
    assert mixed.confidence >= clean.confidence * 0.9
    # And the dissenter is still kept out of the shape.
    for x in (60.0, 120.0):
        assert abs(mixed.lateral_at(x) - clean.lateral_at(x)) < 0.5


def test_oncoming_traffic_is_a_usable_road_source():
    """Opposite traffic drives the same road, just in the other direction.

    Its lateral offset is removed by the same per-source centring that lets an
    adjacent lane contribute shape, so the sign of its heading changes nothing
    about the geometry it carries."""
    kappa = 1.0 / 300.0
    xs = [20.0 * i for i in range(1, 8)]
    # Two opposing lanes, sampled the way an approaching vehicle's trail is:
    # its history lies ahead of it, further from ego than the vehicle itself.
    oncoming = _lane(1, -3.5, xs, kappa=kappa) + _lane(2, -7.0, xs, kappa=kappa)
    model = fit_road_model(_ego_path(kappa), oncoming, kappa)

    assert model.confidence > 0.0, "oncoming-only traffic must still fit a road"
    for x in (40.0, 100.0):
        assert model.lateral_at(x) == pytest.approx(_true_arc(kappa, x), abs=0.6)
    # And it is not dragged into the opposing lanes.
    assert model.offset_of(100.0, _true_arc(kappa, 100.0)) == pytest.approx(0.0, abs=0.6)


def test_cross_traffic_is_not_a_road_source():
    """Between the co-directional and oncoming bands sits traffic turning off."""
    from core.acc.tracker import _direction_weight

    assert _direction_weight(0.0) == 1.0
    assert _direction_weight(180.0) > 0.0
    assert _direction_weight(-175.0) > 0.0
    for turning in (60.0, 90.0, -110.0):
        assert _direction_weight(turning) == 0.0
