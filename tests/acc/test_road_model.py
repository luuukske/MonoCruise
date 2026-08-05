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
    """Samples of a vehicle holding ``lane_offset`` on a road of curvature kappa."""
    out = []
    for i, x in enumerate(xs):
        y = lane_offset + _true_arc(kappa, x)
        if jitter:
            y += jitter * math.sin(i * 1.7)
        out.append((source_id, x, y, weight))
    return out


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
    assert (first.c1, first.c2, first.c3) == (second.c1, second.c2, second.c3)


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


def _smoother_step(sm, model, x=0.0, z=0.0, fwd=(1.0, 0.0), dt=1 / 30):
    return sm.step(model, x, z, fwd[0], fwd[1], dt)


def test_smoother_passes_normal_change_untouched():
    """Frame-to-frame change is far below the slew limit in normal driving."""
    from core.acc.road_model import RoadSmoother, from_curvature

    sm = RoadSmoother()
    _smoother_step(sm, from_curvature(0.0))
    out = _smoother_step(sm, from_curvature(1.0 / 2000.0))
    assert out.lateral_at(50.0) == pytest.approx(
        from_curvature(1.0 / 2000.0).raw_lateral_at(50.0), abs=0.05,
    )


def test_smoother_clips_a_step_event():
    """A sudden centreline snap is rate limited, not passed straight through."""
    from core.acc.road_model import RoadSmoother, from_curvature, node_slew_budget_ms

    sm = RoadSmoother()
    _smoother_step(sm, from_curvature(0.0))
    out = _smoother_step(sm, from_curvature(1.0 / 60.0))
    step = node_slew_budget_ms(50.0) / 30.0
    assert abs(out.lateral_at(50.0)) <= step + 1e-6


def test_slew_budget_is_uniform_in_curvature():
    """Above the floor a node twice as far may move four times as fast, so one
    curvature change costs the same fraction of the budget at every distance."""
    from core.acc.road_model import SMOOTH_MIN_RATE_MS, node_slew_budget_ms

    near, far = node_slew_budget_ms(80.0), node_slew_budget_ms(160.0)
    assert near > SMOOTH_MIN_RATE_MS, "pick distances above the floor knee"
    assert far == pytest.approx(4.0 * near, rel=1e-6)
    # The floor keeps the nodes closest to ego from freezing.
    assert node_slew_budget_ms(5.0) == pytest.approx(SMOOTH_MIN_RATE_MS)


def test_smoother_still_reaches_the_new_shape():
    """Rate limiting delays a step, it does not reject it."""
    from core.acc.road_model import RoadSmoother, from_curvature

    sm = RoadSmoother()
    target = from_curvature(1.0 / 200.0)
    out = None
    for _ in range(120):
        out = _smoother_step(sm, target)
    assert out.lateral_at(50.0) == pytest.approx(target.raw_lateral_at(50.0), abs=0.2)


def test_smoother_keeps_the_centreline_through_ego():
    from core.acc.road_model import RoadSmoother, from_curvature

    sm = RoadSmoother()
    for kappa in (0.0, 1.0 / 400.0, -1.0 / 250.0):
        out = _smoother_step(sm, from_curvature(kappa))
        assert out.lateral_at(0.0) == pytest.approx(0.0, abs=1e-9)


def test_smoother_resets_on_an_ego_jump():
    """A teleport must not be smoothed across."""
    from core.acc.road_model import RoadSmoother, from_curvature

    sm = RoadSmoother()
    _smoother_step(sm, from_curvature(0.0))
    out = _smoother_step(sm, from_curvature(1.0 / 60.0), x=500.0, z=500.0)
    assert out.lateral_at(50.0) == pytest.approx(
        from_curvature(1.0 / 60.0).raw_lateral_at(50.0), abs=1e-6,
    )


def test_smoother_does_not_carry_untrusted_extrapolation():
    """Beyond support the base arc goes on the grid, not the runaway cubic."""
    from core.acc.road_model import RoadSmoother

    wild = RoadModel(c1=0.0, c2=0.0, c3=400.0, confidence=1.0, support_x_m=30.0)
    out = _smoother_step(RoadSmoother(), wild)
    assert abs(out.lateral_at(140.0)) < 5.0
    assert abs(wild.raw_lateral_at(140.0)) > 500.0


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
