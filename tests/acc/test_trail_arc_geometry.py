"""Trail-arc ground truth on synthetic roads: no clips, no radar thread.

A target following the same road as ego must cross the ego row at the target's
own lane offset with an arrival angle of ~0, on straights and on curves alike."""
from __future__ import annotations

import math
import random

import pytest

from core.acc.trail_arc import (
    MIN_FIT_SAMPLES, crossing_offset_and_angle, fit_trail,
)


# Ego sits at the origin facing +Z; ego right is (-1, 0) per the radar frame.
_EGO = (0.0, 0.0, 0.0, 1.0)
_DT = 1.0 / 15.0
_SAMPLES = 25


def _history(radius_signed, lane_off, ahead_m, speed=25.0, noise=0.0, seed=1):
    """(t, x, z) history for a target ``ahead_m`` along a road of signed radius."""
    rng = random.Random(seed)
    fwd_x, fwd_z = 0.0, 1.0
    right_x, right_z = -fwd_z, fwd_x
    points = []
    for i in range(_SAMPLES):
        s = ahead_m - (_SAMPLES - 1 - i) * speed * _DT
        if radius_signed is None:
            x = lane_off * right_x + s * fwd_x
            z = lane_off * right_z + s * fwd_z
        else:
            radius = abs(radius_signed)
            sign = 1.0 if radius_signed > 0 else -1.0
            cx, cz = sign * radius * right_x, sign * radius * right_z
            a0 = math.atan2(-cz, -cx)
            angle = a0 + sign * (s / radius)
            r_target = radius - sign * lane_off
            x = cx + r_target * math.cos(angle)
            z = cz + r_target * math.sin(angle)
        if noise:
            x += rng.gauss(0.0, noise)
            z += rng.gauss(0.0, noise)
        points.append((i * _DT, x, z))
    return points


def _cross(radius_signed, lane_off, ahead_m, **kw):
    fit = fit_trail(_history(radius_signed, lane_off, ahead_m, **kw), math.pi)
    assert fit is not None, "expected a usable trail fit"
    result = crossing_offset_and_angle(fit, *_EGO)
    assert result is not None, "expected the trail to cross the ego row"
    return result


@pytest.mark.parametrize("ahead", [20.0, 50.0, 90.0])
def test_straight_road_centre_lane(ahead):
    offset, angle = _cross(None, 0.0, ahead)
    assert offset == pytest.approx(0.0, abs=0.01)
    assert angle == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("ahead", [20.0, 50.0, 90.0])
def test_straight_road_adjacent_lane(ahead):
    offset, angle = _cross(None, 3.5, ahead)
    assert offset == pytest.approx(3.5, abs=0.05)
    assert angle == pytest.approx(0.0, abs=0.01)


@pytest.mark.parametrize("radius", [2000.0, 500.0, 200.0, 80.0])
@pytest.mark.parametrize("side", [1.0, -1.0])
def test_curve_same_lane_crosses_at_ego_with_zero_angle(radius, side):
    """A target on ego's own curved lane must read offset 0 and angle 0.

    Guards the arc tangent sign: a flipped sweep reads ~180 deg here and drives
    angle_amp to zero for every in-lane target on a curve."""
    offset, angle = _cross(side * radius, 0.0, 50.0)
    assert offset == pytest.approx(0.0, abs=0.05)
    assert math.degrees(angle) == pytest.approx(0.0, abs=1.0)


def test_oncoming_target_reads_reversed_arrival_angle():
    """A target travelling toward ego must arrive near 180 deg, not near 0."""
    history = [(t, x, -z) for t, x, z in _history(None, 0.0, -50.0)]
    fit = fit_trail(history, 0.0)
    assert fit is not None
    result = crossing_offset_and_angle(fit, *_EGO)
    assert result is not None
    assert math.degrees(result[1]) > 150.0


@pytest.mark.parametrize("noise,angle_tol", [(0.02, 0.5), (0.05, 1.0), (0.15, 5.0)])
def test_curve_fit_degrades_gracefully_with_position_noise(noise, angle_tol):
    """Position noise must perturb the arrival angle, not invert it."""
    offset, angle = _cross(500.0, 0.0, 50.0, noise=noise, seed=7)
    assert abs(offset) < 1.0
    assert math.degrees(angle) == pytest.approx(0.0, abs=angle_tol)


def test_a_truly_stationary_target_has_no_fit():
    """A vehicle that never moved leaves no trail to fit."""
    assert fit_trail(_history(None, 0.0, 50.0, speed=0.0), math.pi) is None


def test_slow_target_still_fits_on_the_distance_grid():
    """The distance-retained trail is what lowers the low-speed floor.

    On the old time-capped buffer a 2 m/s target spanned 3.3 m and the 1.0 m
    downsample gate left too few samples, so every slow vehicle read NO_HISTORY."""
    fit = fit_trail(_history(None, 0.0, 50.0, speed=2.0), math.pi)
    assert fit is not None
    assert fit.evidence > 0.0


def test_moving_target_has_enough_downsampled_samples():
    from core.acc.trail_arc import _downsample

    kept = _downsample(_history(None, 0.0, 50.0, speed=25.0))
    assert len(kept) >= MIN_FIT_SAMPLES
