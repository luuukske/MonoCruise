"""The lane fit: the lane's own vertices kept as ego passes them, fitted smooth in ego's frame."""

from __future__ import annotations

import math

import pytest

from core.la.lane_model import LaneModel
from tests.la.servo_sim import game_route
from tests.lane_departure.test_detector import _arc_x, _arc_z


def pose(radius, s, lateral=0.0, psi=0.0):
    """On the arc ``s`` metres along, ``lateral`` right of it, heading ``psi`` right of the lane."""
    phi = 0.0 if radius is None else s / radius
    x = _arc_x(s, radius) + lateral * math.cos(phi)
    z = _arc_z(s, radius) + lateral * math.sin(phi)
    return x, z, math.sin(phi + psi), -math.cos(phi + psi)


@pytest.mark.parametrize("radius", [150.0, -150.0, 400.0, None])
def test_mid_span_reads_the_lane_not_its_chord(radius):
    model = LaneModel()
    for s in (100.0, 125.0, 147.0):
        model.observe(game_route(radius, s - 30.0, 10.0))
        model.observe(game_route(radius, s, 10.0))
        fit = model.local(*pose(radius, s, 0.3, 0.01), ahead_m=10.0)
        assert fit.offset == pytest.approx(0.3, abs=0.01), "a 10 m chord sags 0.08 m at R150"
        assert fit.psi == pytest.approx(0.01, abs=0.001)
        assert fit.kappa == pytest.approx(0.0 if radius is None else 1.0 / radius, abs=2e-4)
        assert fit.kappa_ahead == pytest.approx(fit.kappa, abs=2e-4)


def test_a_rebuilt_route_moves_nothing():
    rebuilt, fixed = LaneModel(), LaneModel()
    route = game_route(150.0, 0.0, 10.0, reach=600.0)
    psis = []
    for i in range(400):
        s = 60.0 + 1.25 * i
        rebuilt.observe(game_route(150.0, s, 10.0))
        fixed.observe(route)
        a, b = rebuilt.local(*pose(150.0, s), ahead_m=10.0), fixed.local(*pose(150.0, s), ahead_m=10.0)
        assert abs(a.offset) < 0.05, "a new lane with nothing behind ego yet still reads close"
        if s >= 90.0:
            assert a.offset == pytest.approx(b.offset, abs=1e-9) and a.psi == pytest.approx(b.psi, abs=1e-9)
            psis.append(a.psi)
    assert max(abs(q - p) for p, q in zip(psis, psis[1:])) < 2e-4, "ego's heading against the lane does not jump"


def test_another_lane_starts_over():
    model = LaneModel()
    model.observe(game_route(None, 100.0, 10.0))
    shifted = game_route(None, 100.0, 10.0)
    shifted.pts = tuple((x + 3.5, z) for x, z in shifted.pts)
    model.observe(shifted)
    fit = model.local(*pose(None, 100.0), ahead_m=10.0)
    assert fit.offset == pytest.approx(-3.5, abs=1e-6), "only the new lane's vertices count"


def test_too_few_vertices_gives_no_fit():
    model = LaneModel()
    model.observe(game_route(None, 100.0, 10.0, reach=10.0))
    assert model.local(*pose(None, 100.0), ahead_m=10.0) is None
