"""ATS mph converts at the edge. The stored setpoint stays km/h."""

from __future__ import annotations

import json

import pytest

from core.settings import Settings
from core.speed_units import (
    MPH_TO_KMH,
    display_from_kmh,
    display_from_ms,
    format_kmh,
    global_limit_bounds,
    kmh_from_display,
    quantize_speed_kmh,
    step_setpoint_kmh,
)


@pytest.fixture
def game(monkeypatch):
    settings = Settings.instance()

    def _set(last_game: int):
        monkeypatch.setattr(settings, "last_game", last_game)

    return _set


def test_ets2_display_stays_integer_kmh(game):
    game(1)
    assert format_kmh(None) == "-- km/h"
    assert format_kmh(80.4) == "80 km/h"
    assert display_from_ms(80.0 / 3.6) == 80
    assert kmh_from_display(90) == 90.0
    assert global_limit_bounds() == (60, 130)


def test_ats_mph_round_trips_and_survives_json(game):
    game(2)
    assert format_kmh(None) == "-- mph"
    assert global_limit_bounds() == (38, 80)
    for mph in range(1, 121):
        stored = mph * MPH_TO_KMH
        assert display_from_kmh(stored) == mph
        assert display_from_kmh(json.loads(json.dumps(stored))) == mph
        assert kmh_from_display(mph) == pytest.approx(stored)


def test_ats_set_speed_walls_are_the_kmh_clamp_in_mph(game):
    game(2)
    # 10 mph is under the 30 km/h floor, 90 mph is over the 130 km/h ceiling.
    assert quantize_speed_kmh(10.0 * MPH_TO_KMH / 3.6, None) == pytest.approx(19 * MPH_TO_KMH)
    assert quantize_speed_kmh(90.0 * MPH_TO_KMH / 3.6, None) == pytest.approx(80 * MPH_TO_KMH)
    assert quantize_speed_kmh(65.4 * MPH_TO_KMH / 3.6, None) == pytest.approx(65 * MPH_TO_KMH)


def test_ats_steps_on_the_mph_grid_and_a_blocked_step_keeps_the_stored_kmh(game):
    game(2)
    assert step_setpoint_kmh(65 * MPH_TO_KMH, 1, None) == pytest.approx(66 * MPH_TO_KMH)
    assert step_setpoint_kmh(65 * MPH_TO_KMH, 5, None) == pytest.approx(70 * MPH_TO_KMH)
    assert step_setpoint_kmh(67 * MPH_TO_KMH, -5, None) == pytest.approx(65 * MPH_TO_KMH)
    assert step_setpoint_kmh(65 * MPH_TO_KMH, -5, None) == pytest.approx(60 * MPH_TO_KMH)

    cap = 65 * MPH_TO_KMH
    assert step_setpoint_kmh(cap, 5, cap) == cap

    legacy = 30.0
    assert display_from_kmh(legacy) == 19
    assert step_setpoint_kmh(legacy, -1, None) == legacy
    assert step_setpoint_kmh(legacy, 1, None) == pytest.approx(20 * MPH_TO_KMH)


def test_ats_exact_mph_cap_is_reachable(game):
    """65 * factor / factor must not floor to 64 and lock the driver out of their own cap."""
    game(2)
    cap = 65 * MPH_TO_KMH
    assert step_setpoint_kmh(60 * MPH_TO_KMH, 5, cap) == pytest.approx(cap)
    assert display_from_kmh(cap) == 65
