"""Grip diagnostics the brake debug log records: surface and worst wheel slip."""

from __future__ import annotations

import math

import pytest

from core.telemetry_thread.thread import _grip_fields, _wheel_slip, grip_debug_fields

R = 0.5


def _rps(speed_ms: float) -> float:
    return speed_ms / (2.0 * math.pi * R)


def test_a_locked_wheel_reads_one_and_a_rolling_one_zero():
    assert _wheel_slip([0.0, _rps(20.0)], [R, R], [True, True], 2, 20.0) == pytest.approx(1.0)
    assert _wheel_slip([_rps(20.0)] * 2, [R, R], [True, True], 2, 20.0) == pytest.approx(0.0, abs=1e-9)


def test_slip_ignores_airborne_wheels_and_walking_pace():
    assert _wheel_slip([0.0, _rps(20.0)], [R, R], [False, True], 2, 20.0) == pytest.approx(0.0, abs=1e-9)
    assert _wheel_slip([0.0, 0.0], [R, R], [True, True], 2, 1.0) == 0.0
    assert _wheel_slip(None, [R], [True], 1, 20.0) == 0.0


def test_trailer_slip_stops_at_the_first_detached_slot():
    rolling = {"attached": True, "wheelCount": 2, "wheelVelocity": [_rps(15.0)] * 2,
               "wheelRadius": [R, R], "wheelOnGround": [True, True]}
    locked_but_detached = {"attached": False, "wheelCount": 2, "wheelVelocity": [0.0, 0.0],
                           "wheelRadius": [R, R], "wheelOnGround": [True, True]}
    raw = {
        "substances": ["static", "road", "concrete"],
        "truck_wheelSubstance": [2, 2, 1, 1, 2, 2],
        "truckWheelOnGround": [True] * 6,
        "truck_wheelVelocity": [_rps(15.0)] * 6,
        "truckWheelRadius": [R] * 6,
        "truckWheelCount": 6,
        "trailer": [rolling, locked_but_detached],
    }
    surface, truck, trailer = _grip_fields(raw, 15.0)
    assert surface == "concrete"
    assert truck == pytest.approx(0.0, abs=1e-9)
    assert trailer == pytest.approx(0.0, abs=1e-9)


def test_debug_fields_tolerate_a_bare_data_object():
    fields = grip_debug_fields(object())
    assert set(fields) >= {"brake_temp_c", "air_psi", "surface", "truck_slip", "trailer_slip"}
    assert all(v == "" for v in fields.values())
