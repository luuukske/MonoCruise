"""Cruise mapper output is visible to automatic hazards as driver pedals."""
from __future__ import annotations

from core.sending_thread.thread import (
    _HAZARD_BRAKE_CLEAR,
    _HAZARD_GAS_RESET,
    _HAZARD_HARD_BRAKE,
    cruise_pedals_for_hazards,
)


def test_idle_mapper_is_ignored():
    gas, brake = cruise_pedals_for_hazards(0.1, 0.0, 0.9, 0.9, False)
    assert (gas, brake) == (0.1, 0.0)


def test_cruise_gas_looks_like_the_driver():
    gas, brake = cruise_pedals_for_hazards(0.0, 0.0, 0.7, 0.0, True)
    assert gas >= _HAZARD_GAS_RESET
    assert brake < _HAZARD_BRAKE_CLEAR


def test_cruise_hard_brake_matches_the_driver_slam_floor():
    _, brake = cruise_pedals_for_hazards(0.0, 0.0, 0.0, 0.85, True)
    assert brake >= _HAZARD_HARD_BRAKE


def test_user_and_cruise_take_the_max():
    gas, brake = cruise_pedals_for_hazards(0.4, 0.2, 0.7, 0.05, True)
    assert gas == 0.7
    assert brake == 0.2


def test_autodisable_waits_for_accelerator_not_brake_release():
    """Easing the slam without gas must not look like autodisable."""
    gas, brake = cruise_pedals_for_hazards(0.0, 0.0, 0.0, 0.0, True)
    assert gas < _HAZARD_GAS_RESET
    gas, brake = cruise_pedals_for_hazards(0.0, 0.0, 0.7, 0.0, True)
    assert gas >= _HAZARD_GAS_RESET
    assert brake < _HAZARD_BRAKE_CLEAR
