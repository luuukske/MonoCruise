"""Automatic hazards follow the brake and gas about to be sent."""
from __future__ import annotations

from core.sending_thread.thread import (
    _HAZARD_BRAKE_CLEAR,
    _HAZARD_GAS_RESET,
    _HAZARD_HARD_BRAKE,
    hazard_action_for_sent_pedals,
)


def _action(gas: float, brake: float, **kwargs: object) -> str | None:
    action, _hard = hazard_action_for_sent_pedals(
        gas,
        brake,
        speed_kmh=float(kwargs.get("speed_kmh", 50.0)),
        aeb_warn=bool(kwargs.get("aeb_warn", False)),
        user_override=bool(kwargs.get("user_override", False)),
        autodisable=bool(kwargs.get("autodisable", True)),
        was_hard=bool(kwargs.get("was_hard", False)),
    )
    return action


def test_hard_brake_turns_hazards_on():
    assert _HAZARD_HARD_BRAKE == 0.8
    assert _action(0.0, 0.85) == "on"
    assert _action(0.0, 0.8) == "on"


def test_held_hard_brake_does_not_retrigger():
    assert _action(0.0, 0.9, was_hard=True) is None


def test_brake_under_the_slam_floor_does_not_turn_hazards_on():
    assert _action(0.0, 0.79) is None


def test_override_gas_with_a_light_brake_does_not_turn_hazards_on():
    """The sent brake is what counts. A dropped ACC brake is not an input."""
    assert _action(0.9, 0.2) is None


def test_autodisable_waits_for_accelerator_not_brake_release():
    assert _action(0.0, 0.0) is None
    assert _action(0.7, 0.0) == "off"
    assert _action(_HAZARD_GAS_RESET, _HAZARD_BRAKE_CLEAR) is None


def test_autodisable_blocked_by_warn_override_speed_and_setting():
    assert _action(0.7, 0.0, aeb_warn=True) is None
    assert _action(0.7, 0.0, user_override=True) is None
    assert _action(0.7, 0.0, speed_kmh=12.0) is None
    assert _action(0.7, 0.0, autodisable=False) is None


def test_hard_brake_wins_over_autodisable():
    action, hard = hazard_action_for_sent_pedals(
        0.9,
        0.85,
        speed_kmh=50.0,
        aeb_warn=False,
        user_override=False,
        autodisable=True,
        was_hard=False,
    )
    assert action == "on"
    assert hard is True
