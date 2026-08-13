"""Warn suppression sources: physical deadzone vs any OPD / mapper brake."""
from __future__ import annotations

from core.aeb.thread import (
    _USER_BRAKE_LATCH_THRESHOLD,
    _addressing_brake_from_sources,
    _user_braking_from_sources,
)


def test_idle_pedals_do_not_count_as_braking():
    assert _user_braking_from_sources() is False
    assert _user_braking_from_sources(0.0, 0.0, 0.0) is False


def test_physical_pedal_keeps_the_deadzone():
    assert _user_braking_from_sources(brakeval=_USER_BRAKE_LATCH_THRESHOLD) is False
    assert _user_braking_from_sources(
        brakeval=_USER_BRAKE_LATCH_THRESHOLD + 1e-6,
    ) is True
    assert _user_braking_from_sources(brakeval=0.02) is False


def test_any_opd_brake_silences_warn():
    assert _user_braking_from_sources(opdbrakeval=1e-6) is True
    assert _user_braking_from_sources(opdbrakeval=0.04) is True


def test_any_mapper_brake_silences_warn():
    assert _user_braking_from_sources(program_brake=1e-6) is True
    assert _user_braking_from_sources(program_brake=0.01) is True


def test_tmp_latch_ignores_opd_and_keeps_the_deadzone():
    assert _addressing_brake_from_sources(program_brake=0.01) is False
    assert _addressing_brake_from_sources(
        program_brake=_USER_BRAKE_LATCH_THRESHOLD + 1e-6,
    ) is True
    assert _addressing_brake_from_sources(brakeval=0.02) is False
    assert _addressing_brake_from_sources(
        brakeval=_USER_BRAKE_LATCH_THRESHOLD + 1e-6,
    ) is True
