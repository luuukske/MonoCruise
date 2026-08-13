"""AEB HMI sound gate: two-tick arm, hard-stop on suppress."""
from __future__ import annotations

from core.aeb.thread import _hmi_sound_step


def test_first_warn_tick_does_not_start_sound():
    action, prev = _hmi_sound_step(True, False, False)
    assert action == "none"
    assert prev is True


def test_second_warn_tick_starts_sound():
    action, prev = _hmi_sound_step(True, True, False)
    assert action == "start"
    assert prev is True


def test_suppress_hard_stops_and_disarms():
    action, prev = _hmi_sound_step(False, True, True)
    assert action == "hard_stop"
    assert prev is False


def test_natural_end_soft_stops():
    action, prev = _hmi_sound_step(False, True, False)
    assert action == "stop"
    assert prev is False


def test_one_tick_warn_then_suppress_never_starts():
    action, prev = _hmi_sound_step(True, False, False)
    assert action == "none"
    action, prev = _hmi_sound_step(False, prev, True)
    assert action == "hard_stop"
    assert prev is False
