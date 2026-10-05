"""AEB HMI sound gate: two-tick arm on warn or brake, soft-stop on any cue end."""
from __future__ import annotations

import threading

from core.aeb.thread import _hmi_sound_step
from core.aeb.warning_player import SoundState, WarningPlayer


def test_first_warn_tick_does_not_start_sound():
    action, prev = _hmi_sound_step(True, False, False)
    assert action == "none"
    assert prev is True


def test_second_warn_tick_starts_sound():
    action, prev = _hmi_sound_step(True, False, True)
    assert action == "start"
    assert prev is True


def test_warn_end_soft_stops():
    action, prev = _hmi_sound_step(False, False, True)
    assert action == "stop"
    assert prev is False


def test_one_tick_warn_then_end_never_starts():
    action, prev = _hmi_sound_step(True, False, False)
    assert action == "none"
    action, prev = _hmi_sound_step(False, False, prev)
    assert action == "stop"
    assert prev is False


def test_later_idle_ticks_keep_soft_stop():
    action, prev = _hmi_sound_step(False, False, True)
    assert action == "stop"
    action, prev = _hmi_sound_step(False, False, prev)
    assert action == "stop"
    assert prev is False


def test_latched_brake_without_warn_still_starts_sound():
    # The 0a2dbd74 clip: 42 brake ticks, warn true on the engagement edge only.
    action, prev = _hmi_sound_step(True, True, False)
    assert action == "none"
    action, prev = _hmi_sound_step(False, True, prev)
    assert action == "start"
    assert prev is True


def test_brake_only_pulse_still_needs_two_ticks():
    action, prev = _hmi_sound_step(False, True, False)
    assert action == "none"
    action, prev = _hmi_sound_step(False, False, prev)
    assert action == "stop"
    assert prev is False


def test_warn_end_while_brake_holds_does_not_stop_sound():
    action, prev = _hmi_sound_step(False, True, True)
    assert action == "start"
    assert prev is True


def test_cue_end_needs_both_warn_and_brake_clear():
    action, prev = _hmi_sound_step(False, False, True)
    assert action == "stop"
    assert prev is False


def _handler_stub(*, state: SoundState) -> WarningPlayer:
    h = object.__new__(WarningPlayer)
    h._sound = object()
    h._state = state
    h._lock = threading.Lock()
    h._stop_extra_replays = 1
    h._min_cycles = 1
    h._cycles_played = 1
    h._replays_remaining = 1
    h._cue_active = True
    h._cleared_at = float("-inf")
    h._one_shot = None
    return h


def test_soft_stop_from_running_schedules_extra_replay():
    h = _handler_stub(state=SoundState.RUNNING)
    h._replays_remaining = 0
    h.stop_warning()
    assert h._state == SoundState.SHUTTING_DOWN
    assert h._replays_remaining == 1


def test_soft_stop_does_not_cut_shutdown_tail():
    h = _handler_stub(state=SoundState.SHUTTING_DOWN)
    h.stop_warning()
    assert h._state == SoundState.SHUTTING_DOWN
    assert h._replays_remaining == 1
