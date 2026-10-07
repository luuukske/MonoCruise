"""AEB HMI sound gate: two-tick arm on warn or brake, soft-stop on any cue end."""
from __future__ import annotations

import threading
from pathlib import Path

from core.aeb.thread import (
    _HMI_CUE_OFF_DELAY_S,
    _HMI_STOPPED_SPEED_MS,
    _hmi_cue_until,
    _hmi_sound_step,
    _hmi_stopped_step,
)
from core.aeb.warning_player import SoundState, WarningPlayer

_ROOT = Path(__file__).resolve().parents[2]
_RESUME_MS = 5.0 / 3.6
_MOVING_MS = 15.0


def _cue_trace(ticks: list[tuple], dt: float = 1 / 30) -> list[bool]:
    """Published AEB_cue per tick for (warn, brake[, speed]) ticks, as the AEB loop steps it."""
    prev, until, stopped, out = False, float("-inf"), False, []
    for i, tick in enumerate(ticks):
        warn, brake, speed = (*tick, _MOVING_MS)[:3]
        now = i * dt
        stopped = _hmi_stopped_step(warn or brake, speed, stopped, _RESUME_MS)
        action, prev = _hmi_sound_step(warn and not stopped, brake and not stopped, prev)
        until = _hmi_cue_until(action, now, until)
        out.append(now < until)
    return out


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
    h._brake_sound = None
    h._braking = False
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


def test_visual_cue_shows_a_brake_without_warn():
    # Clip f61ae726: OPD coast-down suppressed warn for the whole 2.8 s brake.
    cue = _cue_trace([(False, True)] * 10)
    assert cue[0] is False
    assert all(cue[1:])


def test_visual_cue_holds_while_brake_outlasts_warn():
    # Clip 0a2dbd74: warn drops while the latch keeps braking a moving truck.
    cue = _cue_trace([(True, True)] * 5 + [(False, True)] * 30)
    assert all(cue[1:])


def test_cue_goes_quiet_once_aeb_has_stopped_the_truck():
    # Clip 7d9caa56: the geometry latch held the brake 3.8 s at standstill.
    braking = [(True, True, 10.0 - i) for i in range(10)]
    hold = [(False, True, 0.0)] * 60
    cue = _cue_trace(braking + hold)
    assert all(cue[1:10])
    tail = sum(cue[10:]) / 30
    assert tail <= _HMI_CUE_OFF_DELAY_S + 1 / 30
    assert cue[-1] is False


def test_standstill_jitter_does_not_bring_the_cue_back():
    hold = [(True, True, 0.0)] * 5 + [(True, True, s) for s in (0.15, -0.1, 0.6, 1.2, 0.3)] * 6
    assert not any(_cue_trace(hold)[10:])


def test_moving_off_above_the_engage_floor_restores_the_cue():
    cue = _cue_trace([(False, True, 0.0)] * 10 + [(False, True, _RESUME_MS + 0.5)] * 10)
    assert not any(cue[:10])
    assert all(cue[11:])


def test_stopped_latch_clears_when_the_event_ends():
    assert _hmi_stopped_step(True, 0.0, False, _RESUME_MS) is True
    assert _hmi_stopped_step(False, 0.0, True, _RESUME_MS) is False
    assert _hmi_stopped_step(True, _HMI_STOPPED_SPEED_MS + 0.1, False, _RESUME_MS) is False
    assert _hmi_stopped_step(True, _HMI_STOPPED_SPEED_MS + 0.1, True, _RESUME_MS) is True
    assert _hmi_stopped_step(True, -_HMI_STOPPED_SPEED_MS, False, _RESUME_MS) is True


def test_stopped_threshold_sits_between_jitter_and_the_engage_floor():
    # Standstill telemetry reads up to about 0.15 m/s; AEB cannot engage below 5 km/h.
    assert 0.15 < _HMI_STOPPED_SPEED_MS < _RESUME_MS


def test_visual_cue_starts_on_the_tick_the_sound_starts():
    seq = [(True, False)] * 4
    prev = False
    for (warn, brake), shown in zip(seq, _cue_trace(seq)):
        action, prev = _hmi_sound_step(warn, brake, prev)
        assert shown == (action == "start")


def test_one_tick_pulse_shows_no_visual_and_no_sound():
    assert not any(_cue_trace([(True, False)] + [(False, False)] * 10))


def test_visual_cue_outlives_the_cue_by_the_off_delay():
    dt = 1 / 30
    cue = _cue_trace([(False, True)] * 5 + [(False, False)] * 20, dt)
    held = sum(cue[5:]) * dt
    assert _HMI_CUE_OFF_DELAY_S - 2 * dt <= held <= _HMI_CUE_OFF_DELAY_S + dt
    assert cue[-1] is False


def test_off_delay_covers_two_ui_polls():
    # The main window polls every 100 ms; a sounding pulse must land in at least one poll.
    assert _HMI_CUE_OFF_DELAY_S >= 0.2


def test_aeb_visuals_read_the_shared_cue_not_warn():
    for rel in ("ui/main_window/window.py", "core/sending_thread/visualization_bar.py"):
        src = (_ROOT / rel).read_text(encoding="utf-8")
        assert "AEB_cue" in src, rel
        assert "data.AEB_warn" not in src and '"AEB_warn", False' not in src, rel
