"""CC must not read MonoCruise's own brake, echoed late by the game, as the driver's."""

from __future__ import annotations

from core.cruise_control_thread.thread import _CC_GAME_BRAKE_DISENGAGE
from core.sending_thread.recent_brake import RECENT_BRAKE_WINDOW_S, RecentBrakeOutputs

_TICK_S = 0.01
# Worst lag between the sent pedal and gameBrake in the brake debug log.
_GAME_LAG_S = 0.11


def _worst_excess(sent: list[float], lag_s: float) -> float:
    """Largest gameBrake excess over the window max, with the game echoing sent late."""
    recent = RecentBrakeOutputs()
    lag_ticks = round(lag_s / _TICK_S)
    worst = 0.0
    for i, value in enumerate(sent):
        recent.push(i * _TICK_S, value)
        game_brake = sent[max(0, i - lag_ticks)]
        worst = max(worst, game_brake - max(recent.values()))
    return worst


def test_a_fast_release_of_the_own_brake_is_not_an_in_game_brake():
    """AEB slam let go in one tick: the game still reads 1.0 for 110 ms after."""
    sent = [0.0] * 20 + [1.0] * 30 + [0.03] * 100
    assert _worst_excess(sent, _GAME_LAG_S) <= 0.0


def test_a_brake_stab_shorter_than_the_lag_is_covered():
    sent = [0.0] * 20 + [1.0] * 3 + [0.0] * 100
    assert _worst_excess(sent, _GAME_LAG_S) <= 0.0


def test_the_window_outlasts_a_slow_game_frame():
    """A game at 5 fps echoes the pedal 0.2 s late, plus the telemetry poll."""
    sent = [0.0] * 20 + [0.6] * 50 + [0.0] * 100
    assert _worst_excess(sent, 0.3) <= 0.0
    assert RECENT_BRAKE_WINDOW_S >= 3 * _GAME_LAG_S


def test_an_in_game_brake_is_still_seen_once_the_window_has_passed():
    recent = RecentBrakeOutputs()
    for i in range(30):
        recent.push(i * _TICK_S, 0.5)
    t_release = 30 * _TICK_S
    recent.push(t_release, 0.0)
    assert 1.0 - max(recent.values()) > _CC_GAME_BRAKE_DISENGAGE
    recent.push(t_release + RECENT_BRAKE_WINDOW_S + _TICK_S, 0.0)
    assert max(recent.values()) == 0.0


def test_the_window_is_time_based_at_any_polling_rate():
    """At 10 Hz three ticks were 300 ms, at 100 Hz 30 ms: the bug was rate dependent."""
    for tick_s in (0.1, 0.01, 0.004):
        recent = RecentBrakeOutputs()
        recent.push(0.0, 1.0)
        t = 0.0
        while t < RECENT_BRAKE_WINDOW_S - tick_s:
            t += tick_s
            recent.push(t, 0.0)
        assert max(recent.values()) == 1.0
        recent.push(RECENT_BRAKE_WINDOW_S + 2 * tick_s, 0.0)
        assert max(recent.values()) == 0.0


def test_a_stalled_clock_keeps_the_newest_sample():
    recent = RecentBrakeOutputs()
    recent.push(0.0, 0.4)
    recent.push(10.0, 0.7)
    assert recent.values() == (0.7,)
