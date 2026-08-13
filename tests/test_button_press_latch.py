"""A tap must survive resampling by the downstream threads.

main_pedal_thread publishes a level that cruise_control_thread resamples on its
own clock. Without a minimum hold, a tap shorter than that clock disappears
between samples. See core/main_pedal_thread/README.md.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# CI does not install pygame; this test only needs the latch helpers.
sys.modules.setdefault("pygame", MagicMock())

from core.main_pedal_thread.thread import _BUTTON_MIN_HOLD_S, MainPedalThread

NAME = "cc_inc_button"
LONG_FIRST_S = 0.3


class _Latch:
    """Drives only the press-latch part of MainPedalThread, on a fake clock."""

    def __init__(self) -> None:
        self.thread = MainPedalThread.__new__(MainPedalThread)
        self.thread._button_raw_prev = {}
        self.thread._button_hold_until = {}
        self.thread._button_press_counts = {}
        self.thread._button_source_counts = {}

    def step(self, raw: bool, now: float) -> bool:
        t = self.thread
        if raw and not t._button_raw_prev.get(NAME, False):
            t._button_press_counts[NAME] = t._button_press_counts.get(NAME, 0) + 1
            t._button_hold_until[NAME] = now + _BUTTON_MIN_HOLD_S
        t._button_raw_prev[NAME] = raw
        return raw or now < t._button_hold_until.get(NAME, 0.0)

    @property
    def presses(self) -> int:
        return self.thread._button_press_counts.get(NAME, 0)


def _drive(taps, duration, pedal_period, source_period=0.001):
    """Return the level the pedal thread publishes, sampled on its own clock."""
    out = []
    t = 0.0
    latch = _Latch()
    while t <= duration:
        raw = any(start <= t < start + width for start, width in taps)
        out.append((t, latch.step(raw, t)))
        t += pedal_period
    return out, latch.presses


def _resample(samples, period, phase=0.0):
    out = []
    idx = 0
    value = False
    t = samples[0][0] + phase
    while t <= samples[-1][0]:
        while idx < len(samples) and samples[idx][0] <= t:
            value = samples[idx][1]
            idx += 1
        out.append((t, value))
        t += period
    return out


def _fsm_actions(samples, long_threshold=LONG_FIRST_S):
    actions = 0
    pressed_at = None
    long_press = False
    for now, held in samples:
        if held:
            if pressed_at is None:
                pressed_at = now
            if not long_press and now - pressed_at > long_threshold:
                long_press = True
                actions += 1
        elif pressed_at is not None:
            if not long_press:
                actions += 1
            else:
                long_press = False
            pressed_at = None
    return actions


@pytest.mark.parametrize("tap_ms", [15, 25, 40, 60, 90])
@pytest.mark.parametrize("phase_ms", [0, 3, 7, 11, 14])
def test_single_tap_survives_downstream_resampling(tap_ms, phase_ms):
    """Whatever the phase, one tap must produce exactly one FSM action."""
    pedal_period = 0.0154
    cruise_period = 0.0154
    published, presses = _drive(
        [(0.2, tap_ms / 1000.0)], 1.2, pedal_period,
    )
    assert presses == 1, "the pedal thread must latch the press"
    cruise = _resample(published, cruise_period, phase=phase_ms / 1000.0)
    assert _fsm_actions(cruise) == 1


@pytest.mark.parametrize("cruise_ms", [15, 20, 25, 30])
def test_tap_survives_a_slow_cruise_thread(cruise_ms):
    """The cruise thread does far more work per tick than the pedal thread."""
    published, presses = _drive([(0.2, 0.030)], 1.2, 0.0154)
    assert presses == 1
    cruise = _resample(published, cruise_ms / 1000.0)
    assert _fsm_actions(cruise) == 1


def test_long_press_timing_is_not_extended():
    """The hold deadline runs from the press, so a long press is untouched."""
    latch = _Latch()
    held_after_release = latch.step(True, 0.0)
    assert held_after_release is True
    for t in (0.05, 0.10, 0.20, 0.40):
        assert latch.step(True, t) is True
    # Released well past the hold window: no extension at all.
    assert latch.step(False, 0.41) is False


def test_short_press_is_extended_to_the_minimum():
    latch = _Latch()
    latch.step(True, 0.0)
    assert latch.step(False, 0.005) is True, "still latched"
    assert latch.step(False, _BUTTON_MIN_HOLD_S - 0.001) is True
    assert latch.step(False, _BUTTON_MIN_HOLD_S + 0.001) is False


def test_rapid_tapping_counts_every_press():
    taps = [(0.2 + i * 0.12, 0.040) for i in range(8)]
    published, presses = _drive(taps, 1.6, 0.0154)
    assert presses == 8
    cruise = _resample(published, 0.0154)
    assert _fsm_actions(cruise) == 8


def test_latch_reset_drops_a_pending_press():
    latch = _Latch()
    latch.step(True, 0.0)
    latch.thread._reset_button_latch()
    assert latch.step(False, 0.005) is False
