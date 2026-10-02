"""ACC gap-level buttons: one button cycles, two buttons step. See the README."""

from __future__ import annotations

import logging

from core.settings import Settings

logger = logging.getLogger(__name__)

_LONG_PRESS_FIRST_S = 0.3
_LONG_PRESS_REPEAT_S = 0.6

INC_BUTTON = "acc_dist_inc_button"
DEC_BUTTON = "acc_dist_dec_button"
BUTTONS = (INC_BUTTON, DEC_BUTTON)

_MIN_LEVEL = 1
_MAX_LEVEL = 4


class AccDistanceButtons:
    """Press FSM for the gap-level buttons; short steps fire per counted press."""

    def __init__(self, presses) -> None:
        self._presses = presses
        self._time_pressed: dict[str, float | None] = {INC_BUTTON: None, DEC_BUTTON: None}
        self._long_press: dict[str, bool] = {INC_BUTTON: False, DEC_BUTTON: False}
        self._last_assign_warn_mono: float = 0.0

    def tick(self, now: float, inc_held: bool, dec_held: bool) -> None:
        inc_assigned = Settings.acc_dist_inc_button is not None
        dec_assigned = Settings.acc_dist_dec_button is not None

        if not inc_assigned and not dec_assigned:
            if (inc_held or dec_held) and now - self._last_assign_warn_mono > 2.0:
                self._last_assign_warn_mono = now
                logger.info(
                    "Please assign the ACC distance button(s) in the settings",
                    extra={"popup": True},
                )
            self._reset()
            return

        # One button assigned: it cycles through the levels and wraps.
        if inc_assigned ^ dec_assigned:
            name = INC_BUTTON if inc_assigned else DEC_BUTTON
            held = inc_held if inc_assigned else dec_held
            self._tick_button(now, name, held, lambda: step_gap_level(+1, wrap=True))
            other = DEC_BUTTON if inc_assigned else INC_BUTTON
            self._time_pressed[other] = None
            self._long_press[other] = False
            return

        # Both assigned: clamped step, suppressed while both are held.
        if inc_held and dec_held:
            self._reset()
            return

        self._tick_button(now, INC_BUTTON, inc_held, lambda: step_gap_level(+1, wrap=False))
        self._tick_button(now, DEC_BUTTON, dec_held, lambda: step_gap_level(-1, wrap=False))

    def _reset(self) -> None:
        for name in BUTTONS:
            self._time_pressed[name] = None
            self._long_press[name] = False
        self._presses.discard(BUTTONS)

    def _tick_button(self, now: float, name: str, held: bool, apply) -> None:
        pressed_at = self._time_pressed[name]
        long_press = self._long_press[name]

        if held:
            if pressed_at is None:
                pressed_at = now
            held_dt = now - pressed_at
            if (not long_press and held_dt > _LONG_PRESS_FIRST_S) or (
                long_press and held_dt > _LONG_PRESS_REPEAT_S
            ):
                if not long_press:
                    self._presses.consume_one(name)
                long_press = True
                pressed_at = now
                apply()
        else:
            long_press = False
            pressed_at = None

        for _ in range(self._presses.take_short(name, held)):
            apply()

        self._time_pressed[name] = pressed_at
        self._long_press[name] = long_press


def step_gap_level(delta: int, *, wrap: bool) -> None:
    """Move the persisted ACC gap level, wrapping or clamping at the ends."""
    try:
        current = int(Settings.acc_gap_level)
    except (TypeError, ValueError):
        current = 2
    current = max(_MIN_LEVEL, min(_MAX_LEVEL, current))
    new_level = current + int(delta)
    if wrap:
        if new_level > _MAX_LEVEL:
            new_level = _MIN_LEVEL
        elif new_level < _MIN_LEVEL:
            new_level = _MAX_LEVEL
    else:
        new_level = max(_MIN_LEVEL, min(_MAX_LEVEL, new_level))
    if new_level == current:
        return
    try:
        Settings.save(values={"acc_gap_level": new_level})
    except Exception:
        logger.exception("failed to persist acc_gap_level")
        return
    logger.info("ACC gap set to %d/4", new_level)
