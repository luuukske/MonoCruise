"""Accumulated in-game usage time and the support-prompt schedule.

Pure logic: no Qt and no thread of its own. The Qt main window ticks it from
its existing poll timer, and core/settings.py owns persistence.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Prompt N is due at N ** 2.5 * 100 hours: 100, 566, 1559, 3200, and so on.
# The steep exponent is the whole point: a long-time user is asked rarely.
PROMPT_BASE_HOURS = 100.0
PROMPT_EXPONENT = 2.5

SECONDS_PER_MINUTE = 60.0
SECONDS_PER_HOUR = 3600.0

# Settings fields this module owns. They are history, not preferences, so a
# reset to defaults must leave them alone or a 600 hour user is asked again.
RESET_EXEMPT_FIELDS = ("usage_minutes", "support_prompts_dismissed")

# A gap larger than this is machine sleep, a debugger pause or a clock step,
# never play time, so it is dropped rather than counted.
MAX_TICK_S = 5.0


def prompt_threshold_hours(prompts_dismissed: int) -> float:
    """Usage hours at which the next support prompt becomes due."""
    try:
        dismissed = max(0, int(prompts_dismissed))
    except (TypeError, ValueError):
        dismissed = 0
    return ((dismissed + 1) ** PROMPT_EXPONENT) * PROMPT_BASE_HOURS


def prompt_is_due(usage_seconds: float, prompts_dismissed: int) -> bool:
    """True once *usage_seconds* reaches the threshold for the next prompt."""
    try:
        hours = float(usage_seconds) / SECONDS_PER_HOUR
    except (TypeError, ValueError):
        return False
    return hours >= prompt_threshold_hours(prompts_dismissed)


def record_prompt_dismissed(settings) -> int:
    """Count one dismissal so the next prompt moves out to the next threshold."""
    with settings._state_lock:
        try:
            count = max(0, int(settings.support_prompts_dismissed)) + 1
        except (TypeError, ValueError):
            count = 1
        settings.support_prompts_dismissed = count
    try:
        settings.save()
    except Exception:
        logger.exception("could not persist the support prompt schedule")
    return count


class UsageTracker:
    """Accumulates connected game time onto ``Settings.usage_minutes``.

    Seconds short of a minute stay here. Crossing a minute updates the setting
    and writes the config.
    """

    def __init__(self, settings) -> None:
        self._settings = settings
        self._last_tick: float | None = None
        self._remainder_s = 0.0

    def _stored_minutes(self) -> int:
        try:
            with self._settings._state_lock:
                return max(0, int(self._settings.usage_minutes))
        except (AttributeError, TypeError, ValueError):
            return 0

    @property
    def usage_seconds(self) -> float:
        return self._stored_minutes() * SECONDS_PER_MINUTE + self._remainder_s

    @property
    def usage_hours(self) -> float:
        return self.usage_seconds / SECONDS_PER_HOUR

    def tick(self, connected: bool, now: float) -> float:
        """Add the elapsed slice while *connected*. Returns total usage seconds."""
        last = self._last_tick
        self._last_tick = now if connected else None
        if not connected or last is None:
            return self.usage_seconds

        delta = now - last
        if delta <= 0.0 or delta > MAX_TICK_S:
            return self.usage_seconds

        self._remainder_s += delta
        whole = int(self._remainder_s // SECONDS_PER_MINUTE)
        if whole > 0 and self._store(self._stored_minutes() + whole):
            self._remainder_s -= whole * SECONDS_PER_MINUTE
            self._flush()
        return self.usage_seconds

    def _store(self, total_minutes: int) -> bool:
        try:
            with self._settings._state_lock:
                self._settings.usage_minutes = int(total_minutes)
            return True
        except AttributeError:
            logger.debug("settings object cannot hold usage_minutes", exc_info=True)
            return False

    def _flush(self) -> None:
        try:
            self._settings.save()
        except Exception:
            logger.exception("could not persist accumulated usage hours")
