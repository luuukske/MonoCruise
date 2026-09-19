"""Invert live g_brake_intensity so the sent pedal matches the 1.1 tune.

The game multiplies brake force by the cvar (1/3 left, 1 centre, 3 right).
``sent = min(1, logical * 1.1 / I)``. Unreadable files behave as I=1.0.
See README.md.
"""

from __future__ import annotations

import logging
import time

from .reader import read_selected_profile

log = logging.getLogger(__name__)

TUNE_BRAKE_INTENSITY: float = 1.1
DEFAULT_BRAKE_INTENSITY: float = 1.0
_REFRESH_S: float = 2.0
_I_MIN: float = 1.0 / 3.0
_I_MAX: float = 3.0


def clamp_brake_intensity(raw: float | None) -> float:
    """Clamp a profile cvar into the slider range. Unreadable becomes 1.0."""
    if raw is None or raw <= 0.0:
        return DEFAULT_BRAKE_INTENSITY
    return min(max(float(raw), _I_MIN), _I_MAX)


def apply_brake_intensity(logical: float, intensity: float | None) -> float:
    """Logical [0, 1] pedal to the value written into SCS controls."""
    p = min(max(float(logical), 0.0), 1.0)
    if p <= 0.0:
        return 0.0
    scale = TUNE_BRAKE_INTENSITY / clamp_brake_intensity(intensity)
    if abs(scale - 1.0) < 1e-12:
        return p
    return min(p * scale, 1.0)


def learn_decel_scale(intensity: float | None) -> float:
    """Scale load-corrected decel into the I=1.1 units the brake baseline uses."""
    return TUNE_BRAKE_INTENSITY / clamp_brake_intensity(intensity)


class BrakeIntensityCache:
    """Refresh ``g_brake_intensity`` a few times a second, not every control tick."""

    def __init__(self) -> None:
        self._value = DEFAULT_BRAKE_INTENSITY
        self._last_mono = 0.0
        self._have = False

    def get(self, game: str | None = None) -> float:
        now = time.monotonic()
        if self._have and now - self._last_mono < _REFRESH_S:
            return self._value
        self._last_mono = now
        self._have = True
        try:
            settings = read_selected_profile(game, live_shifter_type=None)
            if settings.g_brake_intensity is not None:
                self._value = clamp_brake_intensity(settings.g_brake_intensity)
        except Exception:
            log.debug("brake intensity refresh failed", exc_info=True)
        return self._value
