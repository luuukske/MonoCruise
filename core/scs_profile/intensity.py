"""Invert live g_brake_intensity so the sent pedal matches the 1.1 tune.

The game multiplies brake force by the cvar (1/3 left, 1 centre, 3 right).
Mapper and ACC send ``min(1, logical * 1.1 / I)``. AEB and em_stop pass
``full_authority`` so the logical pedal is the game axis. AEB's capacity is
``tune_max * I / 1.1``, the physical full-pedal decel. I below 1.0 cannot
be fully recovered; warn while AEB is enabled. See README.md.
"""

from __future__ import annotations

import logging
import time

from .reader import read_selected_profile

log = logging.getLogger(__name__)

TUNE_BRAKE_INTENSITY: float = 1.1
DEFAULT_BRAKE_INTENSITY: float = 1.0
AEB_UNSAFE_INTENSITY: float = 1.0
_REFRESH_S: float = 2.0
_I_MIN: float = 1.0 / 3.0
_I_MAX: float = 3.0
_AEB_LOW_I_WARN_S: float = 60.0 * 60.0


def clamp_brake_intensity(raw: float | None) -> float:
    """Clamp a profile cvar into the slider range. Unreadable becomes 1.0."""
    if raw is None or raw <= 0.0:
        return DEFAULT_BRAKE_INTENSITY
    return min(max(float(raw), _I_MIN), _I_MAX)


def apply_brake_intensity(
    logical: float,
    intensity: float | None,
    *,
    full_authority: bool = False,
) -> float:
    """Logical [0, 1] pedal to the value written into SCS controls.

    ``full_authority`` writes the logical pedal: AEB capacity is already physical.
    """
    p = min(max(float(logical), 0.0), 1.0)
    if p <= 0.0:
        return 0.0
    if full_authority:
        return p
    scale = TUNE_BRAKE_INTENSITY / clamp_brake_intensity(intensity)
    if abs(scale - 1.0) < 1e-12:
        return p
    return min(p * scale, 1.0)


def learn_decel_scale(intensity: float | None) -> float:
    """Scale load-corrected decel into the I=1.1 units the brake baseline uses."""
    return TUNE_BRAKE_INTENSITY / clamp_brake_intensity(intensity)


def aeb_available_decel_scale(intensity: float | None) -> float:
    """Scale I=1.1 capacity into the physical full-pedal decel AEB can get."""
    return clamp_brake_intensity(intensity) / TUNE_BRAKE_INTENSITY


def aeb_max_brake_ms2(tune_max: float, intensity: float | None) -> float:
    """Physical decel at pedal 1.0 for AEB planning and the AEB controller."""
    return max(float(tune_max), 0.0) * aeb_available_decel_scale(intensity)


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


class LowBrakeIntensityAebWarning:
    """Hourly popup when I is below 1.0 and AEB is enabled. See README.md."""

    def __init__(self) -> None:
        self._last_popup_mono: float = 0.0

    def tick(
        self,
        intensity: float,
        aeb_enabled: bool,
        now: float | None = None,
    ) -> bool:
        """Return True after firing. No-op when AEB is off or I is at least 1.0."""
        if not aeb_enabled:
            return False
        if clamp_brake_intensity(intensity) >= AEB_UNSAFE_INTENSITY:
            return False
        t = time.monotonic() if now is None else float(now)
        if self._last_popup_mono > 0.0 and t - self._last_popup_mono < _AEB_LOW_I_WARN_S:
            return False
        self._last_popup_mono = t
        log.warning(
            "Braking intensity is below 100%. This reduces "
            "AEB effectiveness.",
            extra={"popup": True},
        )
        log.warning(
            "AEB is enabled with g_brake_intensity %.3f; even a full brake "
            "pedal cannot recover the missing force",
            float(intensity),
        )
        return True
