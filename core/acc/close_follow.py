"""Close-follow warning: a timer on the time gap to the lead. See core/acc/README.md §11."""

from __future__ import annotations

from .blinker import EGO_FRONT_OFFSET_M

# Time gap (bumper gap over ego speed) and how long it must be held before the warning shows.
GAP_NEAR_S: float = 0.4
GAP_FAR_S: float = 0.5
DELAY_NEAR_S: float = 1.0
DELAY_FAR_S: float = 5.0
# Only steady following counts: catching up or being pulled away from is not tailgating.
REL_SPEED_MAX_MS: float = 10.0 / 3.6
# A time gap means nothing at rest; below this ego is treated as not following.
MIN_SPEED_MS: float = 1.0
# A dropped lead or a speed blip shorter than this neither clears the warning nor restarts the timer.
RELEASE_S: float = 0.5
_DT_MAX_S: float = 0.2


def delay_for_gap(gap_s: float) -> float:
    """Seconds a time gap must be held: 1 s at 0.4 s and under, rising to 5 s at 0.5 s."""
    frac = (gap_s - GAP_NEAR_S) / (GAP_FAR_S - GAP_NEAR_S)
    return DELAY_NEAR_S + (DELAY_FAR_S - DELAY_NEAR_S) * max(0.0, min(1.0, frac))


class CloseFollow:
    """Latches ``warn`` once ego has followed closer than 0.5 s for the gap's delay."""

    def __init__(self) -> None:
        self.warn = False
        self._held_s = 0.0
        self._last_t: float | None = None
        self._last_close_t: float | None = None

    def reset(self) -> None:
        self.warn = False
        self._held_s = 0.0
        self._last_t = None
        self._last_close_t = None

    def step(
        self, now: float, dist_m: float | None, ego_speed_ms: float, rel_speed_ms: float = 0.0,
    ) -> bool:
        """Advance to ``now``. ``dist_m`` is the lead's ``LeadInfo.dist_m``, None without a lead."""
        dt = 0.0 if self._last_t is None else max(0.0, min(now - self._last_t, _DT_MAX_S))
        self._last_t = now

        gap_s = GAP_FAR_S
        if dist_m is not None and ego_speed_ms > MIN_SPEED_MS and abs(rel_speed_ms) <= REL_SPEED_MAX_MS:
            gap_s = max(0.0, dist_m - EGO_FRONT_OFFSET_M) / ego_speed_ms

        if gap_s < GAP_FAR_S:
            self._held_s = min(DELAY_FAR_S, self._held_s + dt)
            self._last_close_t = now
            # Latched until ego backs off: easing from 0.3 s to 0.45 s must not blink it off and on.
            self.warn = self.warn or self._held_s >= delay_for_gap(gap_s)
        elif self._last_close_t is None or now - self._last_close_t > RELEASE_S:
            self._held_s = 0.0
            self.warn = False
        return self.warn
