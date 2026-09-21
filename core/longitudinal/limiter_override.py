"""Panic bypass for a speed limit set below the speed the driver wants.

See core/longitudinal/README.md.
"""

from __future__ import annotations

# Pressed enough. The stab does not have to be a stomp to the stop.
_GAS_FLOOR = 0.72
# Minimum lift. The foot does not have to leave the pedal.
_GAS_RELEASE = 0.48
# Been on the gas. Counts on the way up to the cap.
_ARM_HOLD_S = 0.45
# Time to cross the small lift. This is the speed gate, not the travel.
_QUICK_RELEASE_S = 0.15
# Shorter than this is a one-sample dip, not a foot leaving the pedal.
_RELEASE_MIN_S = 0.05
# Time to get back on the gas. A pause and repress is too slow.
_BLIP_WINDOW_S = 0.30
# Stab only counts this close to the cap, or already over it.
_BIND_UNDER_KMH = 3.0
# Restore once speed falls below this far above the cap, after having reached it.
_REARM_ABOVE_KMH = 5.0
# One stalled tick must not complete the hold or the blip by itself.
_DT_CAP_S = 0.1


class LimiterPanicOverride:
    """Latch that drops the limiter after a quick lift-and-press at the cap."""

    def __init__(self) -> None:
        self.overridden = False
        self._floor_hold_s = 0.0
        self._off_floor_s = 0.0
        self._window_s: float | None = None
        self._seen_above = False

    def reset(self) -> None:
        self.overridden = False
        self._floor_hold_s = 0.0
        self._off_floor_s = 0.0
        self._window_s = None
        self._seen_above = False

    def update(
        self,
        *,
        gas: float,
        speed_kmh: float,
        limit_kmh: float | None,
        limiter_active: bool,
        dt: float,
    ) -> bool:
        dt = min(max(float(dt), 0.0), _DT_CAP_S)
        if not limiter_active or limit_kmh is None:
            self.reset()
            return False

        gas = float(gas)
        speed_kmh = float(speed_kmh)
        limit_kmh = float(limit_kmh)
        if self.overridden:
            return self._tick_latched(speed_kmh, limit_kmh)

        on_floor = gas >= _GAS_FLOOR
        released = gas <= _GAS_RELEASE
        binding = speed_kmh >= limit_kmh - _BIND_UNDER_KMH

        if on_floor:
            self._off_floor_s = 0.0
            if self._window_s is not None:
                if self._window_s >= _RELEASE_MIN_S:
                    self.overridden = True
                    self._seen_above = speed_kmh >= limit_kmh + _REARM_ABOVE_KMH
                    self._window_s = None
                    self._floor_hold_s = 0.0
                    return True
                self._window_s = None
            self._floor_hold_s += dt
            return False

        self._off_floor_s += dt
        if self._window_s is None:
            armed = self._floor_hold_s >= _ARM_HOLD_S
            quick = self._off_floor_s <= _QUICK_RELEASE_S
            if released and binding and armed and quick:
                self._window_s = dt
            elif released or self._off_floor_s > _QUICK_RELEASE_S:
                # A real lift that is not the gesture, or a slow roll-off, drops the arm.
                # The next press has to be held again before a blip can count.
                self._floor_hold_s = 0.0
            return False

        self._window_s += dt
        if self._window_s > _BLIP_WINDOW_S:
            self._window_s = None
            self._floor_hold_s = 0.0
        return False

    def _tick_latched(self, speed_kmh: float, limit_kmh: float) -> bool:
        line = limit_kmh + _REARM_ABOVE_KMH
        if speed_kmh >= line:
            self._seen_above = True
        elif self._seen_above:
            self.reset()
            return False
        return True
