"""Driver-facing speed text for ETS2 (km/h) and ATS (mph).

The controller setpoint stays km/h. ATS converts at this edge with the
international mile (1 mph = 1.609344 km/h). The PID already divides that
by 3.6 to get m/s. See core/cruise_control_thread/README.md.
"""

from __future__ import annotations

import math

from core.settings import Settings

# International statute mile. 1609.344 m / 1000 m.
MPH_TO_KMH = 1.609344
KMH_PER_MS = 3.6

# Same walls CruiseController uses, so an mph step cannot land outside them.
_SET_MIN_KMH = 30.0
_SET_MAX_KMH = 130.0
# Same walls as the settings box (ui/main_window/settings_panel.py).
_LIMIT_MIN_KMH = 60.0
_LIMIT_MAX_KMH = 130.0


def uses_mph() -> bool:
    try:
        game = int(Settings.last_game)
    except (TypeError, ValueError):
        return False
    return game == 2


def unit_label() -> str:
    return "mph" if uses_mph() else "km/h"


def display_from_kmh(kmh: float) -> int:
    value = float(kmh)
    if uses_mph():
        return int(round(value / MPH_TO_KMH))
    return int(round(value))


def display_from_ms(speed_ms: float) -> int:
    return display_from_kmh(float(speed_ms) * KMH_PER_MS)


def kmh_from_display(shown: int) -> float:
    if uses_mph():
        return float(shown) * MPH_TO_KMH
    return float(shown)


def format_kmh(kmh: float | None) -> str:
    unit = unit_label()
    if kmh is None:
        return f"-- {unit}"
    return f"{display_from_kmh(kmh)} {unit}"


def global_limit_bounds() -> tuple[int, int]:
    return _display_bounds(_LIMIT_MIN_KMH, _LIMIT_MAX_KMH)


def quantize_speed_kmh(speed_ms: float, glim_kmh: float | None) -> float:
    """Current speed, snapped to the driver's unit, as km/h inside the set-speed walls."""
    shown = display_from_ms(speed_ms)
    lo, hi = _set_bounds(glim_kmh)
    return kmh_from_display(max(lo, min(hi, shown)))


def step_setpoint_kmh(current_kmh: float, delta: float, glim_kmh: float | None) -> float:
    """One button step in the driver's unit. Same grid rule as CruiseController.change_target_kmh."""
    shown = display_from_kmh(current_kmh)
    inc = int(round(float(delta)))
    abs_inc = abs(inc)
    if abs_inc == 0:
        return float(current_kmh)
    if abs_inc >= 5:
        if inc > 0:
            nxt = ((shown // abs_inc) + 1) * abs_inc
        elif shown % abs_inc == 0:
            nxt = ((shown // abs_inc) - 1) * abs_inc
        else:
            nxt = (shown // abs_inc) * abs_inc
    else:
        nxt = shown + inc
    lo, hi = _set_bounds(glim_kmh)
    nxt = max(lo, min(hi, nxt))
    # A no-op must not replace a legacy km/h value with the nearest mph.
    if nxt == shown:
        return float(current_kmh)
    return kmh_from_display(nxt)


def _display_bounds(min_kmh: float, max_kmh: float) -> tuple[int, int]:
    if not uses_mph():
        return int(min_kmh), int(max_kmh)
    lo = math.ceil(min_kmh / MPH_TO_KMH - 1e-9)
    hi = math.floor(max_kmh / MPH_TO_KMH + 1e-6)
    # n * factor / factor can land just under n. Let that exact step back in.
    if (hi + 1) * MPH_TO_KMH <= max_kmh + 1e-4:
        hi += 1
    if lo > hi:
        hi = lo
    return lo, hi


def _set_bounds(glim_kmh: float | None) -> tuple[int, int]:
    upper = _SET_MAX_KMH
    if glim_kmh is not None:
        try:
            g = float(glim_kmh)
        except (TypeError, ValueError):
            g = upper
        else:
            if math.isfinite(g):
                upper = min(upper, g)
    return _display_bounds(_SET_MIN_KMH, upper)
