"""Speed-scheduled acceleration ceiling for the CC bid. See core/longitudinal/README.md."""

from __future__ import annotations

import math
from dataclasses import dataclass

# Above any speed-error bid the cruise PID produces. Sport returns this instead
# of a comfort curve, so the pedal goes to the floor until the set speed is close.
_UNCAPPED_BID_MS2: float = 50.0


@dataclass(frozen=True, slots=True)
class AccelProfile:
    """One driver-selectable acceleration style."""

    label: str
    launch_ms2: float
    half_ms: float
    cruise_ms2: float
    rise_jerk_ms3: float
    # No shape, settings rail or rise limit. The speed error is the bid.
    uncapped: bool = False


# Same curve family for both comfort profiles, so Efficiency feels like a softer
# Normal at every speed instead of a different shape.
EFFICIENCY = AccelProfile(
    label="Efficiency",
    launch_ms2=1.50,
    half_ms=32.0 / 3.6,
    cruise_ms2=0.35,
    rise_jerk_ms3=1.0,
)

NORMAL = AccelProfile(
    label="Normal",
    launch_ms2=2.20,
    half_ms=45.0 / 3.6,
    cruise_ms2=0.50,
    rise_jerk_ms3=2.0,
)

SPORT = AccelProfile(
    label="Sport",
    launch_ms2=_UNCAPPED_BID_MS2,
    half_ms=45.0 / 3.6,
    cruise_ms2=_UNCAPPED_BID_MS2,
    rise_jerk_ms3=math.inf,
    uncapped=True,
)

PROFILES: tuple[AccelProfile, ...] = (EFFICIENCY, NORMAL, SPORT)
PROFILE_LABELS: tuple[str, ...] = tuple(p.label for p in PROFILES)
DEFAULT_PROFILE: AccelProfile = NORMAL

_BY_KEY: dict[str, AccelProfile] = {p.label.lower(): p for p in PROFILES}


def resolve_profile(name: object) -> AccelProfile:
    """Look up a profile by label, case-insensitively. Unknown names fall back."""
    if isinstance(name, str):
        found = _BY_KEY.get(name.strip().lower())
        if found is not None:
            return found
    return DEFAULT_PROFILE


def shape_ceiling_ms2(speed_ms: float, profile: AccelProfile) -> float:
    """Comfort law: what a driver wants to feel at this speed.

    Flat at rest, halfway to the cruise value at `half_ms`, smooth everywhere.
    """
    try:
        v = float(speed_ms)
    except (TypeError, ValueError):
        v = math.nan
    if not math.isfinite(v):
        # Broken speed telemetry: bid the least, never the most.
        return 0.0 if profile.uncapped else profile.cruise_ms2
    if profile.uncapped:
        return _UNCAPPED_BID_MS2
    x = max(v, 0.0) / profile.half_ms
    return profile.cruise_ms2 + (profile.launch_ms2 - profile.cruise_ms2) / (1.0 + x * x)


def rise_limited_ms2(
    wanted_ms2: float,
    prev_bid_ms2: float | None,
    profile: AccelProfile,
    dt: float,
) -> float:
    """Bound how fast the bid may climb. Falls are never limited (brake authority).

    A None `prev_bid_ms2` means the first commanding tick, which ramps from zero.
    """
    if profile.uncapped:
        return wanted_ms2
    prev = 0.0 if prev_bid_ms2 is None else float(prev_bid_ms2)
    if not math.isfinite(prev):
        prev = 0.0
    # Clamping against max(prev, 0) is load-bearing: a raw prev of -0.5 would
    # otherwise hold the bid negative for several ticks and read as phantom brake.
    floor = max(prev, 0.0)
    if wanted_ms2 <= floor:
        return wanted_ms2
    step = profile.rise_jerk_ms3 * max(float(dt), 0.0)
    return min(wanted_ms2, floor + step)
