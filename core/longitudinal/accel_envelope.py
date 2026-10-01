"""Speed-scheduled acceleration ceiling for the CC bid. See core/longitudinal/README.md."""

from __future__ import annotations

import math
from dataclasses import dataclass

# Keeps the pedal just off the rail so the mapper's fast PI keeps trim authority.
# Never a per-profile share: see `core/longitudinal/README.md` for why.
HEADROOM_FRAC: float = 0.95

# Below this the capacity estimate is treated as unknown rather than as a real
# ceiling. The publisher legitimately zeroes it while disconnected or idle.
_MIN_USABLE_CAPACITY_MS2: float = 0.05


@dataclass(frozen=True, slots=True)
class AccelProfile:
    """One driver-selectable acceleration style."""

    label: str
    launch_ms2: float
    knee_ms: float
    taper_power: float
    floor_ms2: float
    rise_jerk_ms3: float


# The knee sits just above the launch itself so the taper starts early: holding
# a flat ceiling out to 30 km/h asked for more pedal than the speed warranted.
EFFICIENCY = AccelProfile(
    label="Efficiency",
    launch_ms2=1.05,
    knee_ms=12.0 / 3.6,
    taper_power=0.50,
    floor_ms2=0.30,
    rise_jerk_ms3=0.5,
)

# launch_ms2 * knee_ms is the whole tail when taper_power is 1.0, so raising the
# launch and lowering the knee in step leaves every value above 35 km/h identical.
NORMAL = AccelProfile(
    label="Normal",
    launch_ms2=2.00,
    knee_ms=26.25 / 3.6,
    taper_power=1.00,
    floor_ms2=0.45,
    rise_jerk_ms3=1.3,
)

# Deliberately above any rig's capability, so the capability guard binds instead
# and the driver meets a ceiling only where the engine does. See the README.
SPORT = AccelProfile(
    label="Sport",
    launch_ms2=2.50,
    knee_ms=40.0 / 3.6,
    taper_power=0.45,
    floor_ms2=1.20,
    rise_jerk_ms3=2.5,
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
    """Comfort law: what a driver wants to feel at this speed, ignoring capability."""
    try:
        v = float(speed_ms)
    except (TypeError, ValueError):
        return profile.floor_ms2
    if not math.isfinite(v):
        # Broken speed telemetry: bid the least, never the most.
        return profile.floor_ms2
    if v <= profile.knee_ms:
        return profile.launch_ms2
    taper = profile.launch_ms2 * (profile.knee_ms / v) ** profile.taper_power
    return max(profile.floor_ms2, taper)


def usable_capacity_ms2(capacity_ms2: object) -> float | None:
    """The published capacity estimate, or None when it carries no information."""
    try:
        cap = float(capacity_ms2)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cap) or cap <= _MIN_USABLE_CAPACITY_MS2:
        return None
    return cap


def envelope_ms2(
    speed_ms: float,
    profile: AccelProfile,
    capacity_ms2: object = None,
) -> float:
    """Positive accel ceiling: the comfort shape, bounded by what the truck can do."""
    shape = shape_ceiling_ms2(speed_ms, profile)
    cap = usable_capacity_ms2(capacity_ms2)
    if cap is None:
        return shape
    return min(shape, HEADROOM_FRAC * cap)


def rise_limited_ms2(
    wanted_ms2: float,
    prev_bid_ms2: float | None,
    profile: AccelProfile,
    dt: float,
) -> float:
    """Bound how fast the bid may climb. Falls are never limited (brake authority).

    A None `prev_bid_ms2` means the first commanding tick, which ramps from zero.
    """
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
