"""A lead that is not constraining must not cap the driver's acceleration profile.

The IIDM law gives a far lead about its free-road accel, `a_max (1 - (v / v0)^delta)`
with `a_max` 1.5, so every tracked lead out to tracker range held Sport and Normal's
low-speed pull near 1.5 m/s2. While the law already grants nearly all of that, the
command is lifted toward the profile's free-road bid. Once the lead starts to bind the
weight is zero and the law is untouched. See core/cruise_control_thread/README.md.
"""

from __future__ import annotations

import math

# Fraction of the free-road accel the law grants: no lift at or below ZERO, full at FULL.
# Equal-speed lead at gap level 3: 0.91 at 60 m and 0.97 at 100 m from 30 km/h.
LIFT_ZERO_FRAC: float = 0.88
LIFT_FULL_FRAC: float = 0.98


def free_road_ms2(cfg, v_ego_ms: float) -> float:
    """IIDM's own acceleration with nothing ahead."""
    ratio = min(max(v_ego_ms, 0.0) / max(cfg.v0_ms, 1e-6), 1.0)
    return cfg.a_max_ms2 * (1.0 - ratio ** cfg.delta)


def lift_weight(cfg, a_law_ms2: float, v_ego_ms: float) -> float:
    """1 while the law grants its free-road accel, 0 once the lead constrains it; C1."""
    free = free_road_ms2(cfg, v_ego_ms)
    if free <= 0.05 or not math.isfinite(a_law_ms2):
        return 0.0
    x = (a_law_ms2 / free - LIFT_ZERO_FRAC) / (LIFT_FULL_FRAC - LIFT_ZERO_FRAC)
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def lift(cfg, a_law_ms2: float, free_road_bid_ms2: float | None, v_ego_ms: float,
         is_emergency: bool) -> float:
    """The law, raised toward the free-road bid by the weight. Never lowers the law."""
    if is_emergency or free_road_bid_ms2 is None or not math.isfinite(free_road_bid_ms2):
        return a_law_ms2
    if free_road_bid_ms2 <= a_law_ms2:
        return a_law_ms2
    return a_law_ms2 + lift_weight(cfg, a_law_ms2, v_ego_ms) * (free_road_bid_ms2 - a_law_ms2)
