"""
ACC in-lane scoring — four pure components accumulated per vehicle id.

Follows SCORING_REFERENCE.md (legacy ETS2radar) faithfully.  Constants
are kept as named variables so the derivation is legible; the only
semantic departure is an optional ``dt`` multiplier in :func:`accumulate`
so the loop can run at any cadence (scaled to the legacy 30 Hz tick
rate).  See ``core/acc/AGENTS.md`` for the full mapping.

Components (per frame, before speed/dt scaling):
    offset   — arc-crossing lateral offset from ego centerline.  Gaussian
               with σ = 2.25 m, distance-amplified, clamped to [-1, +1],
               then scaled by an outer angle-weighted multiplier and
               added to a baseline increment.
    yaw      — heading mismatch.  `(2^(-(|Δyaw|/90°)^5) - 1) × 1.5`.
    path     — ego-path intersection.  `1.03^(-d_m) × slow_amp` with a
               blinker amplitude reduction; capped at +5 in-path /
               -4 out-of-path (× 0.6 when out).
    angle    — reserved.  Currently 0.0 (legacy had it disabled too).

Accumulation uses the legacy `[-5, +15]` clamp × target-speed multiplier:
    ``score += total × max((|v_target|/90 m/s)^0.8, 0.5) × dt × 30``
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Score clamp — asymmetric so lock is fast and unlock is slow.
_SCORE_MIN: float = -5.0
_SCORE_MAX: float = 20.0

# Offset Gaussian width (metres).  σ = 2.25 m legacy; the
# zero-crossing of ``offset_raw`` sits at |x| ≈ 2.58 m at angle_amp = 1.
_OFFSET_SIGMA_M: float = 2.25

# Path decay base.  Legacy uses ``1.03^(-d)`` with d in metres.
# Equivalent to ``exp(-d · ln(1.03))``; keep the base explicit so the
# formula is bit-for-bit identical to the legacy integer-tick maths.
_PATH_DECAY_BASE: float = 1.03

# Slow-speed path amplifier: ``slow_amp = 1.4 + (kmh / 100) × 4.1``.
# Reference table in SCORING_REFERENCE §8.3.2.
_SLOW_AMP_BASE: float = 1.4
_SLOW_AMP_SLOPE: float = 4.1
_SLOW_AMP_REF_KMH: float = 100.0

# In-/out-of-path caps.  Reference: ``min(base, 5.0)`` and
# ``-min(base × 0.6, 4.0)``.
_PATH_IN_CAP: float = 5.0
_PATH_OUT_CAP: float = 4.0
_PATH_OUT_GAIN: float = 0.6

# Yaw component shape.  Kept legacy.
_YAW_SCALE_DEG: float = 90.0
_YAW_POWER: float = 5.0
_YAW_GAIN: float = 1.5

# Offset baseline increments (legacy §8.1).
OFFSET_BASELINE_HIT: float = 0.0
OFFSET_BASELINE_NO_ARC_HIT: float = -0.40
OFFSET_BASELINE_NO_HISTORY: float = -0.16

# Speed multiplier denominator — **90 m/s** (legacy).  NOT 25 m/s (90 km/h).
# Effective floor 0.5 applies at every realistic road speed; anything
# >60 m/s lifts it linearly.
_SPEED_MULT_REF_MS: float = 90.0
_SPEED_MULT_EXP: float = 0.8
_SPEED_MULT_FLOOR: float = 0.5

# Legacy tick rate.  :func:`accumulate` scales its dt input by this so
# at 10 Hz the per-frame delta matches the legacy integer-tick maths.
_LEGACY_RATE_HZ: float = 10.0

# Offset weight — boosts lane-offset influence so vehicles in ego's lane
# are detected even when the ego path arc doesn't geometrically intersect
# them (e.g. during slight steering, distant lead).  Path/yaw unchanged.
_OFFSET_WEIGHT: float = 1.5

# Path weight — scales ego-path intersection relative to offset/yaw.
# Reduced below 1.0 so the path component reacts slower to arc changes.
_PATH_WEIGHT: float = 0.7


@dataclass(slots=True)
class ScoreComponents:
    """Per-frame contribution to a track's score (pre speed / dt scaling)."""
    offset: float = 0.0
    yaw: float = 0.0
    path: float = 0.0
    angle: float = 0.0   # reserved — currently always 0.0

    def total(self) -> float:
        return self.offset + self.yaw + self.path + self.angle


def _distance_amp(dist_m: float) -> float:
    """Offset distance amplifier from SCORING_REFERENCE §8.1.

    ``distance_amp = [2^(-d/100) + 8/(d+3) - 1] / 3 + 1``

    Values: 5m→1.32, 10m→1.18, 30m→1.02, 50m→0.94, 100m→0.86, 150m→0.80.
    """
    d = max(dist_m, 0.0)
    return (math.pow(2.0, -d / 100.0) + 8.0 / (d + 3.0) - 1.0) / 3.0 + 1.0


def offset_component(
    offset_m: float,
    dist_m: float,
    angle_amp: float = 1.0,
    baseline: float = OFFSET_BASELINE_HIT,
) -> float:
    """Offset score — SCORING_REFERENCE §8.1.

    offset_m  — arc-crossing lateral from ego centerline *after* blinker
                 bias has been subtracted by the caller (``offset - blinker·4.5``).
    dist_m    — longitudinal distance to scoring closest-approach point.
    angle_amp — `2^(-(normalized_arc_angle / 0.06)²)` from trail arc fit;
                 pass 1.0 when a trail arc fit isn't available.
    baseline  — per-frame baseline: 0.0 on normal arc hit, -0.40 when
                 an arc was fit but didn't cross ego row, -0.16 when
                 history was too short to fit.
    """
    x = offset_m / _OFFSET_SIGMA_M
    gauss = math.exp(-(x * x) * math.log(2.0))         # 2^(-(x/σ)²)
    raw = gauss * 2.5 * angle_amp - 1.0
    clamped = max(-1.0, min(1.0, raw * _distance_amp(dist_m)))
    outer = 1.5 * (angle_amp * 0.4 + 0.6)
    return baseline + clamped * outer


def yaw_component(yaw_diff_deg: float) -> float:
    """Heading-mismatch penalty: `(2^(-(d/90)^5) - 1) × 1.5`.

    Near 0° → ~0.  Past ±90° → ≈ -1.5.  Symmetric.
    """
    d = abs(yaw_diff_deg)
    if d > 4.0 * _YAW_SCALE_DEG:
        return -_YAW_GAIN
    ratio = d / _YAW_SCALE_DEG
    return (math.pow(2.0, -math.pow(ratio, _YAW_POWER)) - 1.0) * _YAW_GAIN


def path_component(
    dist_m: float,
    ego_speed_kmh: float,
    in_path: bool,
    blinker_offset: float = 0.0,
) -> float:
    """Ego-path intersection score — SCORING_REFERENCE §8.3.2.

    blinker_offset  — signed scalar from the blinker decay curve (the
                       ACC tracker bounds it to [-1, +1], but legacy
                       allowed simultaneous left+right up to ±2 for
                       hazard blinker edge cases).  Reduces the path
                       amplitude by ``b² × 0.4`` — at |b| > √2.5 the
                       reduction exceeds 1 and `base` goes negative,
                       matching legacy behaviour exactly.
    """
    if dist_m < 0.0:
        dist_m = 0.0
    decay = math.pow(_PATH_DECAY_BASE, -dist_m)        # legacy 1.03^(-d)
    slow_amp = _SLOW_AMP_BASE + (ego_speed_kmh / _SLOW_AMP_REF_KMH) * _SLOW_AMP_SLOPE
    blinker_sq = blinker_offset * blinker_offset
    amp = slow_amp * (1.0 - blinker_sq * 0.4)
    base = decay * amp
    if in_path:
        return min(_PATH_IN_CAP, base)
    return -min(_PATH_OUT_CAP, base * _PATH_OUT_GAIN)


def speed_multiplier(target_speed_ms: float) -> float:
    """Per-frame score delta multiplier — SCORING_REFERENCE §9.

    Legacy `max((|v_target|/90 m/s)^0.8, 0.5)`.  The 0.5 floor applies
    at all realistic road speeds; above ~60 m/s (216 km/h) it lifts.
    Input is the **target vehicle's** speed, not ego's.
    """
    v = abs(target_speed_ms)
    if v <= 0.0:
        return _SPEED_MULT_FLOOR
    ratio = v / _SPEED_MULT_REF_MS
    return max(math.pow(ratio, _SPEED_MULT_EXP), _SPEED_MULT_FLOOR)


def accumulate(
    prev_score: float,
    dt: float,
    components: ScoreComponents,
    target_speed_ms: float,
    path_weight: float = _PATH_WEIGHT,
) -> float:
    """Integrate components into the running score; clamp to [-5, +15].

    Per-frame delta = `weighted_total × speed_mult(target) × dt × 30`.

    The ``× 30`` is so dt=1/30 s (legacy tick) produces the unscaled
    legacy increment; higher/lower loop rates scale proportionally.
    path_weight scales the path component relative to offset/yaw.
    """
    weighted_total = (
        components.offset * _OFFSET_WEIGHT
        + components.yaw
        + components.path * path_weight
        + components.angle
    )
    delta = weighted_total * speed_multiplier(target_speed_ms) * dt * _LEGACY_RATE_HZ
    return max(_SCORE_MIN, min(_SCORE_MAX, prev_score + delta))


# Re-exports for tests / debug window.
SCORE_MIN = _SCORE_MIN
SCORE_MAX = _SCORE_MAX
IN_PATH_THRESHOLD: float = 0.0
